"""Sched-IR Phase 2 — FOLD.

Takes a *bound* Sched-IR graph (i.e. one that has already been through
``scheduler.bind``) and a global fold factor ``K``, then:

1. Identifies **fold groups** — sets of vertices that share the same
   `fold_axes` and are connected by an edge whose tensor carries that axis
   with concrete size > 1. The closure ignores edges whose fold-axis dim
   has collapsed to 1 (broadcast inputs and reduction outputs naturally
   exit the group).
2. For each group, derives ``parallelism`` = the concrete fold-axis dim
   shared by the group's edges (e.g. N=8 for JEDI-linear's particle axis).
3. Applies the user-supplied global ``factor`` (clamped per-group to
   ``min(factor, parallelism)``), writing
   ``fold_factor`` / ``fold_group`` / ``physical_instances`` / ``cost.ii``
   onto every group member.
4. **Temporalises reductions** that consume the folded axis: their
   ``reduce_mode`` flips from ``'spatial'`` to ``'temporal_accumulate'``
   (when K = N) or ``'hybrid'`` (1 < K < N), and their ``cost`` is
   recomputed via ``da4ml_reduce_temporal_cost`` in the kernel library.
5. Vertices outside any fold group keep ``fold_factor = 1`` and
   ``physical_instances = 1``.
6. Populates ``g_sched.pmap['fold_plan']`` with per-group metadata.

The folder mutates the graph in place. To compare two K's, run BIND twice
on independently decomposed graphs and call FOLD on each.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import yaml

from heterograph import HGraph


# --------------------------------------------------------------------------- #
# Sibling-module load (IR/Sched-IR has a hyphen)
# --------------------------------------------------------------------------- #

def _load_sibling(name: str):
    here = Path(__file__).resolve().parent
    full_name = f"_sched_ir_{name}"
    spec = importlib.util.spec_from_file_location(full_name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


_kernels = _load_sibling("kernels")


# --------------------------------------------------------------------------- #
# Union-find helpers for fold-group closure
# --------------------------------------------------------------------------- #

class _UF:
    def __init__(self, items):
        self._parent = {x: x for x in items}

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict[int, list]:
        out: dict[int, list] = {}
        for x in self._parent:
            r = self.find(x)
            out.setdefault(r, []).append(x)
        return out


# --------------------------------------------------------------------------- #
# Edge-level fold closure
# --------------------------------------------------------------------------- #

def _shared_fold_axis(p_src: dict, p_dst: dict, edge_shape: tuple | None) -> int | None:
    """Return the (single) fold axis carried by this edge, or None.

    A fold-group constraint exists when both endpoints declare a fold axis,
    the axis index is the same on both sides, and the edge's tensor has that
    dim with concrete size > 1. Broadcast / reduced edges naturally fail
    the size > 1 test and so don't constrain anything.
    """
    fa_s = p_src.get("fold_axes") or []
    fa_d = p_dst.get("fold_axes") or []
    if not fa_s or not fa_d or edge_shape is None:
        return None
    common = set(fa_s) & set(fa_d)
    for ax in common:
        if 0 <= ax < len(edge_shape):
            d = edge_shape[ax]
            if d not in (None, 1):
                return ax
    return None


def _group_parallelism(g: HGraph, members: list[int]) -> int | None:
    """Look up the largest concrete fold-axis dim seen on any edge of the group.

    All in-group edges should report the same N once the closure converges,
    but we max-reduce defensively against malformed inputs.
    """
    best = None
    member_set = set(members)
    for u in members:
        for v in g.out_vx(u):
            if v not in member_set:
                continue
            ep = g.pmap[(u, v)]
            shape = ep.get("tensor_shape")
            p_u = g.pmap[u]
            p_v = g.pmap[v]
            ax = _shared_fold_axis(p_u, p_v, shape)
            if ax is None:
                continue
            d = shape[ax]
            if d in (None, 1):
                continue
            best = max(best or 0, int(d))
    return best


# --------------------------------------------------------------------------- #
# Reduction-axis intersection (for temporalisation)
# --------------------------------------------------------------------------- #

def _reduce_consumes_fold_axis(p: dict, fold_axes: list[int] | None) -> bool:
    if p.get("op") != "reduce" or not fold_axes:
        return False
    op_params = p.get("op_params") or {}
    axes = op_params.get("axes") or []
    return bool(set(axes) & set(fold_axes))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def fold(g_sched: HGraph, *, factor: int) -> HGraph:
    """Phase 2 — apply a global fold factor to every fold group.

    `factor=1` is a valid no-op call: every vertex is still labelled with
    ``fold_factor=1``, ``physical_instances`` is set, ``fold_plan`` is
    populated, and reduction modes are left as ``'spatial'``. Useful for
    making the K=1 baseline look structurally identical to the K>1 outputs
    in downstream passes / visualisation.
    """
    if factor < 1:
        raise ValueError(f"fold factor must be >= 1, got {factor}")

    # Reload the resource YAML stamped on the graph by BIND to recover fpga
    # config + the cost-query callables. fpga is needed by the temporal-
    # reduce cost query.
    yaml_path = g_sched.pmap.get("resource_yaml")
    if yaml_path is None:
        raise ValueError("fold() requires bind() to have run first (no resource_yaml on the graph)")
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    fpga = cfg.get("fpga") or {}

    # ---------------------------------------------------------------- #
    # 1. Build fold groups via union-find over constraint edges
    # ---------------------------------------------------------------- #
    uf = _UF(list(g_sched.vertices))
    in_any_group: set[int] = set()

    for u, v in g_sched.edges:
        ep = g_sched.pmap[(u, v)]
        ax = _shared_fold_axis(g_sched.pmap[u], g_sched.pmap[v], ep.get("tensor_shape"))
        if ax is not None:
            uf.union(u, v)
            in_any_group.add(u)
            in_any_group.add(v)

    raw_groups = uf.groups()
    # Strip singletons that aren't actually in any constraint.
    groups: dict[int, list[int]] = {}
    next_gid = 0
    for members in raw_groups.values():
        actual = [m for m in members if m in in_any_group]
        if not actual:
            continue
        groups[next_gid] = sorted(actual)
        next_gid += 1

    # ---------------------------------------------------------------- #
    # 2. Per-group parallelism + effective K
    # ---------------------------------------------------------------- #
    fold_plan: list[dict[str, Any]] = []
    group_info_by_member: dict[int, dict[str, Any]] = {}

    for gid, members in groups.items():
        N = _group_parallelism(g_sched, members)
        if not N:
            continue
        K = max(1, min(int(factor), int(N)))
        physical_inst = math.ceil(N / K)

        # Group-level fold_axes = intersection of every member's fold_axes.
        common_axes: set[int] | None = None
        for m in members:
            ma = set(g_sched.pmap[m].get("fold_axes") or [])
            common_axes = ma if common_axes is None else (common_axes & ma)
        common_axes_list = sorted(common_axes or set())

        temporalised: list[int] = []
        for m in members:
            if _reduce_consumes_fold_axis(g_sched.pmap[m], common_axes_list):
                temporalised.append(m)

        info = {
            "group_id": gid,
            "axes": common_axes_list,
            "parallelism": int(N),
            "factor": K,
            "physical_instances": int(physical_inst),
            "members": members,
            "reductions_temporalised": temporalised,
        }
        fold_plan.append(info)
        for m in members:
            group_info_by_member[m] = info

    # ---------------------------------------------------------------- #
    # 3. Apply fold to vertices
    # ---------------------------------------------------------------- #
    for vx in g_sched.vertices:
        p = g_sched.pmap[vx]
        info = group_info_by_member.get(vx)

        if info is None:
            # Outside any fold group — single physical instance, K = 1.
            p["fold_factor"]        = 1
            p["fold_group"]         = None
            p["physical_instances"] = 1
            cost = p.get("cost") or {}
            if "ii" not in cost or cost["ii"] is None:
                cost["ii"] = 1
            p["cost"] = cost
            continue

        K = info["factor"]
        p["fold_factor"]        = K
        p["fold_group"]         = info["group_id"]
        p["physical_instances"] = info["physical_instances"]

        cost = dict(p.get("cost") or {})
        cost["ii"] = K
        p["cost"] = cost

        # Reductions consuming the fold axis: switch mode + recompute cost.
        if vx in info["reductions_temporalised"] and K > 1:
            new_cost = _kernels.da4ml_reduce_temporal_cost(
                p, _NoWeights(), fpga,
                parallelism=info["parallelism"],
                factor=K,
            )
            p["cost"] = new_cost
            p["reduce_mode"] = "temporal_accumulate" if K == info["parallelism"] else "hybrid"

    # ---------------------------------------------------------------- #
    # 4. Stamp the graph-level fold plan
    # ---------------------------------------------------------------- #
    g_sched.pmap["fold_plan"] = fold_plan

    _validate_fold(g_sched)
    return g_sched


# --------------------------------------------------------------------------- #
# Tiny stub WeightProvider for the temporal-reduce cost query (it never
# touches weights — pure shape/bw analysis through comb_trace).
# --------------------------------------------------------------------------- #

class _NoWeights:
    def get_kernel(self, layer_name):
        return None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _validate_fold(g: HGraph) -> None:
    for vx in g.vertices:
        p = g.pmap[vx]
        if p.get("op") in ("buffer", "mux"):
            continue
        if p.get("fold_factor") is None:
            raise ValueError(f"FOLD left vertex {vx} without fold_factor")
        if p.get("physical_instances") is None:
            raise ValueError(f"FOLD left vertex {vx} without physical_instances")
        cost = p.get("cost") or {}
        if cost.get("ii") is None:
            raise ValueError(f"FOLD left vertex {vx} without cost.ii")

    # All members of a group must agree on K.
    for entry in g.pmap.get("fold_plan") or []:
        K = entry["factor"]
        for m in entry["members"]:
            if g.pmap[m]["fold_factor"] != K:
                raise ValueError(
                    f"FOLD inconsistent: vertex {m} in group {entry['group_id']} "
                    f"has fold_factor {g.pmap[m]['fold_factor']}, group K={K}"
                )
