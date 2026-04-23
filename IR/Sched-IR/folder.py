"""Sched-IR Phase 2 — FOLD (N–P–T timing model).

Takes a *bound* Sched-IR graph (i.e. one that has already been through
``scheduler.bind``) and a folding policy expressed as either

* ``factor`` — requested temporal-step count T (legacy parameter name), or
* ``lanes``  — hardware lanes P per group,

then populates the N–P–T timing fields on every vertex:

    parallelism_N        — fold-axis size
    lanes_P              — hardware parallel lanes (= elements_per_cycle)
    temporal_steps_T     — ceil(N / P)
    pipeline_latency_L   — intrinsic kernel pipeline depth
    ii                   — initiation interval = T
    latency_total        — L + (T - 1)
    elements_per_cycle   — = P

The invariants ``II == ceil(N / P)`` and ``latency_total == L + (II - 1)``
are enforced by ``_validate_fold`` at the end of this pass.

Flow:

1. Identify **fold groups** — sets of vertices that share the same
   `fold_axes` and are connected by an edge whose tensor carries that axis
   with concrete size > 1. The closure ignores edges whose fold-axis dim
   has collapsed to 1 (broadcast inputs and reduction outputs naturally
   exit the group).
2. For each group, derive ``N`` = the concrete fold-axis dim shared by the
   group's edges (e.g. N=8 for JEDI-linear's particle axis).
3. Derive P from the folding policy and snap T := ceil(N / P). Write
   (N, P, T, L, II=T, latency_total) onto every group member. Legacy
   ``fold_factor`` and ``physical_instances`` fields are kept in sync
   (== T and == P respectively) for downstream consumers that haven't
   migrated.
4. **Temporalise reductions** that consume the folded axis: their
   ``reduce_mode`` flips from ``'spatial'`` to ``'temporal_accumulate'``
   (when P = 1) or ``'hybrid'`` (1 < P < N), and their ``cost`` is
   recomputed via ``da4ml_reduce_temporal_cost``. L is refreshed from the
   new cost and (T, latency_total, ii) are recomputed so the invariants
   still hold.
5. Vertices outside any fold group get (N=1, P=1, T=1, ii=1,
   latency_total=L).
6. Populate ``g_sched.pmap['fold_plan']`` with per-group metadata.

The folder mutates the graph in place.
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

def fold(
    g_sched: HGraph,
    *,
    factor: int | None = None,
    lanes: int | None = None,
) -> HGraph:
    """Phase 2 — apply an N–P–T folding policy to every fold group.

    The policy can be expressed in two equivalent ways:

    * ``factor`` — the legacy parameter: interpreted as the *requested*
      temporal-step count T. Per-group we derive P = ceil(N / factor), then
      snap T := ceil(N / P) so the invariant T == ceil(N / P) always holds.
    * ``lanes``  — directly specifies P, and T is derived.

    Exactly one of the two must be given. ``factor=1`` (or ``lanes=N``) is a
    valid no-op call: every vertex is labelled with T=1, P=N, ``ii=1``, and
    reduction modes stay ``'spatial'`` — useful for the baseline pass.
    """
    if (factor is None) == (lanes is None):
        raise ValueError("fold(...) requires exactly one of factor= or lanes=")
    if factor is not None and factor < 1:
        raise ValueError(f"fold factor must be >= 1, got {factor}")
    if lanes is not None and lanes < 1:
        raise ValueError(f"fold lanes must be >= 1, got {lanes}")

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

        # Resolve P (hardware lanes) from the folding policy, then snap
        # T := ceil(N / P) so the invariant T == ceil(N / P) holds even
        # when the user picks a T that doesn't divide N.
        if lanes is not None:
            P = max(1, min(int(lanes), int(N)))
        else:
            T_req = max(1, min(int(factor), int(N)))
            P = max(1, math.ceil(int(N) / T_req))
        T = max(1, math.ceil(int(N) / P))

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
            "parallelism": int(N),          # N
            "lanes": int(P),                # P
            "temporal_steps": int(T),       # T = ceil(N / P)
            "factor": int(T),               # legacy alias for T
            "physical_instances": int(P),   # legacy alias for P
            "members": members,
            "reductions_temporalised": temporalised,
        }
        fold_plan.append(info)
        for m in members:
            group_info_by_member[m] = info

    # ---------------------------------------------------------------- #
    # 3. Apply N–P–T timing to vertices
    # ---------------------------------------------------------------- #
    for vx in g_sched.vertices:
        p = g_sched.pmap[vx]

        if p.get("op") in ("buffer", "mux"):
            # Scheduler-inserted infrastructure is not a compute node —
            # leave it to infrastructure.py to set timing on.
            continue

        info = group_info_by_member.get(vx)

        if info is None:
            # Outside any fold group: trivially N=P=T=1.
            _apply_timing(p, N=1, P=1, T=1, group_id=None)
            continue

        N = info["parallelism"]
        P = info["lanes"]
        T = info["temporal_steps"]

        # Reductions consuming the fold axis: recompute cost for the
        # spatial/temporal layout dictated by (N, P, T), then flip mode.
        if vx in info["reductions_temporalised"] and T > 1:
            new_cost = _kernels.da4ml_reduce_temporal_cost(
                p, _NoWeights(), fpga,
                parallelism=N,
                factor=T,
            )
            p["cost"] = new_cost
            p["reduce_mode"] = "temporal_accumulate" if P == 1 else "hybrid"

        _apply_timing(p, N=N, P=P, T=T, group_id=info["group_id"])

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
# N–P–T application helper
# --------------------------------------------------------------------------- #

def _apply_timing(p: dict, *, N: int, P: int, T: int, group_id: int | None) -> None:
    """Write the full N–P–T timing record onto a vertex pmap.

    Enforces T == ceil(N / P), II == T, latency_total == L + (T - 1).
    Keeps the legacy aliases (fold_factor, physical_instances, cost.ii) in
    sync so callers that haven't migrated still see consistent values.
    """
    if T != math.ceil(N / max(P, 1)):
        raise ValueError(
            f"_apply_timing: T={T} violates T == ceil(N/P) for N={N}, P={P}"
        )

    cost = dict(p.get("cost") or {})
    L = int(cost.get("latency_cycles") or 1)
    cost["latency_cycles"] = L
    cost["ii"]             = T
    p["cost"] = cost

    p["parallelism_N"]      = int(N)
    p["lanes_P"]            = int(P)
    p["temporal_steps_T"]   = int(T)
    p["pipeline_latency_L"] = int(L)
    p["elements_per_cycle"] = int(P)
    p["ii"]                 = int(T)
    p["latency_total"]      = int(L + (T - 1))

    # Legacy aliases.
    p["fold_factor"]        = int(T)
    p["physical_instances"] = int(P)
    p["fold_group"]         = group_id


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _validate_fold(g: HGraph) -> None:
    for vx in g.vertices:
        p = g.pmap[vx]
        if p.get("op") in ("buffer", "mux"):
            continue

        # Presence of the N–P–T fields.
        for field in (
            "parallelism_N", "lanes_P", "temporal_steps_T",
            "pipeline_latency_L", "ii", "latency_total", "elements_per_cycle",
        ):
            if p.get(field) is None:
                raise ValueError(f"FOLD left vertex {vx} without {field}")

        N = int(p["parallelism_N"])
        P = int(p["lanes_P"])
        T = int(p["temporal_steps_T"])
        L = int(p["pipeline_latency_L"])

        # Hard invariants from the N–P–T model.
        if not (1 <= P <= max(N, 1)):
            raise ValueError(f"FOLD vertex {vx}: P={P} not in [1, N]={N}")
        if T != math.ceil(N / max(P, 1)):
            raise ValueError(
                f"FOLD vertex {vx}: T={T} != ceil(N/P)=ceil({N}/{P})={math.ceil(N / max(P, 1))}"
            )
        if int(p["ii"]) != T:
            raise ValueError(f"FOLD vertex {vx}: ii={p['ii']} != T={T}")
        if int(p["latency_total"]) != L + (T - 1):
            raise ValueError(
                f"FOLD vertex {vx}: latency_total={p['latency_total']} != L+(T-1)={L + (T - 1)}"
            )
        if int(p["elements_per_cycle"]) != P:
            raise ValueError(
                f"FOLD vertex {vx}: elements_per_cycle={p['elements_per_cycle']} != P={P}"
            )

        # Legacy alias consistency.
        if int(p.get("fold_factor") or 0) != T:
            raise ValueError(f"FOLD vertex {vx}: fold_factor alias != T")
        if int(p.get("physical_instances") or 0) != P:
            raise ValueError(f"FOLD vertex {vx}: physical_instances alias != P")
        cost = p.get("cost") or {}
        if int(cost.get("ii") or 0) != T:
            raise ValueError(f"FOLD vertex {vx}: cost.ii != T")

    # All members of a group must agree on (N, P, T).
    for entry in g.pmap.get("fold_plan") or []:
        T = entry["temporal_steps"]
        P = entry["lanes"]
        N = entry["parallelism"]
        for m in entry["members"]:
            pm = g.pmap[m]
            if pm.get("op") in ("buffer", "mux"):
                continue
            if (int(pm["temporal_steps_T"]), int(pm["lanes_P"]), int(pm["parallelism_N"])) != (T, P, N):
                raise ValueError(
                    f"FOLD inconsistent: vertex {m} in group {entry['group_id']} "
                    f"has (N,P,T)={(pm['parallelism_N'], pm['lanes_P'], pm['temporal_steps_T'])}, "
                    f"group has {(N, P, T)}"
                )
