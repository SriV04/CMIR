"""Sched-IR scheduler — Phase 1 (BIND).

Reads the resource YAML, walks an unscheduled Sched-IR graph (produced by
``decomposer.py``), and for every vertex:

* picks the first kernel that claims the vertex's primitive and whose
  constraints are satisfied by ``op_params``;
* writes ``kernel_type`` + a fresh ``kernel_instance`` id;
* runs the kernel's ``cost_query`` (against the original Keras model for
  weight access), normalizes it to the full kernel-result shape, and stores
  both ``cost`` and the richer precision payload on the vertex.

It does **not** touch any folding (`fold_*`) or timing (`t_*`) fields —
those belong to later phases.

Usage::

    g_sched_bound = bind(g_sched, keras_model, "IR/Sched-IR/da4ml-resource.yaml")
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import sys

import yaml

from heterograph import HGraph


# --------------------------------------------------------------------------- #
# Sibling-module load (IR/Sched-IR has a hyphen, so we go via importlib)
# --------------------------------------------------------------------------- #

def _load_sibling(name: str):
    here = Path(__file__).resolve().parent
    full_name = f"_sched_ir_{name}"
    spec = importlib.util.spec_from_file_location(full_name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod  # required for dataclasses / introspection
    spec.loader.exec_module(mod)
    return mod


_kernels = _load_sibling("kernels")
_da4ml = _load_sibling("_da4ml")
_kernel_result = _load_sibling("kernel_result")
WeightProvider = _kernels.WeightProvider
REGISTRY: dict[str, Callable[..., dict[str, Any]]] = _kernels.REGISTRY


# --------------------------------------------------------------------------- #
# fpga-config normalisation (resolves "auto" values from the resource YAML)
# --------------------------------------------------------------------------- #

def normalize_fpga(fpga: dict[str, Any]) -> dict[str, Any]:
    """Resolve `latency_cutoff: auto` (and similar) in the fpga config.

    Mutates a copy and returns it. Idempotent — calling on an already-resolved
    dict is a no-op. The same function is used by `evaluate.py` so that path A
    (end-to-end ground truth) and path B (per-layer Sched-IR) read the same
    `latency_cutoff` from the same resource YAML.
    """
    fpga = dict(fpga or {})

    cutoff = fpga.get("latency_cutoff", "auto")
    if isinstance(cutoff, str) and cutoff.strip().lower() == "auto":
        derived = _da4ml.derive_latency_cutoff(
            target_fmax_hz=float(fpga.get("target_fmax_hz") or 0.0),
            t_logic_ns=float(fpga.get("t_logic_ns") or 1.00),
            routing_margin=float(fpga.get("routing_margin") or 0.30),
        )
        fpga["latency_cutoff"] = int(derived)
    else:
        try:
            fpga["latency_cutoff"] = int(cutoff)
        except (TypeError, ValueError):
            fpga["latency_cutoff"] = -1

    return fpga


# --------------------------------------------------------------------------- #
# Kernel-library construction
# --------------------------------------------------------------------------- #

class KernelEntry:
    """One row of the resource YAML's ``kernels:`` mapping."""

    __slots__ = ("name", "primitives", "constraints", "cost_query_name",
                 "inserted_by", "instances", "max_instances")

    def __init__(self, name, primitives, constraints, cost_query_name,
                 inserted_by, instances, max_instances):
        self.name = name
        self.primitives = primitives
        self.constraints = constraints
        self.cost_query_name = cost_query_name
        self.inserted_by = inserted_by
        self.instances = instances
        self.max_instances = max_instances

    def cost_query(self, p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
        fn = REGISTRY[self.cost_query_name]
        return fn(p, weights, fpga)


def build_kernel_library(cfg: dict) -> dict[str, list[KernelEntry]]:
    """Group kernel entries by the primitive they support."""
    library: dict[str, list[KernelEntry]] = {}
    for kname, spec in (cfg.get("kernels") or {}).items():
        entry = KernelEntry(
            name=kname,
            primitives=tuple(spec.get("supported_ops") or ()),
            constraints=dict(spec.get("constraints") or {}),
            cost_query_name=spec.get("cost_query"),
            inserted_by=spec.get("inserted_by"),
            instances=str(spec.get("instances") or "unlimited"),
            max_instances=spec.get("max_instances"),
        )
        if entry.cost_query_name and entry.cost_query_name not in REGISTRY:
            raise ValueError(
                f"Resource YAML kernel {kname!r} references unknown cost_query "
                f"{entry.cost_query_name!r}; add it to kernels.REGISTRY"
            )
        for prim in entry.primitives:
            library.setdefault(prim, []).append(entry)
    return library


def _select_kernel(p: dict, candidates: list[KernelEntry]) -> KernelEntry:
    """Return the first kernel whose constraints are satisfied by `p`.

    Skips scheduler-inserted kernels — those are only chosen during the
    later INSERT-INFRASTRUCTURE phase, never during BIND of decomposer-emitted
    vertices.
    """
    op_params = p.get("op_params") or {}
    for entry in candidates:
        if entry.inserted_by == "scheduler" and p.get("inserted_by") != "scheduler":
            continue
        if _constraints_ok(entry.constraints, op_params):
            return entry
    raise ValueError(
        f"no kernel in the resource YAML matches vertex "
        f"{p.get('nn_layer_name')!r} (op={p.get('op')!r}, op_params={op_params})"
    )


def _constraints_ok(constraints: dict[str, Any], op_params: dict[str, Any]) -> bool:
    for key, want in constraints.items():
        if key == "weight_source":
            # We only ever produce constant-weight kernels at BIND time.
            if want not in ("constant", None):
                return False
            continue
        if key == "max_depth":
            depth = op_params.get("depth")
            if depth is not None and depth > want:
                return False
            continue
        if key == "min_depth":
            depth = op_params.get("depth")
            if depth is not None and depth < want:
                return False
            continue
        # Unknown constraints: fail closed so we notice misspellings.
        return False
    return True


# --------------------------------------------------------------------------- #
# BIND entry point
# --------------------------------------------------------------------------- #

_NEEDS_BIND = ("dense", "reduce", "elementwise", "activation")


def _first_not_none(*vals):
    for value in vals:
        if value is not None:
            return value
    return None


def _sync_result_to_op_params(p: dict, result: dict) -> None:
    params = p.get("op_params") or {}
    out_qints = result.get("output_qints")
    out_kifs = result.get("output_kifs")

    if out_qints:
        params["output_qint"] = out_qints[0] if len(out_qints) == 1 else out_qints
    if out_kifs:
        params["output_kif"] = out_kifs[0] if len(out_kifs) == 1 else out_kifs
        bits = [int(k["bits"]) for k in out_kifs if k is not None and k.get("bits") is not None]
        if bits:
            params["out_bw"] = bits[0] if len(set(bits)) == 1 else max(bits)


def _apply_kernel_result(p: dict, result: dict) -> None:
    p["cost"] = result["cost"]
    p["kernel_result"] = result

    if result.get("input_qints") is not None:
        p["input_qints"] = result["input_qints"]
    if result.get("input_kifs") is not None:
        p["input_kifs"] = result["input_kifs"]
    if result.get("output_qints") is not None:
        p["output_qints"] = result["output_qints"]
    if result.get("output_kifs") is not None:
        p["output_kifs"] = result["output_kifs"]

    p["input_tensor_width_bits"] = _first_not_none(
        result.get("input_tensor_width_bits"),
        p.get("input_tensor_width_bits"),
    )
    p["output_tensor_width_bits"] = _first_not_none(
        result.get("output_tensor_width_bits"),
        p.get("output_tensor_width_bits"),
    )
    p["precision_source"] = _first_not_none(
        result.get("precision_source"),
        p.get("precision_source"),
        "unknown",
    )
    _sync_result_to_op_params(p, result)


def bind(
    g_sched: HGraph,
    keras_model,
    resource_yaml_path: str | Path,
) -> HGraph:
    """Phase 1 — assign a kernel + cost to every Sched-IR vertex.

    Mutates ``g_sched`` in place and returns it.
    """
    cfg_path = Path(resource_yaml_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    fpga = normalize_fpga(cfg.get("fpga") or {})
    cfg["fpga"] = fpga   # write back so downstream loaders see the resolved values
    library = build_kernel_library(cfg)
    weights = WeightProvider(keras_model)

    g_sched.pmap["resource_yaml"] = str(cfg_path)
    g_sched.pmap["target_device"] = fpga.get("device")
    g_sched.pmap["fpga_config"]   = fpga   # cache for downstream phases (folder, evaluate)

    next_instance: dict[str, int] = {}

    for vx in g_sched.vertices:
        p = g_sched.pmap[vx]
        prim = p.get("op")
        if prim not in _NEEDS_BIND:
            # buffer / mux are scheduler-inserted; nothing to bind here.
            continue

        candidates = library.get(prim) or []
        if not candidates:
            raise ValueError(f"resource YAML has no kernel for primitive {prim!r}")

        chosen = _select_kernel(p, candidates)
        raw_result = chosen.cost_query(p, weights, fpga)
        result = _kernel_result.normalize_kernel_result(raw_result, source="closed_form")

        p["kernel_type"]     = chosen.name
        p["kernel_instance"] = next_instance.setdefault(chosen.name, 0)
        next_instance[chosen.name] += 1
        _apply_kernel_result(p, result)

    _validate_bind(g_sched)
    return g_sched


def _topo_order(g: HGraph) -> list[int]:
    indeg = {v: len(g.in_vx(v)) for v in g.vertices}
    queue = [v for v, d in indeg.items() if d == 0]
    order: list[int] = []
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        order.append(v)
        for u in g.out_vx(v):
            indeg[u] -= 1
            if indeg[u] == 0:
                queue.append(u)
    if len(order) != len(g.vertices):
        raise ValueError("Sched-IR graph is cyclic; cannot topologically bind/propagate")
    return order


def _edge_input_precision(ep: dict) -> tuple[Any | None, Any | None, Any | None]:
    qint = _first_not_none(ep.get("src_qint"), ep.get("element_qint"), ep.get("dst_qint"), ep.get("qint"))
    kif = _first_not_none(ep.get("src_kif"), ep.get("element_kif"), ep.get("dst_kif"), ep.get("kif"))
    width = _first_not_none(ep.get("tensor_width_bits"), ep.get("volume_bits_exact"), ep.get("volume_bits"))
    return qint, kif, width


def _ingest_inputs_from_edges(g: HGraph, vx: int) -> None:
    p = g.pmap[vx]
    preds = g.in_vx(vx)
    if not preds:
        return

    in_qints = []
    in_kifs = []
    in_widths = []
    for u in preds:
        ep = g.pmap[(u, vx)]
        qint, kif, width = _edge_input_precision(ep)
        in_qints.append(qint)
        in_kifs.append(kif)
        in_widths.append(width)

    if any(x is not None for x in in_qints):
        p["input_qints"] = in_qints
    if any(x is not None for x in in_kifs):
        p["input_kifs"] = in_kifs
    if any(x is not None for x in in_widths):
        p["input_tensor_width_bits"] = in_widths

    params = p.get("op_params") or {}
    op = p.get("op")
    if op == "elementwise":
        params["input_qints"] = p.get("input_qints")
        params["input_kifs"] = p.get("input_kifs")
    elif op in ("dense", "reduce", "activation"):
        if p.get("input_qints"):
            params["input_qint"] = p["input_qints"][0]
        if p.get("input_kifs"):
            params["input_kif"] = p["input_kifs"][0]


def _propagate_outputs_to_edges(g: HGraph, vx: int) -> None:
    p = g.pmap[vx]
    out_qints = p.get("output_qints")
    out_kifs = p.get("output_kifs")
    out_width = p.get("output_tensor_width_bits")

    if out_kifs:
        bits = [int(k["bits"]) for k in out_kifs if k is not None and k.get("bits") is not None]
        element_bw = bits[0] if bits and len(set(bits)) == 1 else None
        legacy_bw = max(bits) if bits else None
    else:
        element_bw = None
        legacy_bw = None

    for v in g.out_vx(vx):
        ep = g.pmap[(vx, v)]
        if out_qints is not None:
            ep["src_qint"] = out_qints
        if out_kifs is not None:
            ep["src_kif"] = out_kifs
        if element_bw is not None:
            ep["src_bitwidth_bits"] = float(element_bw)
            ep["element_bitwidth_bits"] = float(element_bw)
        if out_width is not None:
            ep["tensor_width_bits"] = float(out_width)
            ep["volume_bits_exact"] = float(out_width)
            ep["volume_bits"] = float(out_width)
        if legacy_bw is not None:
            ep["bitwidth"] = float(legacy_bw)


def bind_and_propagate(
    g_sched: HGraph,
    keras_model,
    resource_yaml_path: str | Path,
) -> HGraph:
    """Phase 1 (variant) — topological BIND + node/edge precision propagation.

    Requires that fold-plan metadata (`parallelism_N/lanes_P/temporal_steps_T`
    and `reduce_mode`) has already been stamped if you want fold-aware reduce
    evaluation.
    """
    cfg_path = Path(resource_yaml_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    fpga = normalize_fpga(cfg.get("fpga") or {})
    cfg["fpga"] = fpga
    library = build_kernel_library(cfg)
    weights = WeightProvider(keras_model)

    g_sched.pmap["resource_yaml"] = str(cfg_path)
    g_sched.pmap["target_device"] = fpga.get("device")
    g_sched.pmap["fpga_config"] = fpga

    next_instance: dict[str, int] = {}

    for vx in _topo_order(g_sched):
        p = g_sched.pmap[vx]
        prim = p.get("op")
        if prim not in _NEEDS_BIND:
            continue

        _ingest_inputs_from_edges(g_sched, vx)

        candidates = library.get(prim) or []
        if not candidates:
            raise ValueError(f"resource YAML has no kernel for primitive {prim!r}")

        chosen = _select_kernel(p, candidates)
        raw_result = chosen.cost_query(p, weights, fpga)
        result = _kernel_result.normalize_kernel_result(raw_result, source="closed_form")

        p["kernel_type"] = chosen.name
        p["kernel_instance"] = next_instance.setdefault(chosen.name, 0)
        next_instance[chosen.name] += 1
        _apply_kernel_result(p, result)

        _propagate_outputs_to_edges(g_sched, vx)

    _validate_bind(g_sched)
    return g_sched


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

_REQUIRED_COST_KEYS = ("lut", "ff", "dsp", "bram", "latency_cycles", "ii")


def _validate_bind(g_sched: HGraph) -> None:
    for vx in g_sched.vertices:
        p = g_sched.pmap[vx]
        prim = p.get("op")
        if prim not in _NEEDS_BIND:
            continue
        if p.get("kernel_type") is None:
            raise ValueError(f"BIND left vertex {vx} ({p.get('nn_layer_name')!r}) without a kernel_type")
        if p.get("kernel_instance") is None:
            raise ValueError(f"BIND left vertex {vx} without a kernel_instance")
        cost = p.get("cost")
        if not isinstance(cost, dict):
            raise ValueError(f"BIND left vertex {vx} without a cost dict")
        missing = [k for k in _REQUIRED_COST_KEYS if k not in cost]
        if missing:
            raise ValueError(f"BIND vertex {vx} cost dict missing keys: {missing}")
        if p.get("kernel_result") is None:
            raise ValueError(f"BIND left vertex {vx} without kernel_result metadata")
