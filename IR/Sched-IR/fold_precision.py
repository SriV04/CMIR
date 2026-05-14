"""Sched-IR Phase 2b — fold-aware precision and cost correction.

This pass runs after fold planning, BIND, and precision propagation, but
before timing is derived from costs. It rewrites nodes whose implementation
changes under folding; currently that means reductions consuming folded axes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from heterograph import HGraph


def _load_sibling(name: str):
    here = Path(__file__).resolve().parent
    full_name = f"_sched_ir_{name}"
    spec = importlib.util.spec_from_file_location(full_name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


_kernels = _load_sibling("kernels")
_kernel_result = _load_sibling("kernel_result")
_precision = _load_sibling("precision")


class _NoWeights:
    def get_kernel(self, layer_name):
        return None


def _is_folded_reduce(p: dict) -> bool:
    return (
        p.get("op") == "reduce"
        and p.get("reduce_mode") in ("temporal_accumulate", "hybrid")
        and int(p.get("temporal_steps_T") or 1) > 1
    )


def _sync_result_to_reduce(p: dict, result: dict) -> None:
    params = p.get("op_params") or {}
    p["op_params"] = params
    meta = result.get("kernel_meta") or {}

    params["reduce_mode"] = p.get("reduce_mode")
    params["spatial_width_P"] = int(p.get("lanes_P") or 1)
    params["temporal_steps_T"] = int(p.get("temporal_steps_T") or 1)

    if meta.get("partial_sum_qint") is not None:
        params["partial_sum_qint"] = meta["partial_sum_qint"]
    if meta.get("partial_sum_kif") is not None:
        params["partial_sum_kif"] = meta["partial_sum_kif"]
    if meta.get("accumulator_qint") is not None:
        params["accumulator_qint"] = meta["accumulator_qint"]
        params["output_qint"] = meta["accumulator_qint"]
    if meta.get("accumulator_kif") is not None:
        params["accumulator_kif"] = meta["accumulator_kif"]
        params["output_kif"] = meta["accumulator_kif"]

    p["cost"] = result["cost"]
    p["kernel_result"] = result
    p["output_qints"] = result.get("output_qints")
    p["output_kifs"] = result.get("output_kifs")
    p["output_tensor_width_bits"] = result.get("output_tensor_width_bits")
    p["precision_source"] = result.get("precision_source") or "fold_aware_derived"


def _recompute_folded_reduce(p: dict, fpga: dict) -> dict:
    N = int(p.get("parallelism_N") or 1)
    T = int(p.get("temporal_steps_T") or 1)
    raw = _kernels.da4ml_reduce_folded_result(
        p,
        _NoWeights(),
        fpga,
        parallelism=N,
        factor=T,
    )
    return _kernel_result.normalize_kernel_result(raw, source="fold_aware_derived")


def _successors(g: HGraph, vx: int) -> list[int]:
    if hasattr(g, "out_vx"):
        return list(g.out_vx(vx))
    return [dst for src, dst in g.edges if src == vx]


def validate_fold_aware_precision(g_sched: HGraph, *, strict: bool = False) -> list[str]:
    warnings: list[str] = []

    for vx in g_sched.vertices:
        p = g_sched.pmap[vx]
        if not _is_folded_reduce(p):
            continue

        params = p.get("op_params") or {}
        if int(p.get("parallelism_N") or 1) <= 1:
            warnings.append(f"folded reduce {vx} has parallelism_N <= 1")
        if int(p.get("temporal_steps_T") or 1) <= 1:
            warnings.append(f"folded reduce {vx} has temporal_steps_T <= 1")
        if int(p.get("lanes_P") or 1) > 1 and params.get("partial_sum_kif") is None:
            warnings.append(f"folded reduce {vx} has no partial_sum_kif")
        if params.get("accumulator_kif") is None:
            warnings.append(f"folded reduce {vx} has no accumulator_kif")
        if not p.get("output_kifs"):
            warnings.append(f"folded reduce {vx} has no output_kifs")

        cost = p.get("cost") or {}
        if int(cost.get("ii") or 0) != int(p.get("temporal_steps_T") or 1):
            warnings.append(f"folded reduce {vx} cost.ii != temporal_steps_T")
        if int(cost.get("latency_cycles") or 0) < 1:
            warnings.append(f"folded reduce {vx} latency_cycles < 1")

        for dst in _successors(g_sched, vx):
            ep = g_sched.pmap[(vx, dst)]
            if ep.get("src_kif") != p.get("output_kifs"):
                warnings.append(f"edge ({vx}, {dst}) does not carry folded reduce output_kifs")

    if strict and warnings:
        raise ValueError("fold-aware precision validation failed:\n" + "\n".join(warnings))
    return warnings


def apply_fold_aware_precision(g_sched: HGraph, *, strict: bool = False) -> HGraph:
    """Recompute precision/cost for nodes whose implementation changes after folding."""
    warnings: list[str] = []
    fpga = g_sched.pmap.get("fpga_config") or {}

    for vx in list(g_sched.vertices):
        p = g_sched.pmap[vx]
        if not _is_folded_reduce(p):
            continue

        result = _recompute_folded_reduce(p, fpga)
        meta = result.get("kernel_meta") or {}
        if meta.get("precision_warning"):
            warnings.append(f"vertex {vx} ({p.get('nn_layer_name')}): {meta['precision_warning']}")
        _sync_result_to_reduce(p, result)

    _precision.propagate_precision(g_sched, strict=strict)
    warnings.extend(validate_fold_aware_precision(g_sched, strict=strict))

    g_sched.pmap["fold_aware_precision_applied"] = True
    g_sched.pmap["fold_aware_precision_warnings"] = warnings or None
    return g_sched
