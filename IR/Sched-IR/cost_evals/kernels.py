"""Sched-IR kernel library.

Each function in this module is a `cost_query` named in
`da4ml-resource.yaml`. Given a Sched-IR vertex pmap and a `WeightProvider`,
it returns either a full kernel-result payload or a legacy canonical cost
dict ``{lut, ff, dsp, bram, latency_cycles, ii}``.

Cost queries fall into two groups:

1. **da4ml-driven** (`da4ml_*_cost`): build a tiny da4ml trace from the
   Sched-IR `op_params`, run `comb_trace` (or `cmvm.api.solve` for dense),
   and return the full `_da4ml.*_result(...)` payload.
2. **Closed-form** (`register_buffer_cost`, `bram_buffer_cost`,
   `lut_mux_cost`): scheduler-inserted infrastructure with no da4ml
   involvement.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


# --------------------------------------------------------------------------- #
# Sibling-module load (IR/Sched-IR has a hyphen, so we go via importlib)
# --------------------------------------------------------------------------- #

def _load_sibling(name: str):
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(f"_sched_ir_{name}", here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_da4ml = _load_sibling("_da4ml")


# --------------------------------------------------------------------------- #
# Weight provider
# --------------------------------------------------------------------------- #

class WeightProvider:
    """Minimal lookup of constant kernel weights by NN-IR layer name.

    For HGQ layers (QEinsumDenseBatchnorm / QEinsumDense / QDense /
    QBatchNormDense) the *quantized*, BN-folded kernel is exposed via the
    layer's ``qkernel`` property — that's exactly what da4ml's CMVM solver
    expects. For plain Keras Dense / EinsumDense we fall back to
    ``layer.kernel``.
    """

    def __init__(self, keras_model):
        self._model = keras_model
        self._cache: dict[str, np.ndarray] = {}

    def get_kernel(self, layer_name: str) -> np.ndarray | None:
        if layer_name in self._cache:
            return self._cache[layer_name]
        try:
            layer = self._model.get_layer(layer_name)
        except Exception:
            return None
        for attr in ("qkernel", "kernel"):
            obj = getattr(layer, attr, None)
            if obj is None:
                continue
            arr = np.array(obj)
            self._cache[layer_name] = arr
            return arr
        return None


# --------------------------------------------------------------------------- #
# da4ml-driven cost queries
# --------------------------------------------------------------------------- #

def _first_not_none(*vals):
    for value in vals:
        if value is not None:
            return value
    return None


def _as_2d_kernel(kernel) -> np.ndarray:
    kernel = np.asarray(kernel)
    if kernel.ndim != 2:
        kernel = kernel.reshape(-1, kernel.shape[-1])
    return kernel


def _single_input_precision_kwargs(op_params: dict) -> dict[str, Any]:
    return {
        "input_qints": [op_params.get("input_qint")],
        "input_kifs": [op_params.get("input_kif")],
        "input_bws": [float(op_params.get("in_bw"))] if op_params.get("in_bw") is not None else None,
    }


def _input_qints_for_kernel(op_params: dict, kernel: np.ndarray, *, context: str):
    qints = _da4ml.qints_from_precision_payload(
        op_params.get("input_qint"),
        op_params.get("input_kif"),
        fallback_bw=op_params.get("in_bw"),
        feature_count=kernel.shape[0],
        context=context,
    )

    if isinstance(qints, list):
        if len(qints) == 1:
            return qints * kernel.shape[0]
        if len(qints) == kernel.shape[0]:
            return qints
        if len(qints) == kernel.size:
            raise ValueError(
                f"{context}: got elementwise input precision "
                f"({len(qints)} qints) where per-input-feature precision ({kernel.shape[0]}) was required"
            )
        raise ValueError(
            f"{context}: expected 1 or {kernel.shape[0]} input qints, got {len(qints)}"
        )

    if qints is not None:
        return [qints] * kernel.shape[0]

    raise ValueError("dense has no input precision")

def da4ml_dense_cost(p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
    op_params = p.get("op_params") or {}
    kernel = _first_not_none(
        op_params.get("qkernel_values"),
        op_params.get("kernel_values"),
        weights.get_kernel(p["nn_layer_name"]),
    )
    if kernel is None:
        raise ValueError(
            f"dense vertex {p['nn_layer_name']!r}: no constant kernel found on the Keras model"
        )

    kernel = _as_2d_kernel(kernel)
    input_qints = _input_qints_for_kernel(
        op_params,
        kernel,
        context=f"dense vertex {p.get('nn_layer_name')!r}",
    )

    result = _da4ml.solve_dense_result(
        kernel,
        input_qints=input_qints,
        adder_size=int(fpga.get("adder_size", -1)),
        carry_size=int(fpga.get("carry_size", -1)),
        latency_cutoff=int(fpga.get("latency_cutoff", -1)),
    )
    result["kernel_meta"] = {
        "op": "dense",
        "kernel_shape": tuple(kernel.shape),
        "uses_qkernel": bool(op_params.get("uses_qkernel")),
        "kernel_sparsity": op_params.get("kernel_sparsity"),
    }
    return result


def da4ml_reduce_cost(p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
    op_params = p.get("op_params") or {}
    in_shape = op_params.get("in_shape")
    in_bw = op_params.get("in_bw")
    axes = op_params.get("axes")
    mode = (op_params.get("mode") or "sum").lower()
    keepdims = bool(op_params.get("keepdims"))
    reduce_mode = (op_params.get("reduce_mode") or p.get("reduce_mode") or "spatial").lower()

    if in_shape is None or in_bw is None or axes is None:
        raise ValueError(
            f"reduce vertex {p.get('nn_layer_name')!r}: missing in_shape/in_bw/axes"
        )

    # Drop the symbolic batch dim from the trace shape — we only need one
    # batch element to estimate cost/latency for the steady-state pipeline.
    trace_shape = tuple(d for d in in_shape[1:] if d is not None)
    axes_in_trace = tuple(a - 1 for a in axes if a >= 1)

    if reduce_mode in ("temporal_accumulate", "hybrid"):
        return da4ml_reduce_folded_result(
            p,
            weights,
            fpga,
            parallelism=int(p.get("parallelism_N") or 1),
            factor=int(p.get("temporal_steps_T") or 1),
        )

    if mode == "sum":
        body = lambda x: np.sum(x, axis=axes_in_trace, keepdims=keepdims)  # noqa: E731
    elif mode == "max":
        body = lambda x: np.amax(x, axis=axes_in_trace, keepdims=keepdims)  # noqa: E731
    elif mode == "min":
        body = lambda x: np.amin(x, axis=axes_in_trace, keepdims=keepdims)  # noqa: E731
    elif mode == "mean":
        body = lambda x: np.sum(x, axis=axes_in_trace, keepdims=keepdims)  # noqa: E731
    else:
        raise NotImplementedError(f"reduce mode {mode!r} not supported")

    result = _da4ml.trace_lambda_result(
        [trace_shape],
        body,
        **_single_input_precision_kwargs(op_params),
        adder_size=int(fpga.get("adder_size", -1)),
        carry_size=int(fpga.get("carry_size", -1)),
        latency_cutoff=int(fpga.get("latency_cutoff", -1)),
    )
    result["kernel_meta"] = {
        "op": "reduce",
        "mode": mode,
        "axes": axes,
        "reduction_width": op_params.get("reduction_width"),
        "reduce_mode": reduce_mode,
    }
    return result


def _as_list(value):
    if value is None:
        return None
    return value if isinstance(value, list) else [value]


def _qint_from_kif(kif: dict) -> dict:
    k = bool(kif.get("k"))
    i = int(kif.get("i") or 0)
    f = int(kif.get("f") or 0)
    step = 2.0 ** (-f)
    return {
        "min": -float(2**i) if k else 0.0,
        "max": float(2**i) - step,
        "step": step,
    }


def _kif_from_qint(qint: dict) -> dict:
    qmin = float(qint["min"])
    qmax = float(qint["max"])
    step = float(qint["step"])
    k = qmin < 0
    if step <= 0:
        f = 0
    else:
        f = max(int(math.ceil(-math.log2(step))), 0)
    positive_bound = qmax + step if qmax >= 0 else abs(qmax)
    bound = max(abs(qmin), positive_bound, 1.0)
    i = max(int(math.ceil(math.log2(bound))), 0)
    return {"k": k, "i": i, "f": f, "bits": int(k) + i + f}


def _sum_qint(qint: dict, width: int) -> dict:
    return {
        "min": float(width) * float(qint["min"]),
        "max": float(width) * float(qint["max"]),
        "step": float(qint["step"]),
    }


def _qint_values_from_payload(payload) -> list[dict] | None:
    values = _as_list(payload)
    if values is None:
        return None
    result = []
    for value in values:
        if not isinstance(value, dict):
            continue
        if {"min", "max", "step"}.issubset(value.keys()):
            result.append(value)
    return result or None


def _has_array_values(record: dict, keys: tuple[str, ...]) -> bool:
    return any(np.asarray(record[key]).ndim > 0 for key in keys)


def _conservative_array_qint(record: dict) -> dict:
    return {
        "min": float(np.min(np.asarray(record["min"], dtype=float))),
        "max": float(np.max(np.asarray(record["max"], dtype=float))),
        "step": float(np.min(np.asarray(record["step"], dtype=float))),
    }


def _conservative_array_kif(record: dict) -> dict:
    k = np.asarray(record["k"])
    i = np.asarray(record["i"], dtype=float)
    f = np.asarray(record["f"], dtype=float)
    step = np.power(2.0, -f)
    max_val = np.power(2.0, i) - step
    min_val = np.where(k.astype(bool), -np.power(2.0, i), 0.0)
    return {
        "min": float(np.min(min_val)),
        "max": float(np.max(max_val)),
        "step": float(np.min(step)),
    }


def _conservative_qint(qints: list[dict]) -> tuple[dict, str | None]:
    if len(qints) == 1 or all(q == qints[0] for q in qints):
        return qints[0], None
    return (
        {
            "min": min(float(q["min"]) for q in qints),
            "max": max(float(q["max"]) for q in qints),
            "step": min(float(q["step"]) for q in qints),
        },
        "used conservative folded-reduce precision for heterogeneous input qints",
    )


def _input_reduce_qint_and_kif(op_params: dict) -> tuple[dict, dict, str | None]:
    input_qint = op_params.get("input_qint")
    if isinstance(input_qint, dict) and {"min", "max", "step"}.issubset(input_qint.keys()):
        if _has_array_values(input_qint, ("min", "max", "step")):
            qint = _conservative_array_qint(input_qint)
            return qint, _kif_from_qint(qint), "used conservative folded-reduce precision for array input qint"

    qints = _qint_values_from_payload(input_qint)
    warning = None
    if qints:
        qint, warning = _conservative_qint(qints)
        return qint, _kif_from_qint(qint), warning

    input_kif = op_params.get("input_kif")
    if isinstance(input_kif, dict) and {"k", "i", "f"}.issubset(input_kif.keys()):
        if _has_array_values(input_kif, ("k", "i", "f")):
            qint = _conservative_array_kif(input_kif)
            return qint, _kif_from_qint(qint), "used conservative folded-reduce precision for array input kif"

    kifs = [k for k in (_as_list(input_kif) or []) if isinstance(k, dict)]
    complete = [k for k in kifs if all(k.get(key) is not None for key in ("k", "i", "f"))]
    if complete:
        if len(complete) == 1 or all(k == complete[0] for k in complete):
            kif = dict(complete[0])
            kif.setdefault("bits", int(bool(kif["k"])) + int(kif["i"]) + int(kif["f"]))
            return _qint_from_kif(kif), kif, None
        qints = [_qint_from_kif(kif) for kif in complete]
        qint, warning = _conservative_qint(qints)
        return qint, _kif_from_qint(qint), "used conservative folded-reduce precision for heterogeneous input kifs"

    in_bw = op_params.get("in_bw")
    if in_bw is None:
        raise ValueError("folded reduce has no input precision")
    bits = max(int(math.ceil(float(in_bw))), 1)
    kif = {"k": True, "i": max(bits - 1, 0), "f": 0, "bits": bits, "source": "legacy_in_bw"}
    return _qint_from_kif(kif), kif, "used legacy in_bw folded-reduce precision"


def _output_element_count(in_shape, axes) -> int:
    full_shape = tuple(d for d in in_shape[1:] if d is not None)
    axes_in_trace = {a - 1 for a in axes if a >= 1}
    count = 1
    for idx, dim in enumerate(full_shape):
        if idx not in axes_in_trace:
            count *= int(dim)
    return count


def _spatial_trace_shape(in_shape, axes, p_reduce: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    full_shape = tuple(d for d in in_shape[1:] if d is not None)
    axes_in_trace = tuple(a - 1 for a in axes if a >= 1)
    spatial_shape = list(full_shape)
    for axis in axes_in_trace:
        spatial_shape[axis] = p_reduce
    return tuple(spatial_shape), axes_in_trace


def da4ml_reduce_folded_result(
    p: dict,
    weights: WeightProvider,
    fpga: dict,
    *,
    parallelism: int,
    factor: int,
) -> dict[str, Any]:
    """Fold-aware reduction result with explicit partial/accumulator precision."""
    _ = weights
    op_params = p.get("op_params") or {}
    in_shape = op_params.get("in_shape")
    axes = op_params.get("axes") or []
    mode = (op_params.get("mode") or "sum").lower()
    keepdims = bool(op_params.get("keepdims"))

    if mode != "sum":
        raise NotImplementedError(f"fold-aware reduce mode {mode!r} is not supported")
    if in_shape is None:
        raise ValueError(f"reduce vertex {p.get('nn_layer_name')!r}: missing in_shape")

    N = int(parallelism)
    T = max(int(factor), 1)
    P_reduce = max(math.ceil(N / T), 1)
    n_outputs = _output_element_count(in_shape, axes)
    input_qint, input_kif, precision_warning = _input_reduce_qint_and_kif(op_params)

    partial_qint = _sum_qint(input_qint, P_reduce)
    accumulator_qint = _sum_qint(input_qint, N)
    partial_kif = _kif_from_qint(partial_qint)
    accumulator_kif = _kif_from_qint(accumulator_qint)

    if P_reduce > 1:
        spatial_shape, axes_in_trace = _spatial_trace_shape(in_shape, axes, P_reduce)
        body = lambda x: np.sum(x, axis=axes_in_trace, keepdims=keepdims)  # noqa: E731
        spatial_result = _da4ml.trace_lambda_result(
            [spatial_shape],
            body,
            input_qints=[input_qint],
            input_kifs=[input_kif],
            adder_size=int(fpga.get("adder_size", -1)),
            carry_size=int(fpga.get("carry_size", -1)),
            latency_cutoff=int(fpga.get("latency_cutoff", -1)),
        )
        spatial_cost = spatial_result.get("cost") or {}
        spatial_lut = int(spatial_cost.get("lut") or 0)
        spatial_ff = int(spatial_cost.get("ff") or 0)
        spatial_lat = int(spatial_cost.get("latency_cycles") or 0)
    else:
        spatial_lut = 0
        spatial_ff = 0
        spatial_lat = 0

    accum_bits = int(accumulator_kif["bits"])
    accum_lut = n_outputs * accum_bits if T > 1 else 0
    accum_ff = n_outputs * accum_bits if T > 1 else 0
    accum_lat = 1 if T > 1 else 0
    output_qints = [accumulator_qint] * int(n_outputs)
    output_kifs = [accumulator_kif] * int(n_outputs)

    result = _da4ml.empty_kernel_result()
    result["cost"].update(
        {
            "lut": spatial_lut + accum_lut,
            "ff": spatial_ff + accum_ff,
            "dsp": 0,
            "bram": 0,
            "latency_cycles": max(spatial_lat + accum_lat, 1),
            "ii": T,
        }
    )
    result["input_qints"] = [input_qint]
    result["input_kifs"] = [input_kif]
    result["output_qints"] = output_qints
    result["output_kifs"] = output_kifs
    result["input_bitwidths"] = [int(input_kif.get("bits") or 0)]
    result["output_bitwidths"] = [int(kif.get("bits") or 0) for kif in output_kifs]
    result["input_tensor_width_bits"] = sum(result["input_bitwidths"])
    result["output_tensor_width_bits"] = sum(result["output_bitwidths"])
    result["precision_source"] = "fold_aware_derived"
    result["kernel_meta"] = {
        "op": "reduce",
        "mode": mode,
        "reduce_mode": "temporal_accumulate" if P_reduce == 1 else "hybrid",
        "parallelism": N,
        "lanes": P_reduce,
        "temporal_steps": T,
        "partial_sum_qint": partial_qint,
        "partial_sum_kif": partial_kif,
        "accumulator_qint": accumulator_qint,
        "accumulator_kif": accumulator_kif,
        "precision_warning": precision_warning,
    }
    return result


def da4ml_elementwise_cost(p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
    op_params = p.get("op_params") or {}
    op = (op_params.get("op") or "add").lower()
    in_shapes = op_params.get("in_shapes") or []
    in_bws = op_params.get("in_bws") or [op_params.get("in_bw")]

    if not in_shapes or not in_bws or any(b is None for b in in_bws):
        raise ValueError(
            f"elementwise vertex {p.get('nn_layer_name')!r}: missing in_shapes/in_bws"
        )

    # Steady-state pipeline trace: drop the batch dim.
    trace_shapes = [tuple(d for d in s[1:] if d is not None) for s in in_shapes]

    if op == "add":
        body = lambda *xs: _broadcast_reduce(xs, lambda a, b: a + b)  # noqa: E731
    elif op == "sub":
        body = lambda a, b: a - b  # noqa: E731
    elif op == "mul":
        body = lambda *xs: _broadcast_reduce(xs, lambda a, b: a * b)  # noqa: E731
    elif op == "max":
        body = lambda *xs: _broadcast_reduce(xs, lambda a, b: np.maximum(a, b))  # noqa: E731
    elif op == "min":
        body = lambda *xs: _broadcast_reduce(xs, lambda a, b: np.minimum(a, b))  # noqa: E731
    else:
        raise NotImplementedError(f"elementwise op {op!r} not supported")

    result = _da4ml.trace_lambda_result(
        trace_shapes,
        body,
        input_qints=op_params.get("input_qints"),
        input_kifs=op_params.get("input_kifs"),
        input_bws=[float(b) for b in in_bws] if in_bws else None,
        adder_size=int(fpga.get("adder_size", -1)),
        carry_size=int(fpga.get("carry_size", -1)),
        latency_cutoff=int(fpga.get("latency_cutoff", -1)),
    )
    result["kernel_meta"] = {
        "op": "elementwise",
        "elementwise_op": op,
        "n_inputs": len(trace_shapes),
    }
    return result


def _broadcast_reduce(arrays, binop):
    """Apply `binop` left-to-right after broadcasting all inputs to a common shape."""
    bc = np.broadcast_arrays(*arrays)
    out = bc[0]
    for a in bc[1:]:
        out = binop(out, a)
    return out


def da4ml_reduce_temporal_cost(
    p: dict,
    weights: WeightProvider,
    fpga: dict,
    *,
    parallelism: int,
    factor: int,
) -> dict[str, Any]:
    """Cost of a spatial+temporal reduction under the N–P–T model.

    Arguments are:

    * ``parallelism`` — N, the size of the folded (reduced) axis.
    * ``factor``      — T, the temporal step count = ceil(N / P_reduce).

    From these we derive ``P_reduce = ceil(N / T)`` — the number of elements
    the reduction consumes per cycle. Three regimes:

    * ``P_reduce == N``  (T == 1) — *spatial*: one full tree, no accumulator.
      Caller should normally not hit this branch; BIND's cost already
      covers it.
    * ``P_reduce == 1``  (T == N) — *temporal_accumulate*: one accumulator,
      no spatial tree.
    * ``1 < P_reduce < N`` — *hybrid*: spatial tree of width P_reduce plus a
      T-step accumulator, replicated across the un-reduced (channel) dims.

    The cost dict returned has ``ii = T`` and ``latency_cycles = L_reduce``
    (spatial tree + accumulator depth). The caller is responsible for
    writing ``latency_total = L_reduce + (T - 1)`` on the vertex.
    """
    op_params = p.get("op_params") or {}
    in_shape = op_params.get("in_shape")
    in_bw = op_params.get("in_bw")
    axes = op_params.get("axes") or []
    keepdims = bool(op_params.get("keepdims"))

    if in_shape is None or in_bw is None:
        raise ValueError(
            f"reduce vertex {p.get('nn_layer_name')!r}: missing in_shape/in_bw"
        )

    N = int(parallelism)
    T = max(int(factor), 1)
    P_reduce = max(math.ceil(N / T), 1)

    # Trace a per-cycle slice: replace each reduced dim with its
    # (possibly collapsed) per-cycle width = P_reduce.
    full_shape = tuple(d for d in in_shape[1:] if d is not None)
    axes_in_trace = tuple(a - 1 for a in axes if a >= 1)

    spatial_shape = list(full_shape)
    for a in axes_in_trace:
        spatial_shape[a] = P_reduce
    spatial_shape_t = tuple(spatial_shape)

    # ---- spatial tree cost (only meaningful when P_reduce > 1) ----
    if P_reduce > 1:
        body = lambda x: np.sum(x, axis=axes_in_trace, keepdims=keepdims)  # noqa: E731
        spatial_cost = _da4ml.trace_lambda(
            [spatial_shape_t],
            [float(in_bw)],
            body,
            adder_size=int(fpga.get("adder_size", -1)),
            carry_size=int(fpga.get("carry_size", -1)),
            latency_cutoff=int(fpga.get("latency_cutoff", -1)),
        )
        spatial_lut = int(spatial_cost["lut"])
        spatial_ff  = int(spatial_cost["ff"])
        spatial_lat = int(spatial_cost["latency_cycles"])
    else:
        spatial_lut = 0
        spatial_ff = 0
        spatial_lat = 0

    # ---- temporal accumulator: one adder + one register per un-reduced element
    n_channels = 1
    for i, d in enumerate(full_shape):
        if i not in axes_in_trace:
            n_channels *= int(d)

    # The accumulator's intrinsic *pipeline depth* L is one adder stage; the
    # T-step accumulation is reflected via ii=T and latency_total=L+(T-1) by
    # the caller — we must not double-count it here.
    if T > 1:
        accum_bw_in = int(math.ceil(float(in_bw)))
        accum_bw_out = accum_bw_in + max(int(math.ceil(math.log2(T))), 1)
        accum_lut = n_channels * accum_bw_out          # ~1 LUT per output bit
        accum_ff  = n_channels * accum_bw_out          # one FF per accumulated bit
        accum_lat = 1                                   # one adder stage in the datapath
    else:
        accum_lut = 0
        accum_ff = 0
        accum_lat = 0

    cost = {
        "lut": spatial_lut + accum_lut,
        "ff": spatial_ff + accum_ff,
        "dsp": 0,
        "bram": 0,
        "latency_cycles": spatial_lat + accum_lat,  # L_reduce (pipeline depth only)
        "ii": T,
    }

    # Precision model (fold-aware):
    # - spatial tree across P_reduce inputs grows integer bits by ceil(log2(P_reduce))
    # - temporal accumulation across T steps grows integer bits by ceil(log2(T))
    input_kif = op_params.get("input_kif")
    if isinstance(input_kif, dict) and all(k in input_kif for k in ("k", "i", "f")):
        k = bool(input_kif["k"])
        i = int(input_kif["i"])
        f = int(input_kif["f"])
    else:
        # Fallback: derive from scalar in_bw (assumed integer-only).
        bw_int = max(int(math.ceil(float(in_bw))), 1)
        k = True
        i = max(bw_int - 1, 0)
        f = 0

    def _grow(bits: int) -> int:
        return int(math.ceil(math.log2(bits))) if bits and bits > 1 else 0

    i_partial = i + _grow(P_reduce)
    i_acc = i_partial + _grow(T)

    out_kif = {"k": k, "i": i_acc, "f": f, "bits": int(k) + i_acc + f}
    partial_kif = {"k": k, "i": i_partial, "f": f, "bits": int(k) + i_partial + f}

    # Total tensor width: one output element per un-reduced channel.
    output_kifs = [out_kif] * int(n_channels)

    result = _da4ml.empty_kernel_result()
    result["cost"].update(cost)
    result["input_kifs"] = [input_kif] if input_kif is not None else [{"k": k, "i": i, "f": f, "bits": int(k) + i + f}]
    result["output_kifs"] = output_kifs
    result["input_bitwidths"] = [int(kif.get("bits") or 0) for kif in (result["input_kifs"] or [])]
    result["output_bitwidths"] = [int(kif.get("bits") or 0) for kif in output_kifs]
    result["input_tensor_width_bits"] = sum(result["input_bitwidths"]) if result["input_bitwidths"] else None
    result["output_tensor_width_bits"] = sum(result["output_bitwidths"]) if result["output_bitwidths"] else None
    result["precision_source"] = "derived"
    result["kernel_meta"] = {
        "op": "reduce",
        "mode": (op_params.get("mode") or "sum").lower(),
        "reduce_mode": "temporal_accumulate" if P_reduce == 1 else "hybrid",
        "parallelism": N,
        "lanes": P_reduce,
        "temporal_steps": T,
        "partial_sum_kif": partial_kif,
        "accumulator_kif": out_kif,
    }
    return result


def da4ml_activation_cost(p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
    op_params = p.get("op_params") or {}
    func = (op_params.get("func") or "linear").lower()
    in_shape = op_params.get("in_shape")
    in_bw = op_params.get("in_bw")

    if func in ("linear", None):
        return _da4ml.legacy_cost_to_result(
            0.0,
            (0.0, 0.0),
            output_qints=[op_params.get("input_qint")] if op_params.get("input_qint") is not None else None,
            output_kifs=[op_params.get("input_kif")] if op_params.get("input_kif") is not None else None,
            precision_source="inherited",
        )

    if in_shape is None or in_bw is None:
        raise ValueError(
            f"activation vertex {p.get('nn_layer_name')!r}: missing in_shape/in_bw"
        )

    trace_shape = tuple(d for d in in_shape[1:] if d is not None)

    if func == "relu":
        body = lambda x: _da4ml.relu(x)  # noqa: E731
        result = _da4ml.trace_lambda_result(
            [trace_shape],
            body,
            **_single_input_precision_kwargs(op_params),
            adder_size=int(fpga.get("adder_size", -1)),
            carry_size=int(fpga.get("carry_size", -1)),
            latency_cutoff=int(fpga.get("latency_cutoff", -1)),
        )
        result["kernel_meta"] = {
            "op": "activation",
            "func": func,
            "implementation": op_params.get("implementation"),
        }
        return result

    # TODO: route sigmoid/tanh/softmax through a LUT-based estimator. For
    # now flag it loudly so we notice the day a non-ReLU activation appears.
    raise NotImplementedError(
        f"activation {func!r} has no Sched-IR cost query yet (only ReLU is wired)"
    )


# --------------------------------------------------------------------------- #
# Closed-form cost queries (scheduler-inserted infrastructure)
# --------------------------------------------------------------------------- #

def register_buffer_cost(p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
    op_params = p.get("op_params") or {}
    width = int(op_params.get("width_bits") or 0)
    depth = int(op_params.get("depth") or 0)
    return {
        "lut": 0,
        "ff": width * depth,
        "dsp": 0,
        "bram": 0,
        "latency_cycles": 1,
        "ii": 1,
    }


def bram_buffer_cost(p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
    op_params = p.get("op_params") or {}
    width = int(op_params.get("width_bits") or 0)
    depth = int(op_params.get("depth") or 0)
    return {
        "lut": 0,
        "ff": 0,
        "dsp": 0,
        "bram": math.ceil((width * depth) / 36864) if width and depth else 0,
        "latency_cycles": 2,
        "ii": 1,
    }


def lut_mux_cost(p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
    op_params = p.get("op_params") or {}
    n = int(op_params.get("n_inputs") or 0)
    width = int(op_params.get("width_bits") or 0)
    sel = max(int(math.ceil(math.log2(max(n, 1)))), 0)
    return {
        "lut": sel * width,
        "ff": 0,
        "dsp": 0,
        "bram": 0,
        "latency_cycles": 0,
        "ii": 1,
    }


# --------------------------------------------------------------------------- #
# Registry — name (matches `cost_query` in the YAML) → callable
# --------------------------------------------------------------------------- #

REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "da4ml_dense_cost":           da4ml_dense_cost,
    "da4ml_reduce_cost":          da4ml_reduce_cost,
    "da4ml_reduce_folded_result": da4ml_reduce_folded_result,
    "da4ml_reduce_temporal_cost": da4ml_reduce_temporal_cost,
    "da4ml_elementwise_cost":     da4ml_elementwise_cost,
    "da4ml_activation_cost":      da4ml_activation_cost,
    "register_buffer_cost":       register_buffer_cost,
    "bram_buffer_cost":           bram_buffer_cost,
    "lut_mux_cost":               lut_mux_cost,
}
