"""Sched-IR kernel library.

Each function in this module is a `cost_query` named in
`da4ml-resource.yaml`. Given a Sched-IR vertex pmap and a `WeightProvider`,
it returns the canonical cost dict
``{lut, ff, dsp, bram, latency_cycles, ii}`` that the BIND pass writes onto
the vertex.

Cost queries fall into two groups:

1. **da4ml-driven** (`da4ml_*_cost`): build a tiny da4ml trace from the
   Sched-IR `op_params`, run `comb_trace` (or `cmvm.api.solve` for dense),
   read `.cost` / `.latency` back out via `_da4ml.to_cost_dict`.
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

def da4ml_dense_cost(p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
    op_params = p.get("op_params") or {}
    in_bw = op_params.get("in_bw")
    if in_bw is None:
        raise ValueError(f"dense vertex {p.get('nn_layer_name')!r} has no in_bw")

    kernel = weights.get_kernel(p["nn_layer_name"])
    if kernel is None:
        raise ValueError(
            f"dense vertex {p['nn_layer_name']!r}: no constant kernel found on the Keras model"
        )

    # Collapse higher-rank kernels (e.g. (M, K, N)) to a 2-D (in_features,
    # out_features) view by flattening leading dims into the input axis.
    # JEDI-linear kernels are already 2-D so this is a no-op there.
    if kernel.ndim != 2:
        kernel = kernel.reshape(-1, kernel.shape[-1])

    in_qint = _da4ml.qint_from_bw(in_bw)
    cost_total, latency = _da4ml.solve_dense(
        kernel,
        in_qint,
        adder_size=int(fpga.get("adder_size", -1)),
        carry_size=int(fpga.get("carry_size", -1)),
    )
    return _da4ml.to_cost_dict(cost_total, latency)


def da4ml_reduce_cost(p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
    op_params = p.get("op_params") or {}
    in_shape = op_params.get("in_shape")
    in_bw = op_params.get("in_bw")
    axes = op_params.get("axes")
    mode = (op_params.get("mode") or "sum").lower()
    keepdims = bool(op_params.get("keepdims"))

    if in_shape is None or in_bw is None or axes is None:
        raise ValueError(
            f"reduce vertex {p.get('nn_layer_name')!r}: missing in_shape/in_bw/axes"
        )

    # Drop the symbolic batch dim from the trace shape — we only need one
    # batch element to estimate cost/latency for the steady-state pipeline.
    trace_shape = tuple(d for d in in_shape[1:] if d is not None)
    axes_in_trace = tuple(a - 1 for a in axes if a >= 1)

    if mode == "sum":
        body = lambda x: np.sum(x, axis=axes_in_trace, keepdims=keepdims)  # noqa: E731
    elif mode == "max":
        body = lambda x: np.amax(x, axis=axes_in_trace, keepdims=keepdims)  # noqa: E731
    elif mode == "min":
        body = lambda x: np.amin(x, axis=axes_in_trace, keepdims=keepdims)  # noqa: E731
    elif mode == "mean":
        # Mean = sum * (1/N); the * by a constant doesn't add real cost
        # in da4ml so this is the same as sum for BIND purposes.
        body = lambda x: np.sum(x, axis=axes_in_trace, keepdims=keepdims)  # noqa: E731
    else:
        raise NotImplementedError(f"reduce mode {mode!r} not supported")

    cost_total, latency = _da4ml.trace_lambda(
        [trace_shape],
        [float(in_bw)],
        body,
        adder_size=int(fpga.get("adder_size", -1)),
        carry_size=int(fpga.get("carry_size", -1)),
    )
    return _da4ml.to_cost_dict(cost_total, latency)


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

    cost_total, latency = _da4ml.trace_lambda(
        trace_shapes,
        [float(b) for b in in_bws],
        body,
        adder_size=int(fpga.get("adder_size", -1)),
        carry_size=int(fpga.get("carry_size", -1)),
    )
    return _da4ml.to_cost_dict(cost_total, latency)


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
        cost_total, latency = _da4ml.trace_lambda(
            [spatial_shape_t],
            [float(in_bw)],
            body,
            adder_size=int(fpga.get("adder_size", -1)),
            carry_size=int(fpga.get("carry_size", -1)),
        )
        spatial_lut = int(round(float(cost_total)))
        spatial_lat = int(math.ceil(float(latency[1] if isinstance(latency, (tuple, list)) else latency)))
    else:
        spatial_lut = 0
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

    return {
        "lut": spatial_lut + accum_lut,
        "ff":  accum_ff,
        "dsp": 0,
        "bram": 0,
        "latency_cycles": spatial_lat + accum_lat,   # L_reduce (pipeline depth only)
        "ii": T,                                      # = T, matches node.ii
    }


def da4ml_activation_cost(p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
    op_params = p.get("op_params") or {}
    func = (op_params.get("func") or "linear").lower()
    in_shape = op_params.get("in_shape")
    in_bw = op_params.get("in_bw")

    if func in ("linear", None):
        return _da4ml.to_cost_dict(0.0, (0.0, 0.0))

    if in_shape is None or in_bw is None:
        raise ValueError(
            f"activation vertex {p.get('nn_layer_name')!r}: missing in_shape/in_bw"
        )

    trace_shape = tuple(d for d in in_shape[1:] if d is not None)

    if func == "relu":
        body = lambda x: _da4ml.relu(x)  # noqa: E731
        cost_total, latency = _da4ml.trace_lambda(
            [trace_shape],
            [float(in_bw)],
            body,
            adder_size=int(fpga.get("adder_size", -1)),
            carry_size=int(fpga.get("carry_size", -1)),
        )
        return _da4ml.to_cost_dict(cost_total, latency)

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
    "da4ml_reduce_temporal_cost": da4ml_reduce_temporal_cost,
    "da4ml_elementwise_cost":     da4ml_elementwise_cost,
    "da4ml_activation_cost":      da4ml_activation_cost,
    "register_buffer_cost":       register_buffer_cost,
    "bram_buffer_cost":           bram_buffer_cost,
    "lut_mux_cost":               lut_mux_cost,
}
