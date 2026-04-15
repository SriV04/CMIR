"""Thin runtime adapter around da4ml.

The rest of the Sched-IR package only ever talks to da4ml through this
module. Keeping the import surface in one place means:

* a single place to spell the (long, slightly-moving) da4ml import paths,
* a single place to convert between da4ml's native types (`QInterval`,
  `FixedVariableArray`, `Solution`/`CascadedSolution`) and the canonical
  Sched-IR `cost` dict shape `{lut, ff, dsp, bram, latency_cycles, ii}`,
* a clean `RuntimeError` if da4ml is not importable in the current
  interpreter (e.g. running outside the `jedi-linear` conda env).

The module is loaded lazily by `kernels.py`. If `import da4ml` fails the
module still loads — calls into the public functions raise a clear error
instead of crashing at import time.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Lazy import — gate everything behind a clear error if da4ml isn't available
# --------------------------------------------------------------------------- #

try:
    from da4ml.cmvm.api import solve as _da4ml_solve, minimal_latency as _da4ml_minimal_latency
    from da4ml.cmvm.types import QInterval
    from da4ml.trace.fixed_variable_array import FixedVariableArray
    from da4ml.trace.fixed_variable import HWConfig
    from da4ml.trace.tracer import comb_trace
    from da4ml.trace.ops import relu as _da4ml_relu

    _DA4ML_OK = True
    _DA4ML_ERR: str | None = None
except Exception as _exc:  # pragma: no cover
    _DA4ML_OK = False
    _DA4ML_ERR = str(_exc)
    QInterval = None  # type: ignore
    FixedVariableArray = None  # type: ignore
    HWConfig = None  # type: ignore


def _require():
    if not _DA4ML_OK:
        raise RuntimeError(
            "da4ml is not importable in this interpreter. "
            "Run inside the jedi-linear conda env: "
            "`KERAS_BACKEND=jax conda run -n jedi-linear python …`. "
            f"Underlying error: {_DA4ML_ERR}"
        )


# --------------------------------------------------------------------------- #
# QInterval / HWConfig helpers
# --------------------------------------------------------------------------- #

def qint_from_bw(bw: float | int, signed: bool = True) -> "QInterval":
    """Build a representative QInterval from a scalar HGQ-average bitwidth.

    HGQ stores per-parameter bitwidths and we average them to a single
    float — it's lossy but it's all we have at BIND time. We model it as a
    fixed-point integer: `k=signed`, `i=ceil(bw)-k`, `f=0`. The resulting
    interval is `[-2^(i+1), 2^(i+1)-1]` for signed.
    """
    _require()
    bw_int = max(int(math.ceil(float(bw))), 1)
    k = 1 if signed else 0
    i = max(bw_int - k, 0)
    return QInterval.from_kif(k, i, 0)  # type: ignore[union-attr]


def hwconf(adder_size: int = -1, carry_size: int = -1, latency_cutoff: float = -1) -> "HWConfig":
    _require()
    return HWConfig(adder_size, carry_size, latency_cutoff)  # type: ignore[union-attr]


def make_input_array(shape: tuple[int, ...], bw: float, *, signed: bool = True,
                     hw: "HWConfig | None" = None) -> "FixedVariableArray":
    """Allocate a FixedVariableArray of the given shape, every element with the same kif."""
    _require()
    bw_int = max(int(math.ceil(float(bw))), 1)
    k_arr = np.full(shape, 1 if signed else 0, dtype=np.int8)
    i_arr = np.full(shape, max(bw_int - (1 if signed else 0), 0), dtype=np.int32)
    f_arr = np.zeros(shape, dtype=np.int32)
    return FixedVariableArray.from_kif(k_arr, i_arr, f_arr, hw or hwconf())  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Cost-dict normalisation
# --------------------------------------------------------------------------- #

def _empty_cost() -> dict[str, Any]:
    return {"lut": 0, "ff": 0, "dsp": 0, "bram": 0, "latency_cycles": 0, "ii": 1}


def to_cost_dict(da4ml_cost: float, da4ml_latency: tuple[float, float] | float | None) -> dict[str, Any]:
    """Convert a da4ml `(cost, latency)` pair to the canonical Sched-IR dict.

    da4ml's `.cost` is the total adder-cell count (1 cell ≈ 1 LUT in
    ideal-adder mode, more under multi-bit LUT6 packing). `.latency` is
    either a `(min, max)` tuple (Solution) or a single float (worst-case
    path). We always report `latency_cycles = ceil(max)`. da4ml is a
    LUT-only flow so `dsp = 0` and `bram = 0`. Kernels are fully
    pipelined so `ii = 1`.
    """
    cost = _empty_cost()
    cost["lut"] = int(round(float(da4ml_cost)))

    if da4ml_latency is None:
        lat_max = 0.0
    elif isinstance(da4ml_latency, (tuple, list)):
        lat_max = float(da4ml_latency[1])
    else:
        lat_max = float(da4ml_latency)
    cost["latency_cycles"] = int(math.ceil(lat_max))
    return cost


# --------------------------------------------------------------------------- #
# Cost-query helpers
# --------------------------------------------------------------------------- #

def solve_dense(
    kernel: np.ndarray,
    in_qint: "QInterval",
    *,
    adder_size: int = -1,
    carry_size: int = -1,
) -> tuple[float, tuple[float, float]]:
    """Run da4ml's CMVM solver on a constant 2-D kernel.

    `kernel` must be 2-D `(in_features, out_features)`. Returns the rolled-up
    `(cost, (lat_min, lat_max))` from the resulting CascadedSolution.
    """
    _require()
    if kernel.ndim != 2:
        raise NotImplementedError(
            f"da4ml solve_dense currently only handles 2-D constant kernels; "
            f"got shape {kernel.shape}"
        )
    kernel_f = np.ascontiguousarray(kernel.astype(np.float32))
    qintervals = [in_qint] * kernel_f.shape[0]
    sol = _da4ml_solve(
        kernel_f,
        qintervals=qintervals,
        adder_size=adder_size,
        carry_size=carry_size,
    )
    return float(sol.cost), tuple(sol.latency)  # type: ignore[return-value]


def trace_lambda(
    input_shapes: list[tuple[int, ...]],
    input_bws: list[float],
    body: Callable[..., Any],
    *,
    adder_size: int = -1,
    carry_size: int = -1,
) -> tuple[float, tuple[float, float]]:
    """Generic post-trace estimator.

    Allocates one `FixedVariableArray` per `input_shapes[i]` at bitwidth
    `input_bws[i]`, calls `body(*inputs)`, then traces the resulting
    output(s) through `comb_trace` and reads `.cost` / `.latency`.

    The body is free to use FixedVariable operator overloads (`+`, `-`,
    `*`, `np.maximum`, `np.sum`, `np.broadcast_arrays`, …) and any helper
    in `da4ml.trace.ops` (relu, einsum, reduce, …).
    """
    _require()
    hw = hwconf(adder_size=adder_size, carry_size=carry_size)
    inputs = [make_input_array(s, bw, hw=hw) for s, bw in zip(input_shapes, input_bws)]

    output = body(*inputs)
    # `body` may return a FixedVariableArray, an ndarray of FixedVariables,
    # or a single FixedVariable. comb_trace accepts all three; let it deal.
    sol = comb_trace(_concat_inputs(inputs), output)
    return float(sol.cost), tuple(sol.latency)  # type: ignore[return-value]


def _concat_inputs(inputs: list["FixedVariableArray"]):
    """Flatten multiple FixedVariableArrays into a single sequence for comb_trace."""
    if len(inputs) == 1:
        return inputs[0]
    flat = []
    for a in inputs:
        flat.extend(np.ravel(a._vars).tolist())  # type: ignore[attr-defined]
    return flat


def relu(x):
    """ReLU on a FixedVariableArray."""
    _require()
    return _da4ml_relu(x)


def minimal_latency_dense(kernel: np.ndarray, in_qint: "QInterval") -> float:
    """Cheap latency lower-bound for a dense kernel — used by sanity checks."""
    _require()
    kernel_f = np.ascontiguousarray(kernel.astype(np.float32))
    qintervals = [in_qint] * kernel_f.shape[0]
    latencies = [0.0] * kernel_f.shape[0]
    return float(_da4ml_minimal_latency(kernel_f, qintervals, latencies))
