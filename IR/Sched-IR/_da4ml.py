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
    from da4ml.cmvm.types import QInterval, Solution, CascadedSolution
    from da4ml.trace.fixed_variable_array import FixedVariableArray
    from da4ml.trace.fixed_variable import HWConfig
    from da4ml.trace.tracer import comb_trace
    from da4ml.trace.pipeline import to_pipeline as _da4ml_to_pipeline
    from da4ml.trace.ops import relu as _da4ml_relu

    _DA4ML_OK = True
    _DA4ML_ERR: str | None = None
except Exception as _exc:  # pragma: no cover
    _DA4ML_OK = False
    _DA4ML_ERR = str(_exc)
    QInterval = None  # type: ignore
    Solution = None  # type: ignore
    CascadedSolution = None  # type: ignore
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
# Latency-cutoff derivation (for path B, mirrored by evaluate.py for path A)
# --------------------------------------------------------------------------- #

def derive_latency_cutoff(
    target_fmax_hz: float,
    t_logic_ns: float = 1.00,
    routing_margin: float = 0.30,
) -> int:
    """Derive a `latency_cutoff` (logic levels per pipeline stage) that should
    meet timing at `target_fmax_hz` on the given device.

    The model is:

        T_clk_ns        = 1e9 / target_fmax_hz
        usable_T_clk_ns = T_clk_ns × (1 - routing_margin)
        latency_cutoff  = floor(usable_T_clk_ns / t_logic_ns)

    `t_logic_ns` is the mean delay of one da4ml logic level (one full adder +
    local routing). It is device- and `carry_size`-dependent. Calibrated
    starting points:

        VU13P  carry_size=-1 → 1.00 ns   (paper default; gives cutoff=2 @ 300 MHz)
        VU13P  carry_size= 4 → 0.65 ns   (CARRY8 four-bit chunk)
        VU13P  carry_size= 8 → 0.95 ns   (CARRY8 full)
        S10    carry_size=-1 → 0.85 ns

    `routing_margin` is the fraction of T_clk reserved for inter-CLB routing
    and setup/hold (typical 0.30 for moderate fanout). For final paper
    figures, calibrate against actual Vivado WNS and override the YAML.

    Returns at least 1.
    """
    if not target_fmax_hz or float(target_fmax_hz) <= 0:
        return -1
    T_clk_ns = 1e9 / float(target_fmax_hz)
    usable_ns = T_clk_ns * max(0.0, 1.0 - float(routing_margin))
    return max(1, int(usable_ns / max(float(t_logic_ns), 1e-9)))


# --------------------------------------------------------------------------- #
# Cost-dict construction (now reg_bits-aware via to_pipeline)
# --------------------------------------------------------------------------- #

def _empty_cost() -> dict[str, Any]:
    return {"lut": 0, "ff": 0, "dsp": 0, "bram": 0, "latency_cycles": 0, "ii": 1}


def _ensure_pipeline(sol, latency_cutoff: int):
    """Coerce a `Solution`/`CascadedSolution` into a `CascadedSolution` whose
    stages respect `latency_cutoff` logic-levels each.

    * `Solution`               → `to_pipeline(sol, cutoff)` if cutoff > 0,
                                 else wrap as a single-stage CascadedSolution
                                 so we can still read `.reg_bits`.
    * `CascadedSolution`       → re-pipeline each component `Solution`
                                 individually at `cutoff`, then concatenate.
                                 (`solve()` returns a 2-stage CSD cascade
                                 whose stages can be far longer than one
                                 pipeline cutoff each, so this re-staging is
                                 essential for accurate FF/cycle counts.)
    """
    _require()
    if sol is None:
        return None

    cutoff = int(latency_cutoff) if latency_cutoff is not None else -1

    # CascadedSolution path — re-pipeline each sub-Solution at `cutoff`
    # and concatenate. We disable `retiming` on sub-stages because the binary
    # search inside retime_pipeline does a forward-eval that assumes the input
    # is the *original* cascade's input — applying it to a sub-Solution of a
    # 2-stage CSD cascade triggers shape mismatches between stage0's output
    # and stage1's input. Without retiming, to_pipeline still slices the ops
    # by latency and produces a correct CascadedSolution; we just don't get
    # the binary-search delay-balancing pass (acceptable for cost estimation,
    # since stage count is what matters).
    if isinstance(sol, CascadedSolution):  # type: ignore[arg-type]
        if cutoff > 0:
            stages: list = []
            for sub in sol.solutions:
                if not getattr(sub, "ops", None):
                    continue
                try:
                    sub_pipe = _da4ml_to_pipeline(sub, cutoff, retiming=False, verbose=False)
                    stages.extend(sub_pipe.solutions)
                except (AssertionError, ValueError, KeyError, IndexError):
                    # to_pipeline has rough edges for tiny / degenerate
                    # sub-Solutions (single-op stages, no output ops in stage
                    # 0, etc.). Fall back to using the sub as a single stage
                    # so we still get its inp/out_qint contribution to reg_bits.
                    stages.append(sub)
            if not stages:
                return sol
            return CascadedSolution(solutions=tuple(stages))  # type: ignore[call-arg]
        return sol

    # Single-stage Solution path — full retiming is safe here since the
    # forward-eval matches the original input/output shapes exactly.
    if cutoff > 0 and getattr(sol, "ops", None):
        for retime in (True, False):
            try:
                return _da4ml_to_pipeline(sol, cutoff, retiming=retime, verbose=False)
            except (AssertionError, ValueError, KeyError, IndexError):
                continue
    # Wrap as a 1-stage cascade so reg_bits is queryable.
    return CascadedSolution(solutions=(sol,))  # type: ignore[call-arg]


def solution_to_cost(sol, latency_cutoff: int = -1) -> dict[str, Any]:
    """Read `(lut, ff, latency_cycles)` off a `Solution`/`CascadedSolution`.

    Always returns the canonical Sched-IR cost dict. Pipelining is applied
    when `latency_cutoff > 0`; otherwise we still wrap a single-stage
    `Solution` in a 1-stage cascade so `.reg_bits` (input qint bits + output
    qint bits) is queryable as a baseline FF count.
    """
    _require()
    pipe = _ensure_pipeline(sol, latency_cutoff)
    out = _empty_cost()
    if pipe is None:
        return out
    out["lut"] = int(round(float(pipe.cost)))
    out["ff"] = int(getattr(pipe, "reg_bits", 0))
    out["latency_cycles"] = int(len(pipe.solutions))
    return out


# Back-compat shim: a couple of call sites still expect `(cost, latency)`.
def to_cost_dict(da4ml_cost: float, da4ml_latency: tuple[float, float] | float | None) -> dict[str, Any]:
    """Legacy `(cost, latency)` → cost-dict converter.

    Kept for the `da4ml_activation_cost` linear path which never builds a
    Solution. New code should call `solution_to_cost` instead.
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
    latency_cutoff: int = -1,
) -> dict[str, Any]:
    """Run da4ml's CMVM solver on a constant 2-D kernel and return a cost dict.

    `kernel` must be 2-D `(in_features, out_features)`.

    When `latency_cutoff > 0`, each stage of the resulting CSD cascade is
    re-pipelined at the cutoff so `ff` (= `reg_bits`) and `latency_cycles`
    (= number of pipeline stages) reflect the real hardware. When
    `latency_cutoff <= 0`, the raw 2-stage CSD cascade is reported (fewer
    stages, smaller reg_bits — useful only for relative comparisons).
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
    return solution_to_cost(sol, latency_cutoff=latency_cutoff)


def trace_lambda(
    input_shapes: list[tuple[int, ...]],
    input_bws: list[float],
    body: Callable[..., Any],
    *,
    adder_size: int = -1,
    carry_size: int = -1,
    latency_cutoff: int = -1,
) -> dict[str, Any]:
    """Generic post-trace estimator; returns a full cost dict.

    Allocates one `FixedVariableArray` per `input_shapes[i]` at bitwidth
    `input_bws[i]`, calls `body(*inputs)`, then `comb_trace`s the result.
    When `latency_cutoff > 0` the resulting `Solution` is re-pipelined so
    `ff` and `latency_cycles` are realistic.

    The body is free to use FixedVariable operator overloads (`+`, `-`,
    `*`, `np.maximum`, `np.sum`, `np.broadcast_arrays`, …) and any helper
    in `da4ml.trace.ops` (relu, einsum, reduce, …).
    """
    _require()
    # Plumb the cutoff into HWConfig so latencies inside the FixedVariable
    # graph are *also* aware of stage boundaries — this matches the rounding
    # `to_pipeline` will do in solution_to_cost.
    hw = hwconf(adder_size=adder_size, carry_size=carry_size,
                latency_cutoff=latency_cutoff if latency_cutoff and latency_cutoff > 0 else -1)
    inputs = [make_input_array(s, bw, hw=hw) for s, bw in zip(input_shapes, input_bws)]

    output = body(*inputs)
    sol = comb_trace(_concat_inputs(inputs), output)
    return solution_to_cost(sol, latency_cutoff=latency_cutoff)


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
