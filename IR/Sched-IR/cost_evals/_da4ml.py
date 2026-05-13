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


def qint_to_dict(qint) -> dict | None:
    if qint is None:
        return None
    if isinstance(qint, dict):
        return qint
    try:
        return {
            "min": float(qint.min),
            "max": float(qint.max),
            "step": float(qint.step),
        }
    except AttributeError:
        low, high, step = qint
        return {
            "min": float(low),
            "max": float(high),
            "step": float(step),
        }


def qints_to_dicts(qints) -> list[dict] | None:
    if qints is None:
        return None
    return [qint_to_dict(q) for q in qints]


def kif_to_dict(kif) -> dict | None:
    if kif is None:
        return None
    if isinstance(kif, dict):
        if all(np.isscalar(kif.get(x)) for x in ("k", "i", "f")):
            k = bool(kif["k"])
            i = int(kif["i"])
            f = int(kif["f"])
            return {
                "k": k,
                "i": i,
                "f": f,
                "bits": int(k) + i + f,
            }
        raise ValueError("array-valued KIF dict must be flattened before kif_to_dict")
    if hasattr(kif, "keep_negative"):
        k = bool(kif.keep_negative)
        i = int(kif.integers)
        f = int(kif.fractional)
    else:
        k, i, f = kif
        k = bool(k)
        i = int(i)
        f = int(f)
    return {
        "k": k,
        "i": i,
        "f": f,
        "bits": int(k) + i + f,
    }


def flatten_kif_dict(kif: dict) -> list[dict]:
    k = np.asarray(kif["k"])
    i = np.asarray(kif["i"])
    f = np.asarray(kif["f"])
    k, i, f = np.broadcast_arrays(k, i, f)
    return [
        kif_to_dict({"k": kk, "i": ii, "f": ff})
        for kk, ii, ff in zip(k.ravel(), i.ravel(), f.ravel())
    ]


def kifs_payload_to_dicts(kifs) -> list[dict] | None:
    if kifs is None:
        return None
    if isinstance(kifs, dict):
        if any(np.asarray(kifs.get(x)).ndim > 0 for x in ("k", "i", "f")):
            return flatten_kif_dict(kifs)
        return [kif_to_dict(kifs)]
    if isinstance(kifs, (list, tuple)):
        if kifs and isinstance(kifs[0], (list, tuple)) and len(kifs[0]) == 3:
            return [kif_to_dict(tuple(kif)) for kif in kifs]
        if len(kifs) == 3 and all(np.isscalar(x) for x in kifs):
            return [kif_to_dict(tuple(kifs))]
    if isinstance(kifs, np.ndarray):
        arr = np.asarray(kifs)
    else:
        arr = np.asarray(kifs, dtype=object)

    if arr.ndim == 2 and arr.shape[0] == 3:
        return [kif_to_dict((arr[0, idx], arr[1, idx], arr[2, idx])) for idx in range(arr.shape[1])]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return [kif_to_dict(tuple(row)) for row in arr]
    if arr.ndim == 1 and arr.size == 3 and all(np.isscalar(x) for x in arr.tolist()):
        return [kif_to_dict(tuple(arr.tolist()))]

    if isinstance(kifs, (list, tuple)):
        return [kif_to_dict(kif) for kif in kifs]
    return [kif_to_dict(kifs)]


def _scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def _all_equal(arr: np.ndarray) -> bool:
    flat = arr.reshape(-1)
    return bool(flat.size and np.all(arr == flat[0]))


def _collapse_precision_record_to_features(
    record: dict,
    keys: tuple[str, ...],
    feature_count: int | None,
    *,
    kind: str,
    context: str,
) -> list[dict]:
    arrays = [np.asarray(record[key]) for key in keys]
    arrays = list(np.broadcast_arrays(*arrays))

    # A broadcast-shaped but globally uniform precision is exact as one
    # reusable CMVM input precision.
    if all(_all_equal(arr) for arr in arrays):
        return [
            {key: _scalar(arr.reshape(-1)[0]) for key, arr in zip(keys, arrays)}
        ]

    if feature_count is None:
        raise ValueError(f"{context}: array-valued {kind} requires feature_count")

    shape = arrays[0].shape
    if not shape or shape[-1] != feature_count:
        raise ValueError(
            f"{context}: array-valued {kind} has shape {shape}, "
            f"but dense kernel expects last dimension == feature_count={feature_count}"
        )

    collapsed: list[np.ndarray] = []
    for key, arr in zip(keys, arrays):
        flat = arr.reshape(-1, feature_count)
        if not np.all(flat == flat[0:1, :]):
            raise ValueError(
                f"{context}: dense input precision varies across non-feature axes "
                f"for {kind}.{key}; cannot represent this exactly as CMVM "
                f"input-feature precision"
            )
        collapsed.append(flat[0, :])

    return [
        {key: _scalar(values[j]) for key, values in zip(keys, collapsed)}
        for j in range(feature_count)
    ]


def _qints_from_array_dict(qint_payload: dict, feature_count: int | None, context: str):
    records = _collapse_precision_record_to_features(
        qint_payload,
        ("min", "max", "step"),
        feature_count,
        kind="qint",
        context=context,
    )
    if len(records) == 1:
        return qint_from_dict(records[0])
    return [qint_from_dict(record) for record in records]


def _qints_from_kif_array_dict(kif_payload: dict, feature_count: int | None, context: str):
    records = _collapse_precision_record_to_features(
        kif_payload,
        ("k", "i", "f"),
        feature_count,
        kind="kif",
        context=context,
    )
    kifs = [kif_to_dict(record) for record in records]
    if len(kifs) == 1:
        return qint_from_kif_dict(kifs[0])
    return [qint_from_kif_dict(kif) for kif in kifs]


def qint_from_dict(obj):
    if obj is None:
        return None
    if not isinstance(obj, dict):
        return obj
    _require()
    return QInterval(float(obj["min"]), float(obj["max"]), float(obj["step"]))  # type: ignore[union-attr]


def qint_from_kif_dict(kif):
    if kif is None:
        return None
    _require()
    return QInterval.from_kif(int(bool(kif["k"])), int(kif["i"]), int(kif["f"]))  # type: ignore[union-attr]


def _shape_size(shape: tuple[int, ...] | None) -> int | None:
    if shape is None:
        return None
    dims = [dim for dim in shape if dim is not None]
    if len(dims) != len(shape):
        return None
    return int(np.prod(dims)) if dims else 1


def qints_from_precision_payload(
    qint_payload,
    kif_payload=None,
    fallback_bw=None,
    shape=None,
    feature_count: int | None = None,
    context: str = "precision payload",
):
    """Coerce qint/kif/bitwidth payloads into QInterval values.

    Priority: explicit qint payload, then kif payload, then scalar bw fallback.
    Scalar payloads stay scalar. Array/list payloads stay explicit.
    """
    _ = shape
    if qint_payload is not None:
        # da4ml's `QInterval` is tuple-like (iterable) in some versions, so
        # detect it *before* treating tuples/lists as per-element payloads.
        try:
            if QInterval is not None and isinstance(qint_payload, QInterval):  # type: ignore[arg-type]
                return qint_payload
        except TypeError:
            pass
        if isinstance(qint_payload, dict):
            qint_keys = {"min", "max", "step"}
            if qint_keys.issubset(qint_payload.keys()):
                if any(np.asarray(qint_payload[key]).ndim > 0 for key in qint_keys):
                    return _qints_from_array_dict(
                        qint_payload,
                        feature_count=feature_count,
                        context=context,
                    )
                return qint_from_dict(qint_payload)
            raise ValueError("array-valued qint dicts are not supported; flatten before coercion")
        if isinstance(qint_payload, (list, tuple)):
            return [qint_from_dict(q) for q in qint_payload]
        if isinstance(qint_payload, np.ndarray):
            return [qint_from_dict(q) for q in np.ravel(qint_payload).tolist()]
        return qint_from_dict(qint_payload)

    if kif_payload is not None:
        if isinstance(kif_payload, dict):
            kif_keys = {"k", "i", "f"}
            if kif_keys.issubset(kif_payload.keys()) and any(
                np.asarray(kif_payload[key]).ndim > 0 for key in kif_keys
            ):
                return _qints_from_kif_array_dict(
                    kif_payload,
                    feature_count=feature_count,
                    context=context,
                )
        kifs = kifs_payload_to_dicts(kif_payload)
        if kifs is not None:
            if len(kifs) == 1:
                return qint_from_kif_dict(kifs[0])
            return [qint_from_kif_dict(kif) for kif in kifs]

    if fallback_bw is not None:
        return qint_from_bw(fallback_bw)
    return None

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
    return FixedVariableArray.from_kif(k_arr, i_arr, f_arr, hwconf=hw or hwconf())  # type: ignore[union-attr]


def make_input_array_from_qint(shape: tuple[int, ...], qint, *, hw: "HWConfig | None" = None) -> "FixedVariableArray":
    _require()
    qint = qint_from_dict(qint) if isinstance(qint, dict) else qint
    qdict = qint_to_dict(qint)
    low = np.full(shape, float(qdict["min"]), dtype=np.float64)
    high = np.full(shape, float(qdict["max"]), dtype=np.float64)
    step = np.full(shape, float(qdict["step"]), dtype=np.float64)
    return FixedVariableArray.from_lhs(low, high, step, hwconf=hw or hwconf())  # type: ignore[union-attr]


def make_input_array_from_kif(shape: tuple[int, ...], kif, *, hw: "HWConfig | None" = None) -> "FixedVariableArray":
    _require()
    kif = kif_to_dict(kif)
    k_arr = np.full(shape, int(bool(kif["k"])), dtype=np.int8)
    i_arr = np.full(shape, int(kif["i"]), dtype=np.int32)
    f_arr = np.full(shape, int(kif["f"]), dtype=np.int32)
    return FixedVariableArray.from_kif(k_arr, i_arr, f_arr, hwconf=hw or hwconf())  # type: ignore[union-attr]


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
    return {
        "lut": 0,
        "ff": 0,
        "dsp": 0,
        "bram": 0,
        "uram": 0,
        "latency_cycles": 0,
        "ii": 1,
        "reg_bits": None,
        "logic_cost_raw": None,
        "pipeline_stages": None,
        "cost_source": None,
        "cost_notes": None,
    }


def empty_kernel_result() -> dict:
    return {
        "cost": _empty_cost(),
        "input_qints": None,
        "input_kifs": None,
        "output_qints": None,
        "output_kifs": None,
        "input_bitwidths": None,
        "output_bitwidths": None,
        "input_tensor_width_bits": None,
        "output_tensor_width_bits": None,
        "precision_source": "unknown",
        "da4ml": {
            "solution_type": None,
            "n_inputs": None,
            "n_outputs": None,
            "latency": None,
            "out_latency": None,
            "reg_bits": None,
            "pipeline_stages": None,
        },
    }


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


def _validate_kernel_result(result: dict) -> None:
    output_qints = result.get("output_qints") or []
    output_kifs = result.get("output_kifs") or []
    if output_qints and output_kifs and len(output_qints) != len(output_kifs):
        raise ValueError("kernel result mismatch: output_qints and output_kifs lengths differ")
    if output_kifs:
        width = sum(int(kif["bits"]) for kif in output_kifs if kif is not None)
        if result.get("output_tensor_width_bits") is not None and width != result["output_tensor_width_bits"]:
            raise ValueError("kernel result mismatch: output_tensor_width_bits does not match output_kifs")


def solution_to_result(sol, latency_cutoff: int = -1) -> dict[str, Any]:
    _require()
    pipe = _ensure_pipeline(sol, latency_cutoff)
    result = empty_kernel_result()
    if pipe is None:
        return result

    cost = result["cost"]
    reg_bits = int(getattr(pipe, "reg_bits", 0))
    n_stages = int(len(pipe.solutions))
    cost.update(
        {
            "lut": int(round(float(pipe.cost))),
            "ff": reg_bits,
            "dsp": 0,
            "bram": 0,
            "uram": 0,
            "latency_cycles": n_stages,
            "ii": 1,
            "reg_bits": reg_bits,
            "logic_cost_raw": float(pipe.cost),
            "pipeline_stages": n_stages,
            "cost_source": "da4ml",
        }
    )

    raw_input_qints = getattr(pipe, "inp_qint", None)
    raw_output_qints = getattr(pipe, "out_qint", None)
    input_qints = qints_to_dicts(raw_input_qints)
    output_qints = qints_to_dicts(raw_output_qints)

    raw_input_kifs = getattr(pipe, "inp_kifs", None)
    raw_output_kifs = getattr(pipe, "out_kifs", None)
    if raw_input_kifs is None and raw_input_qints is not None:
        raw_input_kifs = [getattr(q, "precision", None) for q in raw_input_qints]
    if raw_output_kifs is None and raw_output_qints is not None:
        raw_output_kifs = [getattr(q, "precision", None) for q in raw_output_qints]
    input_kifs = kifs_payload_to_dicts(raw_input_kifs)
    output_kifs = kifs_payload_to_dicts(raw_output_kifs)

    result["input_qints"] = input_qints
    result["output_qints"] = output_qints
    result["input_kifs"] = input_kifs
    result["output_kifs"] = output_kifs
    result["input_bitwidths"] = [k["bits"] for k in input_kifs] if input_kifs else None
    result["output_bitwidths"] = [k["bits"] for k in output_kifs] if output_kifs else None
    result["input_tensor_width_bits"] = sum(result["input_bitwidths"]) if result["input_bitwidths"] else None
    result["output_tensor_width_bits"] = sum(result["output_bitwidths"]) if result["output_bitwidths"] else None
    result["precision_source"] = "da4ml"
    n_inputs = len(input_qints) if input_qints else len(input_kifs) if input_kifs else None
    n_outputs = len(output_qints) if output_qints else len(output_kifs) if output_kifs else None
    result["da4ml"] = {
        "solution_type": type(pipe).__name__,
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "shape": tuple(pipe.shape) if hasattr(pipe, "shape") else None,
        "latency": tuple(map(float, pipe.latency)) if hasattr(pipe, "latency") else None,
        "out_latency": list(map(float, pipe.out_latencies)) if hasattr(pipe, "out_latencies") else None,
        "reg_bits": reg_bits,
        "pipeline_stages": n_stages,
    }
    _validate_kernel_result(result)
    return result


def solution_to_cost(sol, latency_cutoff: int = -1) -> dict[str, Any]:
    """Legacy wrapper: return only the canonical cost dict."""
    return solution_to_result(sol, latency_cutoff)["cost"]


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


def legacy_cost_to_result(
    da4ml_cost: float,
    da4ml_latency: tuple[float, float] | float | None,
    *,
    output_qints=None,
    output_kifs=None,
    precision_source: str | None = None,
):
    result = empty_kernel_result()
    result["cost"] = to_cost_dict(da4ml_cost, da4ml_latency)
    result["output_qints"] = output_qints
    result["output_kifs"] = output_kifs
    result["output_bitwidths"] = [k["bits"] for k in output_kifs] if output_kifs else None
    result["output_tensor_width_bits"] = sum(result["output_bitwidths"]) if result["output_bitwidths"] else None
    result["precision_source"] = precision_source or ("derived" if (output_qints or output_kifs) else "unknown")
    _validate_kernel_result(result)
    return result


# --------------------------------------------------------------------------- #
# Cost-query helpers
# --------------------------------------------------------------------------- #

def solve_dense_result(
    kernel: np.ndarray,
    in_qint: "QInterval | None" = None,
    *,
    input_qints: list | None = None,
    adder_size: int = -1,
    carry_size: int = -1,
    latency_cutoff: int = -1,
) -> dict[str, Any]:
    """Run da4ml's CMVM solver on a constant 2-D kernel and return full result metadata.

    `kernel` must be 2-D `(in_features, out_features)`.
    """
    _require()
    if kernel.ndim != 2:
        raise NotImplementedError(
            f"da4ml solve_dense currently only handles 2-D constant kernels; "
            f"got shape {kernel.shape}"
        )
    kernel_f = np.ascontiguousarray(kernel.astype(np.float32))
    if input_qints is not None:
        qintervals = [qint_from_dict(q) if isinstance(q, dict) else q for q in input_qints]
        if len(qintervals) != kernel_f.shape[0]:
            raise ValueError(
                f"solve_dense_result expected {kernel_f.shape[0]} input_qints, got {len(qintervals)}"
            )
    else:
        if in_qint is None:
            raise ValueError("solve_dense_result requires in_qint or input_qints")
        qintervals = [in_qint] * kernel_f.shape[0]
    sol = _da4ml_solve(
        kernel_f,
        qintervals=qintervals,
        adder_size=adder_size,
        carry_size=carry_size,
    )
    result = solution_to_result(sol, latency_cutoff=latency_cutoff)
    result["da4ml"]["kernel_shape"] = tuple(kernel_f.shape)
    return result


def solve_dense(
    kernel: np.ndarray,
    in_qint: "QInterval",
    *,
    adder_size: int = -1,
    carry_size: int = -1,
    latency_cutoff: int = -1,
) -> dict[str, Any]:
    return solve_dense_result(
        kernel,
        in_qint,
        adder_size=adder_size,
        carry_size=carry_size,
        latency_cutoff=latency_cutoff,
    )["cost"]


def trace_lambda_result(
    input_shapes: list[tuple[int, ...]],
    body: Callable[..., Any],
    *,
    input_qints=None,
    input_kifs=None,
    input_bws=None,
    adder_size: int = -1,
    carry_size: int = -1,
    latency_cutoff: int = -1,
) -> dict[str, Any]:
    _require()
    hw = hwconf(adder_size=adder_size, carry_size=carry_size,
                latency_cutoff=latency_cutoff if latency_cutoff and latency_cutoff > 0 else -1)
    inputs = []
    for idx, shape in enumerate(input_shapes):
        qint_payload = input_qints[idx] if input_qints is not None else None
        kif_payload = input_kifs[idx] if input_kifs is not None else None
        bw_payload = input_bws[idx] if input_bws is not None else None

        if qint_payload is not None:
            qints = qints_from_precision_payload(qint_payload)
            if not isinstance(qints, list):
                inputs.append(make_input_array_from_qint(shape, qints, hw=hw))
            else:
                expected = _shape_size(shape)
                if expected is None:
                    raise ValueError(
                        f"input {idx}: per-element qints require a fully concrete shape, got {shape}"
                    )
                if len(qints) != expected:
                    raise ValueError(
                        f"input {idx}: got {len(qints)} qints for shape {shape}, expected {expected}"
                    )
                qints_arr = np.array([qint_to_dict(q) for q in qints], dtype=object).reshape(shape)
                low = np.vectorize(lambda d: float(d["min"]))(qints_arr)
                high = np.vectorize(lambda d: float(d["max"]))(qints_arr)
                step = np.vectorize(lambda d: float(d["step"]))(qints_arr)
                inputs.append(FixedVariableArray.from_lhs(low, high, step, hwconf=hw))  # type: ignore[union-attr]
        elif kif_payload is not None:
            kifs = kifs_payload_to_dicts(kif_payload)
            if kifs is not None and len(kifs) == 1:
                inputs.append(make_input_array_from_kif(shape, kifs[0], hw=hw))
            else:
                if kifs is None:
                    raise ValueError("input_kifs payload could not be converted")
                expected = _shape_size(shape)
                if expected is None:
                    raise ValueError(
                        f"input {idx}: per-element kifs require a fully concrete shape, got {shape}"
                    )
                if len(kifs) != expected:
                    raise ValueError(
                        f"input {idx}: got {len(kifs)} kifs for shape {shape}, expected {expected}"
                    )
                kifs_arr = np.array(kifs, dtype=object).reshape(shape)
                k_arr = np.vectorize(lambda d: int(bool(d["k"])))(kifs_arr)
                i_arr = np.vectorize(lambda d: int(d["i"]))(kifs_arr)
                f_arr = np.vectorize(lambda d: int(d["f"]))(kifs_arr)
                inputs.append(FixedVariableArray.from_kif(k_arr, i_arr, f_arr, hwconf=hw))  # type: ignore[union-attr]
        elif input_bws is not None:
            inputs.append(make_input_array(shape, bw_payload, hw=hw))
        else:
            raise ValueError("trace_lambda_result requires input_qints, input_kifs, or input_bws")

    output = body(*inputs)
    sol = comb_trace(_concat_inputs(inputs), output)
    return solution_to_result(sol, latency_cutoff=latency_cutoff)


def trace_lambda(
    input_shapes: list[tuple[int, ...]],
    input_bws: list[float],
    body: Callable[..., Any],
    *,
    adder_size: int = -1,
    carry_size: int = -1,
    latency_cutoff: int = -1,
) -> dict[str, Any]:
    return trace_lambda_result(
        input_shapes=input_shapes,
        input_bws=input_bws,
        body=body,
        adder_size=adder_size,
        carry_size=carry_size,
        latency_cutoff=latency_cutoff,
    )["cost"]


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
