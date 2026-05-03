"""Serializable HGQ metadata extraction for NN-IR."""

from __future__ import annotations

from typing import Any

import numpy as np


def safe_array(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "value"):
        value = value.value
    try:
        arr = np.array(value)
    except Exception:
        return None
    return arr


def safe_get_config(obj) -> dict[str, Any] | None:
    if obj is None or not hasattr(obj, "get_config"):
        return None
    try:
        cfg = obj.get_config()
    except Exception:
        return None
    return cfg if isinstance(cfg, dict) else None


def _to_scalar_or_array(value):
    arr = safe_array(value)
    if arr is None:
        return None
    if arr.shape == ():
        item = arr.item()
        return bool(item) if isinstance(item, (bool, np.bool_)) else item
    return arr


def extract_quantizer_variables(q) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if q is None or not hasattr(q, "variables"):
        return out
    for var in getattr(q, "variables", []) or []:
        name = getattr(var, "name", "")
        tag = name.split("/")[-1].split(":")[0]
        if tag in {"k", "i", "f", "b", "bits"}:
            arr = safe_array(var)
            if arr is not None:
                out[tag] = arr
    return out


def _granularity_from_shape(shape: tuple | None, place: str | None) -> str:
    if shape is None:
        return "unknown"
    if len(shape) == 0:
        return "scalar"
    if place == "kernel":
        return "per-weight"
    if place in {"input", "output", "activation"}:
        return "per-activation"
    if len(shape) == 1:
        return "per-channel"
    return "unknown"


def extract_quantizer_modes(q) -> dict[str, Any]:
    cfg = safe_get_config(q) or {}
    overflow_mode = (
        getattr(q, "overflow_mode", None)
        or cfg.get("overflow_mode")
        or cfg.get("overflow")
    )
    round_mode = (
        getattr(q, "round_mode", None)
        or cfg.get("round_mode")
        or cfg.get("rounding_mode")
    )
    symmetric = None
    if overflow_mode is not None:
        symmetric = "SYM" in str(overflow_mode).upper()
    elif "symmetric" in cfg:
        symmetric = bool(cfg["symmetric"])
    return {
        "overflow_mode": str(overflow_mode) if overflow_mode is not None else None,
        "round_mode": str(round_mode) if round_mode is not None else None,
        "symmetric": symmetric,
    }


def extract_kif(q, *, place: str | None = None) -> dict[str, Any] | None:
    if q is None:
        return None

    direct_k = getattr(q, "k", None)
    direct_i = getattr(q, "i", None)
    direct_f = getattr(q, "f", None)
    direct_b = getattr(q, "b", None)
    direct_kif = getattr(q, "kif", None)

    k = i = f = bits = None

    if direct_kif is not None:
        try:
            k, i, f = direct_kif
        except Exception:
            pass

    if k is None:
        k = direct_k
    if i is None:
        i = direct_i
    if f is None:
        f = direct_f
    if bits is None:
        bits = direct_b or getattr(q, "bits", None)

    vars_map = extract_quantizer_variables(q)
    k = vars_map.get("k", k)
    i = vars_map.get("i", i)
    f = vars_map.get("f", f)
    bits = vars_map.get("bits", vars_map.get("b", bits))

    cfg = safe_get_config(q) or {}
    k = cfg.get("k", k)
    i = cfg.get("i", cfg.get("integers", i))
    f = cfg.get("f", cfg.get("fractional", f))
    bits = cfg.get("bits", cfg.get("b", bits))

    k_val = _to_scalar_or_array(k)
    i_val = _to_scalar_or_array(i)
    f_val = _to_scalar_or_array(f)
    bits_val = _to_scalar_or_array(bits)

    if bits_val is None and i_val is not None and f_val is not None:
        bits_val = np.array(i_val) + np.array(f_val) + np.array(k_val if k_val is not None else 0)
        bits_val = _to_scalar_or_array(bits_val)
    elif f_val is None and bits_val is not None and i_val is not None:
        f_val = _to_scalar_or_array(np.array(bits_val) - np.array(i_val) - np.array(k_val if k_val is not None else 0))

    if k_val is None and i_val is None and f_val is None and bits_val is None:
        return None

    shape = None
    for candidate in (bits_val, f_val, i_val, k_val):
        if isinstance(candidate, np.ndarray):
            shape = tuple(candidate.shape)
            break
    if shape is None:
        shape = ()

    granularity = _granularity_from_shape(shape, place)
    return {
        "k": k_val,
        "i": i_val,
        "f": f_val,
        "bits": bits_val,
        "shape": None if shape is None else tuple(shape),
        "granularity": granularity,
    }


def bitwidth_from_kif(kif: dict[str, Any] | None) -> float | None:
    if not kif:
        return None
    bits = kif.get("bits")
    if bits is not None:
        arr = np.array(bits, dtype=float)
        return float(arr.max()) if arr.size else None
    i = kif.get("i")
    f = kif.get("f")
    k = kif.get("k")
    if i is None or f is None:
        return None
    arr = np.array(i, dtype=float) + np.array(f, dtype=float)
    if k is not None:
        arr = arr + np.array(k, dtype=float)
    return float(arr.max()) if arr.size else None


def kif_to_qint(kif: dict[str, Any] | None, *, symmetric: bool | None = None) -> dict[str, Any] | None:
    if not kif:
        return None
    k = kif.get("k", 0)
    i = kif.get("i")
    f = kif.get("f")
    if i is None or f is None:
        return None

    k_arr = np.array(k, dtype=float)
    i_arr = np.array(i, dtype=float)
    f_arr = np.array(f, dtype=float)
    step = np.power(2.0, -f_arr)
    max_val = np.power(2.0, i_arr) - step
    if symmetric:
        min_val = -k_arr * max_val
    else:
        min_val = -k_arr * np.power(2.0, i_arr)

    def _maybe_scalar(arr: np.ndarray):
        return arr.item() if arr.shape == () else arr

    return {
        "min": _maybe_scalar(min_val),
        "max": _maybe_scalar(max_val),
        "step": _maybe_scalar(step),
    }


def quantizer_summary(q, *, place: str | None = None) -> dict[str, Any] | None:
    if q is None:
        return None
    cfg = safe_get_config(q) or {}
    modes = extract_quantizer_modes(q)
    kif = extract_kif(q, place=place)
    qint = kif_to_qint(kif, symmetric=modes.get("symmetric")) if kif else None
    shape = kif.get("shape") if kif else None
    granularity = kif.get("granularity", "unknown") if kif else "unknown"

    return {
        "exists": True,
        "class_name": type(q).__name__,
        "config": cfg,
        "kif": kif,
        "qint": qint,
        "overflow_mode": modes.get("overflow_mode"),
        "round_mode": modes.get("round_mode"),
        "granularity": granularity,
        "shape": shape,
        "place": place,
        "source": "HGQ",
    }


def _bitwidth_stats(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary or not summary.get("kif"):
        return {"avg": None, "max": None, "min": None, "shape": None}
    bits = summary["kif"].get("bits")
    if bits is None:
        return {"avg": None, "max": None, "min": None, "shape": summary.get("shape")}
    arr = np.array(bits, dtype=float)
    if arr.size == 0:
        return {"avg": None, "max": None, "min": None, "shape": summary.get("shape")}
    return {
        "avg": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "shape": tuple(arr.shape),
    }


def bw_array(quantizer) -> np.ndarray | None:
    summary = quantizer_summary(quantizer)
    if not summary or not summary.get("kif"):
        return None
    bits = summary["kif"].get("bits")
    return np.array(bits, dtype=float) if bits is not None else None


def avg_bw(quantizer) -> float | None:
    return _bitwidth_stats(quantizer_summary(quantizer)).get("avg")


def max_bw(quantizer) -> float | None:
    return _bitwidth_stats(quantizer_summary(quantizer)).get("max")


def min_bw(quantizer) -> float | None:
    return _bitwidth_stats(quantizer_summary(quantizer)).get("min")


def find_output_quantizer(layer) -> dict[str, Any] | None:
    for attr in ("oq", "_oq"):
        q = getattr(layer, attr, None)
        if q is not None:
            return quantizer_summary(q, place="output")
    return None


def find_activation_quantizer(layer) -> dict[str, Any] | None:
    for attr in ("aq", "_aq"):
        q = getattr(layer, attr, None)
        if q is not None:
            return quantizer_summary(q, place="activation")
    return None


def extract_all_quantizers(layer) -> dict[str, dict[str, Any] | None]:
    return {
        "iq": quantizer_summary(getattr(layer, "iq", None), place="input"),
        "kq": quantizer_summary(getattr(layer, "kq", None), place="kernel"),
        "bq": quantizer_summary(getattr(layer, "bq", None), place="bias"),
        "oq": find_output_quantizer(layer),
        "aq": find_activation_quantizer(layer),
    }


def _extract_batchnorm_values(layer) -> dict[str, np.ndarray] | None:
    names = {
        "bn_gamma": getattr(layer, "bn_gamma", None),
        "bn_beta": getattr(layer, "bn_beta", None),
        "moving_mean": getattr(layer, "moving_mean", None),
        "moving_variance": getattr(layer, "moving_variance", None),
    }
    out = {name: safe_array(value) for name, value in names.items() if safe_array(value) is not None}
    return out or None


def extract_layer_values(layer) -> dict[str, Any]:
    kernel = safe_array(getattr(layer, "kernel", None))
    qkernel = safe_array(getattr(layer, "qkernel", None))
    bias = safe_array(getattr(layer, "bias", None))
    qbias = safe_array(getattr(layer, "qbias", None))
    uses_qkernel = qkernel is not None
    kernel_values = qkernel if qkernel is not None else kernel

    return {
        "kernel_values": kernel_values,
        "kernel_float_values": kernel,
        "bias_values": bias,
        "batchnorm_values": _extract_batchnorm_values(layer),
        "qkernel_values": qkernel,
        "qbias_values": qbias,
        "uses_qkernel": uses_qkernel,
    }


def weight_stats(arr, *, include_histogram: bool = True) -> dict[str, Any]:
    data = safe_array(arr)
    if data is None or data.size == 0:
        return {
            "sparsity": None,
            "nonzero_count": None,
            "zero_count": None,
            "unique_values": None,
            "unique_count": None,
            "value_histogram": None,
            "min": None,
            "max": None,
            "dtype": None,
        }

    flat = data.reshape(-1)
    zero_count = int((np.abs(flat) <= 1e-12).sum())
    nonzero_count = int(flat.size - zero_count)
    unique_vals, counts = np.unique(flat, return_counts=True)
    histogram = None
    if include_histogram:
        histogram = [
            {"value": float(value), "count": int(count)}
            for value, count in zip(unique_vals[:32], counts[:32], strict=False)
        ]

    return {
        "sparsity": float(zero_count / flat.size),
        "nonzero_count": nonzero_count,
        "zero_count": zero_count,
        "unique_values": unique_vals,
        "unique_count": int(unique_vals.size),
        "value_histogram": histogram,
        "min": float(flat.min()),
        "max": float(flat.max()),
        "dtype": str(data.dtype),
    }
