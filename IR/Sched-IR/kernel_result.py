from __future__ import annotations

from typing import Any


_COST_KEYS = {"lut", "ff", "dsp", "bram", "latency_cycles", "ii"}


def empty_cost() -> dict[str, Any]:
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


def empty_kernel_result(*, source: str = "unknown") -> dict[str, Any]:
    return {
        "cost": empty_cost(),
        "input_qints": None,
        "input_kifs": None,
        "output_qints": None,
        "output_kifs": None,
        "input_bitwidths": None,
        "output_bitwidths": None,
        "input_tensor_width_bits": None,
        "output_tensor_width_bits": None,
        "precision_source": source,
        "da4ml": None,
    }


def is_cost_dict(obj: dict) -> bool:
    return isinstance(obj, dict) and _COST_KEYS.issubset(obj.keys()) and "cost" not in obj


def normalize_kernel_result(obj: dict | None, *, source: str = "unknown") -> dict[str, Any]:
    if obj is None:
        obj = empty_cost()

    if isinstance(obj, dict) and "cost" in obj:
        result = dict(obj)
        if result.get("precision_source") is None:
            result["precision_source"] = source
        result.setdefault("input_qints", None)
        result.setdefault("input_kifs", None)
        result.setdefault("output_qints", None)
        result.setdefault("output_kifs", None)
        result.setdefault("input_bitwidths", None)
        result.setdefault("output_bitwidths", None)
        result.setdefault("input_tensor_width_bits", None)
        result.setdefault("output_tensor_width_bits", None)
        result.setdefault("da4ml", None)
        return result

    if is_cost_dict(obj):
        result = empty_kernel_result(source=source)
        result["cost"] = obj
        return result

    raise TypeError(f"Invalid kernel result shape: {type(obj)} {obj}")
