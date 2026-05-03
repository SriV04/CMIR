from __future__ import annotations


def default_precision_record() -> dict:
    return {
        "qint": None,
        "kif": None,
        "bitwidth_bits": None,
        "tensor_width_bits": None,
        "shape": None,
        "source": "unknown",
        "quantizer": None,
    }


def default_precision_interface() -> dict:
    return {
        "input_precisions": None,
        "output_precisions": None,
        "input_qints": None,
        "input_kifs": None,
        "output_qints": None,
        "output_kifs": None,
        "input_tensor_width_bits": None,
        "output_tensor_width_bits": None,
        "precision_source": "unknown",
    }

