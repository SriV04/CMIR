from __future__ import annotations


def common_op_params() -> dict:
    return {
        "op_type": None,
        "input_shapes": None,
        "output_shapes": None,
        "input_qints": None,
        "input_kifs": None,
        "output_qints": None,
        "output_kifs": None,
        "has_explicit_output_quantizer": False,
        "output_quantizer": None,
        "in_bw": None,
        "out_bw": None,
    }

