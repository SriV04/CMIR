from __future__ import annotations


def default_activation_params() -> dict:
    return {
        "op_type": "activation",
        "func": None,
        "input_shape": None,
        "output_shape": None,
        "input_qint": None,
        "input_kif": None,
        "output_qint": None,
        "output_kif": None,
        "activation_quantizer": None,
        "output_quantizer": None,
        "implementation": None,
        "lut_entries": None,
        "lut_input_qint": None,
        "lut_output_qint": None,
        "in_shape": None,
        "in_bw": None,
        "out_bw": None,
    }

