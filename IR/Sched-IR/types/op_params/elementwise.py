from __future__ import annotations


def default_elementwise_params() -> dict:
    return {
        "op_type": "elementwise",
        "elementwise_op": None,
        "op": None,
        "input_shapes": None,
        "output_shape": None,
        "broadcast": None,
        "input_qints": None,
        "input_kifs": None,
        "output_qint": None,
        "output_kif": None,
        "requires_input_alignment": False,
        "common_qint": None,
        "common_kif": None,
        "n_inputs": None,
        "in_shapes": None,
        "in_bws": None,
        "out_bw": None,
    }

