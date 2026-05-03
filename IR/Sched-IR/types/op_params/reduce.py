from __future__ import annotations


def default_reduce_params() -> dict:
    return {
        "op_type": "reduce",
        "mode": None,
        "axes": None,
        "keepdims": None,
        "input_shape": None,
        "output_shape": None,
        "reduction_width": None,
        "input_qint": None,
        "input_kif": None,
        "output_qint": None,
        "output_kif": None,
        "partial_sum_qint": None,
        "partial_sum_kif": None,
        "accumulator_qint": None,
        "accumulator_kif": None,
        "reduce_mode": None,
        "spatial_width_P": None,
        "temporal_steps_T": None,
        "scale": None,
        "scale_qint": None,
        "scale_kif": None,
        "in_shape": None,
        "in_bw": None,
        "out_bw": None,
    }

