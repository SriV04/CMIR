from __future__ import annotations


def default_buffer_params() -> dict:
    return {
        "op_type": "buffer",
        "tensor_shape": None,
        "element_qint": None,
        "element_kif": None,
        "element_width_bits": None,
        "tensor_width_bits": None,
        "depth": None,
        "lifetime_cycles": None,
        "buffer_kind": None,
        "width_bits": None,
        "total_bits": None,
        "source_edge": None,
        "reason": None,
    }

