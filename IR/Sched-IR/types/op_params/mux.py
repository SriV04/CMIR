from __future__ import annotations


def default_mux_params() -> dict:
    return {
        "op_type": "mux",
        "n_inputs": None,
        "select_bits": None,
        "input_qints": None,
        "input_kifs": None,
        "output_qint": None,
        "output_kif": None,
        "width_bits": None,
        "source_edges": None,
        "select_schedule": None,
        "mux_kind": None,
    }

