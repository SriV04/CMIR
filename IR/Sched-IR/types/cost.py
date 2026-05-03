from __future__ import annotations


def default_cost_dict() -> dict:
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

