from __future__ import annotations


def default_timing_fields() -> dict:
    return {
        "parallelism_N": None,
        "lanes_P": None,
        "temporal_steps_T": None,
        "pipeline_latency_L": None,
        "elements_per_cycle": None,
        "ii": None,
        "latency_total": None,
        "fold_factor": None,
        "physical_instances": None,
        "fold_axes": None,
        "fold_iteration": None,
        "fold_group": None,
        "reuse_group": None,
        "t_ready": None,
        "t_start": None,
        "t_end": None,
        "critical_path": False,
    }

