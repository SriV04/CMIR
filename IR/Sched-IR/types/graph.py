from __future__ import annotations


def default_graph_properties() -> dict:
    return {
        "name": None,
        "source_nn_ir": None,
        "resource_yaml": None,
        "schema_version": 2,
        "target_device": None,
        "target_fmax": None,
        "target_slrs": None,
        "fpga_config": None,
        "objective": None,
        "area_budget": None,
        "latency_budget": None,
        "fold_plan": None,
        "makespan": None,
        "initiation_interval": None,
        "pipeline_depth": None,
        "total_luts": None,
        "total_ffs": None,
        "total_dsps": None,
        "total_brams": None,
        "total_urams": None,
        "precision_propagated": False,
        "precision_warnings": None,
        "kernel_utilization": None,
        "critical_path": None,
    }

