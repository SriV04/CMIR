from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
_precision = _load_path("_sched_ir_precision", HERE / "precision.py")
_schedule = _load_path("_sched_ir_schedule", HERE / "schedule.py")


def default_node_properties() -> dict:
    base = {
        "nn_layer_idx": None,
        "nn_layer_name": None,
        "nn_op_kind": None,
        "decomp_index": None,
        "inserted_by": None,
        "op": None,
        "op_params": None,
        "kernel_type": None,
        "kernel_instance": None,
        "cost": None,
        "kernel_result": None,
        "reduce_mode": None,
        "schema_version": 2,
        "schema_notes": None,
    }
    base.update(_precision.default_precision_interface())
    base.update(_schedule.default_timing_fields())
    return base
