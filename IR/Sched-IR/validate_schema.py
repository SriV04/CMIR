from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "heterograph"))


REQUIRED_NODE_FIELDS = [
    "op",
    "op_params",
    "nn_layer_name",
    "inserted_by",
    "kernel_type",
    "kernel_instance",
    "cost",
    "fold_axes",
    "parallelism_N",
    "lanes_P",
    "temporal_steps_T",
    "ii",
    "latency_total",
    "t_start",
    "t_end",
]

REQUIRED_EDGE_FIELDS = [
    "tensor_shape",
    "bitwidth",
    "volume_bits",
    "edge_kind",
    "t_produce",
    "t_consume",
    "lifetime",
    "needs_buffer",
    "buffer_width_bits",
]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


schema_m = _load("_sched_ir_schema_validate", HERE / "schema.py")

from heterograph import HGraph  # noqa: E402


def _missing(props: dict, required: list[str]) -> list[str]:
    return [key for key in required if key not in props]


def main() -> None:
    g = HGraph(
        vinit=schema_m.vinit_sched,
        einit=schema_m.einit_sched,
        ginit=schema_m.ginit_sched,
    )
    u = g.add_vx()
    v = g.add_vx()
    e = g.add_edge(u, v)[0]

    missing_node = _missing(g.pmap[u], REQUIRED_NODE_FIELDS)
    missing_edge = _missing(g.pmap[e], REQUIRED_EDGE_FIELDS)

    if missing_node:
        raise SystemExit(f"Missing node fields: {missing_node}")
    if missing_edge:
        raise SystemExit(f"Missing edge fields: {missing_edge}")

    print("Sched-IR schema validation passed.")


if __name__ == "__main__":
    main()
