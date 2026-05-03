"""Interactive NN-IR viewer for the current JEDI-linear reference model."""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, str(REPO / "JEDI-linear" / "src"))
sys.path.insert(0, str(REPO / "heterograph"))


N_CONSTITUENTS = 8
USE_PERMINV = True
LOAD_TRAINED_WEIGHTS = True
VARIANT_DIR = "3-feature-perminv" if USE_PERMINV else "3-feature"
CHECKPOINT_GLOB = (
    REPO / "official_models" / VARIANT_DIR / f"jet_classifier_large_{N_CONSTITUENTS}" / "models" / "*.keras"
)


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _summarize_array(value, *, max_items: int = 32) -> dict[str, Any] | None:
    if value is None:
        return None
    arr = np.array(value)
    flat = arr.reshape(-1)
    preview = flat[:max_items].tolist()
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "preview": preview,
        "truncated": bool(flat.size > max_items),
    }


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "get_config"):
        try:
            return _json_safe(value.get_config())
        except Exception:
            pass
    if hasattr(value, "__dict__") and value.__dict__:
        return {
            str(k): _json_safe(v)
            for k, v in value.__dict__.items()
            if not str(k).startswith("_")
        }
    return value


def _node_details(graph, vx: int) -> dict[str, Any]:
    p = graph.pmap[vx]
    return {
        "kind": "node",
        "id": int(vx),
        "summary": {
            "layer_name": p.get("layer_name"),
            "layer_class": p.get("layer_class"),
            "op_kind": p.get("op_kind"),
            "equation": p.get("equation"),
            "activation": p.get("activation"),
            "kernel_shape": _json_safe(p.get("kernel_shape")),
            "in_shapes": _json_safe(p.get("in_shapes")),
            "out_shapes": _json_safe(p.get("out_shapes")),
            "uses_qkernel": p.get("uses_qkernel"),
            "has_bn": p.get("has_bn"),
            "bn_folded_into_qkernel": p.get("bn_folded_into_qkernel"),
            "num_params": p.get("num_params"),
            "iq_bw_avg": p.get("iq_bw_avg"),
            "kq_bw_avg": p.get("kq_bw_avg"),
            "bq_bw_avg": p.get("bq_bw_avg"),
        },
        "quantizers": _json_safe(
            {
                "iq": p.get("iq"),
                "kq": p.get("kq"),
                "bq": p.get("bq"),
                "oq": p.get("oq"),
                "aq": p.get("aq"),
                "iq_kif": p.get("iq_kif"),
                "kq_kif": p.get("kq_kif"),
                "bq_kif": p.get("bq_kif"),
                "oq_kif": p.get("oq_kif"),
            }
        ),
        "weights": {
            "kernel_sparsity": p.get("kernel_sparsity"),
            "kernel_nonzero_count": p.get("kernel_nonzero_count"),
            "kernel_zero_count": p.get("kernel_zero_count"),
            "kernel_unique_count": p.get("kernel_unique_count"),
            "kernel_histogram": _json_safe(p.get("kernel_value_histogram")),
            "weights": _summarize_array(p.get("weights")),
            "qkernel_values": _summarize_array(p.get("qkernel_values")),
            "bias_values": _summarize_array(p.get("bias_values")),
            "qbias_values": _summarize_array(p.get("qbias_values")),
        },
        "raw_properties": _json_safe(
            {
                key: value
                for key, value in p.items()
                if key
                not in {
                    "weights",
                    "kernel_values",
                    "kernel_float_values",
                    "qkernel_values",
                    "bias_values",
                    "qbias_values",
                    "kernel_unique_values",
                }
            }
        ),
    }


def _edge_details(graph, edge: tuple[int, int]) -> dict[str, Any]:
    p = graph.pmap[edge]
    return {
        "kind": "edge",
        "id": [int(edge[0]), int(edge[1])],
        "summary": _json_safe(
            {
                "tensor_shape": p.get("tensor_shape"),
                "bitwidth_src": p.get("bitwidth_src"),
                "bitwidth_dst": p.get("bitwidth_dst"),
                "tensor_width_bits": p.get("tensor_width_bits"),
                "volume_bits": p.get("volume_bits"),
                "has_quantization_boundary": p.get("has_quantization_boundary"),
                "needs_cast": p.get("needs_cast"),
                "cast_mode": p.get("cast_mode"),
                "src_kif": p.get("src_kif"),
                "dst_kif": p.get("dst_kif"),
            }
        ),
        "raw_properties": _json_safe(dict(p)),
    }


def graph_details(graph, elem) -> dict[str, Any]:
    if isinstance(elem, tuple):
        return _edge_details(graph, elem)
    return _node_details(graph, elem)


def _details_callback(graph, elem):
    return graph_details(graph, elem)


def _pretty_print_node_schema(graph, node_id: int) -> None:
    print(f"[jedi_linear_nn_ir] node {node_id} schema:")
    pprint.pprint(graph_details(graph, node_id), sort_dicts=False, width=120)

def _pretty_print_edge_schema(graph, edge: tuple[int, int]) -> None:
    print(f"[jedi_linear_nn_ir] edge {edge} schema:")
    pprint.pprint(graph_details(graph, edge), sort_dicts=False, width=120)


def _load_trained_model():
    import keras
    import hgq  # noqa: F401
    from model import get_gnn

    if LOAD_TRAINED_WEIGHTS:
        checkpoints = sorted(glob.glob(str(CHECKPOINT_GLOB)))
        if checkpoints:
            checkpoint = checkpoints[0]
            print(f"Loading trained model: {Path(checkpoint).name}")
            return keras.models.load_model(checkpoint), Path(checkpoint)

    conf = SimpleNamespace(n_constituents=N_CONSTITUENTS, pt_eta_phi=True)
    print("WARN: no trained checkpoint found, falling back to fresh model weights")
    return get_gnn(conf, uq1=USE_PERMINV), None


def main() -> None:
    from heterograph.webview import WebView
    from model import get_gnn

    nn_ir_builder = _load_path("nn_ir_builder_viewer", HERE / "NN-IR" / "builder.py")
    nn_styling = _load_path("nn_ir_styling_viewer", HERE / "NN-IR" / "styling.py")

    build_nn_ir = nn_ir_builder.build_nn_ir
    apply_nn_style = nn_styling.apply_nn_style

    model, checkpoint = _load_trained_model()
    conf = SimpleNamespace(n_constituents=N_CONSTITUENTS, pt_eta_phi=True)
    expected_model = get_gnn(conf, uq1=USE_PERMINV)

    graph = build_nn_ir(model, name="jedi_linear_nn_ir")
    apply_nn_style(graph)
    graph.style["!details"] = _details_callback

    model_desc = f"{VARIANT_DIR}, N={N_CONSTITUENTS}, weights={'trained' if checkpoint else 'fresh'}"
    print(f"[jedi_linear_nn_ir] keras layers: {len(model.layers)}")
    print(f"[jedi_linear_nn_ir] reference architecture layers: {len(expected_model.layers)}")
    print(f"[jedi_linear_nn_ir] nn-ir: {graph.num_vx} vertices, {graph.num_edges} edges")
    print(f"[jedi_linear_nn_ir] model: {model_desc}")

    _pretty_print_node_schema(graph, 1)

    if checkpoint is not None:
        print(f"[jedi_linear_nn_ir] checkpoint: {checkpoint.name}")

    viewer = WebView()
    viewer.add_graph(
        graph,
        title=f"JEDI-linear NN-IR — {model_desc}",
    )
    print("Serving on http://localhost:8888  (Ctrl-C to stop)")
    viewer.run(host="127.0.0.1", port="8888")


if __name__ == "__main__":
    main()
