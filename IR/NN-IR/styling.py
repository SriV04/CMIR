"""Compatibility shim for NN-IR styling."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_nn_ir_graph_style",
        here / "graphing" / "graph_style.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_impl = _load()

OP_COLORS = _impl.OP_COLORS
OP_SHAPES = _impl.OP_SHAPES
vx_label = _impl.vx_label
edge_label = _impl.edge_label
edge_penwidth = _impl.edge_penwidth
apply_nn_style = _impl.apply_nn_style
