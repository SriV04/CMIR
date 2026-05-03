from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
_nodes = _load_path("_nn_ir_nodes", HERE / "types" / "nodes.py")
_edges = _load_path("_nn_ir_edges", HERE / "types" / "edges.py")


def vinit_nn(g, vx):
    g.pmap[vx] = _nodes.default_node_properties()


def einit_nn(g, e):
    g.pmap[e] = _edges.default_edge_properties()


def ginit_nn(g):
    g.pmap["name"] = None
    g.pmap["model_source"] = None
    g.pmap["n_features"] = None
    g.pmap["n_classes"] = None
