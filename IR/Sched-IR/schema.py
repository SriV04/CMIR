from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
_nodes = _load_path("_sched_ir_nodes", HERE / "types" / "nodes.py")
_edges = _load_path("_sched_ir_edges", HERE / "types" / "edges.py")
_graph = _load_path("_sched_ir_graph", HERE / "types" / "graph.py")
_enums = _load_path("_sched_ir_enums", HERE / "types" / "enums.py")

OP_PRIMITIVES = _enums.OP_PRIMITIVES
REDUCE_OPS = _enums.REDUCE_OPS
REDUCE_MODES = _enums.REDUCE_MODES
REDUCE_IMPL_MODES = _enums.REDUCE_IMPL_MODES
ELEMENTWISE_OPS = _enums.ELEMENTWISE_OPS
ACTIVATION_FUNCS = _enums.ACTIVATION_FUNCS
BUFFER_KINDS = _enums.BUFFER_KINDS
MUX_KINDS = _enums.MUX_KINDS
EDGE_KINDS = _enums.EDGE_KINDS
PRECISION_SOURCES = _enums.PRECISION_SOURCES
INSERTED_BY = _enums.INSERTED_BY


def vinit_sched(g, vx):
    g.pmap[vx] = _nodes.default_node_properties()


def einit_sched(g, e):
    g.pmap[e] = _edges.default_edge_properties()


def ginit_sched(g):
    g.pmap.update(_graph.default_graph_properties())
