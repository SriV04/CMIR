"""NN-IR → unscheduled Sched-IR decomposer.

This is the first pass of Sched-IR creation: a compiler-independent lowering
of every NN-IR vertex into one (or more) of the six Sched-IR primitives
defined in ``schema.py``:

    dense | reduce | elementwise | activation | buffer | mux

No kernel binding, no folding, no scheduling, no buffer/mux insertion — those
all run later. This pass only produces the *shape* of the graph and fills in
op_params so downstream phases have everything they need.

Usage (from a notebook or main.py):

    import importlib.util
    from pathlib import Path

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    nn_builder = _load("nn_ir_builder", Path("IR/NN-IR/builder.py"))
    sched_decomp = _load("sched_decomposer", Path("IR/Sched-IR/decomposer.py"))

    g_nn    = nn_builder.build_nn_ir(model, name="jedi_gnn")
    g_sched = sched_decomp.decompose_nn_to_sched(g_nn)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from heterograph import HGraph


# --------------------------------------------------------------------------- #
# Load the sibling schema module (IR/Sched-IR has a hyphen, not a valid package)
# --------------------------------------------------------------------------- #

def _load_sibling(name: str):
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(f"_sched_ir_{name}", here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_schema = _load_sibling("schema")
vinit_sched = _schema.vinit_sched
einit_sched = _schema.einit_sched
ginit_sched = _schema.ginit_sched


# --------------------------------------------------------------------------- #
# Shape inference helpers
# --------------------------------------------------------------------------- #

def _first(lst):
    return lst[0] if lst else None


def _infer_reduction(in_shape, out_shape) -> tuple[list[int] | None, int | None, bool | None]:
    """Return (axes, reduction_width, keepdims) for a reduction from in to out.

    Handles both ``keepdims=True`` (same rank, reduced dims become 1) and
    ``keepdims=False`` (reduced dims are dropped). Axis indices are given in
    the producer's shape, batch dim included.
    """
    if in_shape is None or out_shape is None:
        return None, None, None

    in_dims = list(in_shape)
    out_dims = list(out_shape)

    axes: list[int] = []
    width = 1

    if len(in_dims) == len(out_dims):
        keepdims = True
        for i, (a, b) in enumerate(zip(in_dims, out_dims)):
            if a != b:
                axes.append(i)
                if a is not None:
                    width *= a
    else:
        keepdims = False
        # Greedy match: skip dims of in_shape that don't appear at the next
        # position of out_shape. Correct for simple reductions like
        # (B, N, C) -> (B, C) (axis 1 dropped).
        j = 0
        for i, a in enumerate(in_dims):
            if j < len(out_dims) and a == out_dims[j]:
                j += 1
            else:
                axes.append(i)
                if a is not None:
                    width *= a

    return axes, (width if width > 1 else None), keepdims


def _infer_broadcast(in_shapes, out_shape) -> dict[int, list[int]] | None:
    """For each input port, return the axes that are broadcast (size 1 → size N)."""
    if not in_shapes or out_shape is None:
        return None
    result: dict[int, list[int]] = {}
    for port, in_shape in enumerate(in_shapes):
        if in_shape is None or len(in_shape) != len(out_shape):
            continue
        axes = [i for i, (a, b) in enumerate(zip(in_shape, out_shape)) if a == 1 and b not in (None, 1)]
        if axes:
            result[port] = axes
    return result or None


def _guess_fold_axes(in_shape) -> list[int] | None:
    """Cheap fold-axis detection — returns tensor dimension indices.

    Any concrete dim > 1 that sits *after* the batch dim (index 0) is a
    candidate fold axis: by definition the op replicates the same
    computation across those elements, so they're foldable. Indices are
    given in the producer's shape with the batch dim included, matching the
    convention used by ``reduce.axes``.

    Shapes of the form ``(batch, C)`` have no foldable axes — dim 0 is the
    batch and dim 1 is the channel axis that the op actually operates on.

    The scheduler can override this once we support more architectures or
    want to fold over additional axes (e.g. channels for a tiled dense).
    """
    if in_shape is None:
        return None
    dims = list(in_shape)
    if len(dims) < 3:
        return None
    axes = [i for i, d in enumerate(dims[1:-1], start=1) if d not in (None, 1)]
    return axes or None


def _out_bw_add(in_bws) -> float | None:
    """Bitwidth growth for an elementwise add: max(in_bws) + 1."""
    vals = [b for b in (in_bws or []) if b is not None]
    return max(vals) + 1 if vals else None


def _out_bw_reduce_sum(in_bw, width) -> float | None:
    """Bitwidth growth for a k-input sum tree: in_bw + ceil(log2(k))."""
    if in_bw is None or width is None or width <= 1:
        return in_bw
    import math
    return in_bw + math.ceil(math.log2(width))


# --------------------------------------------------------------------------- #
# Per-NN-vertex lowering
# --------------------------------------------------------------------------- #

def _lower_dense(p: dict) -> dict[str, Any]:
    return {
        "kernel_shape": p.get("kernel_shape"),
        "equation":     p.get("equation"),
        "in_bw":        p.get("iq_bw"),
        "kq_bw":        p.get("kq_bw"),
        "out_bw":       None,   # filled by the kernel cost query later
        "sparsity":     p.get("sparsity"),
        "activation":   p.get("activation"),
        "has_bn":       p.get("op_kind") == "einsum_dense_bn",
        "has_bias":     p.get("bq_bw") is not None,
    }


def _lower_reduce(p: dict) -> dict[str, Any]:
    in_shape = _first(p.get("in_shapes"))
    out_shape = _first(p.get("out_shapes"))
    axes, width, keepdims = _infer_reduction(in_shape, out_shape)
    in_bw = p.get("iq_bw")
    mode = "sum" if p.get("op_kind") == "qsum" else "N/A - ERROR"  # default to sum if not specified
    return {
        "mode":             mode,      # QSum; mean/max would set 'mean'/'max'
        "axes":             axes,
        "in_shape":         in_shape,
        "in_bw":            in_bw,
        "out_bw":           _out_bw_reduce_sum(in_bw, width),
        "reduction_width":  width,
        "keepdims":         keepdims,
        "scale":            None,
    }


def _lower_elementwise(p: dict) -> dict[str, Any]:
    in_shapes = p.get("in_shapes") or []
    out_shape = _first(p.get("out_shapes"))
    # NN-IR stores a single iq_bw summary; use it for every port until we get
    # per-port quantizer info.
    in_bw = p.get("iq_bw")
    in_bws = [in_bw for _ in in_shapes] if in_bw is not None else None
    return {
        "op":         "add",   # QAdd → 'add'; other elementwise ops will set this
        "in_shapes":  in_shapes,
        "in_bws":     in_bws,
        "out_bw":     _out_bw_add(in_bws),
        "broadcast":  _infer_broadcast(in_shapes, out_shape),
    }


def _lower_activation(p: dict) -> dict[str, Any]:
    in_shape = _first(p.get("in_shapes"))
    in_bw = p.get("iq_bw")
    return {
        "func":        p.get("activation") or "linear",
        "in_shape":    in_shape,
        "in_bw":       in_bw,
        "out_bw":      in_bw,
        "lut_entries": None,
    }


_LOWERING = {
    "einsum_dense_bn": ("dense",       _lower_dense),
    "einsum_dense":    ("dense",       _lower_dense),
    "dense":           ("dense",       _lower_dense),
    "qsum":            ("reduce",      _lower_reduce),
    "qadd":            ("elementwise", _lower_elementwise),
    "activation":      ("activation",  _lower_activation),
}


def _lower_vertex(p: dict) -> tuple[str | None, dict | None]:
    op_kind = p.get("op_kind")
    if op_kind == "input":
        return None, None   # graph inputs carry no computation
    if op_kind not in _LOWERING:
        raise NotImplementedError(
            f"Sched-IR decomposer has no lowering for NN-IR op_kind {op_kind!r} "
            f"(layer {p.get('layer_name')!r})"
        )
    prim, fn = _LOWERING[op_kind]
    return prim, fn(p)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def decompose_nn_to_sched(nn_g: HGraph, name: str | None = None) -> HGraph:
    """Return an unscheduled Sched-IR graph built from an NN-IR graph.

    One sched vertex per NN-IR vertex (except ``input``, which is elided).
    Each sched vertex gets ``op``, ``op_params``, ``fold_axes`` and
    provenance fields set. Kernel binding, folding, and timing are all left
    at their schema defaults — later phases fill them in.
    """
    sg = HGraph(vinit=vinit_sched, einit=einit_sched, ginit=ginit_sched)

    sg.pmap["name"] = name or f"{nn_g.pmap.get('name') or 'unnamed'}_sched"
    sg.pmap["source_nn_ir"] = nn_g.pmap.get("name")
    sg.pmap["objective"] = None   # the scheduler will set this

    # ---- vertices ----------------------------------------------------- #
    nn_to_sched: dict[int, int | None] = {}
    for nn_vx in nn_g.vertices:
        p = nn_g.pmap[nn_vx]
        prim, params = _lower_vertex(p)
        if prim is None:
            nn_to_sched[nn_vx] = None
            continue

        sv = sg.add_vx()
        nn_to_sched[nn_vx] = sv

        sp = sg.pmap[sv]
        sp["nn_layer_idx"]  = p.get("layer_idx")
        sp["nn_layer_name"] = p.get("layer_name")
        sp["nn_op_kind"]    = p.get("op_kind")
        sp["decomp_index"]  = 0
        sp["inserted_by"]   = "decomposer"
        sp["op"]            = prim
        sp["op_params"]     = params
        sp["fold_axes"]     = _guess_fold_axes(_first(p.get("in_shapes")))

        # Reduction-specific default: spatial tree until the scheduler
        # flips it to temporal_accumulate during FOLD.
        if prim == "reduce":
            sp["reduce_mode"] = "spatial"

    # ---- edges -------------------------------------------------------- #
    for nn_edge in nn_g.edges:
        s_nn, t_nn = nn_edge
        s = nn_to_sched.get(s_nn)
        t = nn_to_sched.get(t_nn)
        if s is None or t is None:
            continue   # edge touches an elided vertex (e.g. input layer)

        created = sg.add_edge(s, t)
        if not created:
            continue
        e = created[0]

        ep = nn_g.pmap[nn_edge]
        sp = sg.pmap[e]
        sp["tensor_shape"] = ep.get("tensor_shape")
        # Prefer the consumer-side bitwidth; fall back to the source-side.
        sp["bitwidth"]     = ep.get("bitwidth_dst") or ep.get("bitwidth_src")
        sp["volume_bits"]  = ep.get("volume_bits")
        sp["edge_kind"]    = "data"

    return sg
