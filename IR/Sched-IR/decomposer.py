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


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_schema = _load_sibling("schema")
vinit_sched = _schema.vinit_sched
einit_sched = _schema.einit_sched
ginit_sched = _schema.ginit_sched


def _load_op_param_defaults():
    here = Path(__file__).resolve().parent
    base = here / "types" / "op_params"
    return {
        "dense": _load_path("_sched_dense_params", base / "dense.py").default_dense_params,
        "reduce": _load_path("_sched_reduce_params", base / "reduce.py").default_reduce_params,
        "elementwise": _load_path("_sched_elementwise_params", base / "elementwise.py").default_elementwise_params,
        "activation": _load_path("_sched_activation_params", base / "activation.py").default_activation_params,
    }


_OP_PARAM_DEFAULTS = _load_op_param_defaults()


# --------------------------------------------------------------------------- #
# Shape inference helpers
# --------------------------------------------------------------------------- #

def _first(lst):
    return lst[0] if lst else None


def _first_not_none(*vals):
    for value in vals:
        if value is not None:
            return value
    return None


def _copy_first_present(src: dict, keys: list[str]):
    for key in keys:
        value = src.get(key)
        if value is not None:
            return value
    return None


def _bits_from_kif(kif):
    """Return a scalar bitwidth when the KIF payload is scalar-ish."""
    if kif is None:
        return None
    bits = kif.get("bits") if isinstance(kif, dict) else None
    if bits is None:
        return None
    if isinstance(bits, (int, float)):
        return float(bits)
    if hasattr(bits, "item") and getattr(bits, "shape", ()) == ():
        return float(bits.item())
    if hasattr(bits, "size") and getattr(bits, "size", None) == 1:
        return float(bits.reshape(-1)[0])
    return None


def _precision_record_from_nn(p: dict, prefix: str) -> dict | None:
    qint = p.get(f"{prefix}_qint")
    kif = p.get(f"{prefix}_kif")
    quantizer = p.get(prefix)
    if qint is None and kif is None and quantizer is None:
        return None
    return {
        "qint": qint,
        "kif": kif,
        "bitwidth_bits": _bits_from_kif(kif),
        "tensor_width_bits": None,
        "shape": p.get(f"{prefix}_bw_shape"),
        "source": "hgq",
        "quantizer": quantizer,
    }


def _legacy_bw_from_nn(p: dict, prefix: str) -> float | None:
    return _copy_first_present(
        p,
        [f"{prefix}_bw_avg", f"{prefix}_bw", f"{prefix}_bw_max"],
    )


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


def _activation_impl_guess(func):
    if func in (None, "linear"):
        return "free"
    if func == "relu":
        return "free_sign_clip"
    if func in ("sigmoid", "tanh", "softmax"):
        return "lookup_table"
    return "unsupported"


# --------------------------------------------------------------------------- #
# Per-NN-vertex lowering
# --------------------------------------------------------------------------- #

def _lower_dense(p: dict) -> dict[str, Any]:
    params = _OP_PARAM_DEFAULTS["dense"]()
    params.update(
        {
            "equation": p.get("equation"),
            "kernel_shape": p.get("kernel_shape"),
            "input_shape": _first(p.get("in_shapes")),
            "output_shape": _first(p.get("out_shapes")),
            "kernel_values": _first_not_none(
                p.get("qkernel_values"),
                p.get("kernel_values"),
                p.get("weights"),
            ),
            "kernel_float_values": p.get("kernel_float_values"),
            "qkernel_values": p.get("qkernel_values"),
            "bias_values": _first_not_none(p.get("bias_values"), p.get("biases")),
            "qbias_values": p.get("qbias_values"),
            "uses_qkernel": bool(p.get("uses_qkernel")),
            "input_quantizer": p.get("iq"),
            "kernel_quantizer": p.get("kq"),
            "bias_quantizer": p.get("bq"),
            "output_quantizer": _first_not_none(p.get("oq"), p.get("aq")),
            "input_qint": p.get("iq_qint"),
            "input_kif": p.get("iq_kif"),
            "kernel_qint": p.get("kq_qint"),
            "kernel_kif": p.get("kq_kif"),
            "bias_qint": p.get("bq_qint"),
            "bias_kif": p.get("bq_kif"),
            "output_qint": p.get("oq_qint"),
            "output_kif": p.get("oq_kif"),
            "has_bias": any(
                value is not None
                for value in (p.get("bias_values"), p.get("qbias_values"), p.get("bq"))
            ),
            "has_bn": bool(p.get("has_bn")) or p.get("op_kind") == "einsum_dense_bn",
            "bn_folded_into_qkernel": bool(p.get("bn_folded_into_qkernel")),
            "activation": p.get("activation"),
            "kernel_sparsity": _first_not_none(p.get("kernel_sparsity"), p.get("sparsity")),
            "kernel_nonzero_count": p.get("kernel_nonzero_count"),
            "kernel_zero_count": p.get("kernel_zero_count"),
            "kernel_unique_values": p.get("kernel_unique_values"),
            "kernel_unique_count": p.get("kernel_unique_count"),
            "kernel_min": p.get("kernel_min"),
            "kernel_max": p.get("kernel_max"),
            "kernel_dtype": p.get("kernel_dtype"),
            "in_bw": _legacy_bw_from_nn(p, "iq"),
            "kq_bw": _legacy_bw_from_nn(p, "kq"),
            "out_bw": None,
            "sparsity": _first_not_none(p.get("kernel_sparsity"), p.get("sparsity")),
        }
    )
    return params


def _lower_reduce(p: dict) -> dict[str, Any]:
    params = _OP_PARAM_DEFAULTS["reduce"]()
    in_shape = _first(p.get("in_shapes"))
    out_shape = _first(p.get("out_shapes"))
    axes, width, keepdims = _infer_reduction(in_shape, out_shape)
    in_bw = _legacy_bw_from_nn(p, "iq")
    mode = "sum" if p.get("op_kind") == "qsum" else "N/A - ERROR"  # default to sum if not specified
    params.update(
        {
            "mode": mode,
            "axes": axes,
            "keepdims": keepdims,
            "input_shape": in_shape,
            "output_shape": out_shape,
            "reduction_width": width,
            "input_qint": p.get("iq_qint"),
            "input_kif": p.get("iq_kif"),
            "output_qint": p.get("oq_qint"),
            "output_kif": p.get("oq_kif"),
            "partial_sum_qint": None,
            "partial_sum_kif": None,
            "accumulator_qint": None,
            "accumulator_kif": None,
            "reduce_mode": "spatial",
            "spatial_width_P": None,
            "temporal_steps_T": None,
            "scale": None,
            "scale_qint": None,
            "scale_kif": None,
            "in_shape": in_shape,
            "in_bw": in_bw,
            "out_bw": _out_bw_reduce_sum(in_bw, width),
        }
    )
    return params


def _lower_elementwise(p: dict) -> dict[str, Any]:
    params = _OP_PARAM_DEFAULTS["elementwise"]()
    in_shapes = p.get("in_shapes") or []
    out_shape = _first(p.get("out_shapes"))
    in_bw = _legacy_bw_from_nn(p, "iq")
    in_bws = [in_bw for _ in in_shapes] if in_bw is not None else None
    params.update(
        {
            "elementwise_op": "add",
            "op": "add",
            "input_shapes": in_shapes,
            "output_shape": out_shape,
            "broadcast": _infer_broadcast(in_shapes, out_shape),
            "input_qints": [p.get("iq_qint") for _ in in_shapes] if p.get("iq_qint") is not None else None,
            "input_kifs": [p.get("iq_kif") for _ in in_shapes] if p.get("iq_kif") is not None else None,
            "output_qint": p.get("oq_qint"),
            "output_kif": p.get("oq_kif"),
            "requires_input_alignment": False,
            "common_qint": None,
            "common_kif": None,
            "n_inputs": len(in_shapes),
            "in_shapes": in_shapes,
            "in_bws": in_bws,
            "out_bw": _out_bw_add(in_bws),
        }
    )
    return params


def _lower_activation(p: dict) -> dict[str, Any]:
    params = _OP_PARAM_DEFAULTS["activation"]()
    in_shape = _first(p.get("in_shapes"))
    out_shape = _first(p.get("out_shapes"))
    in_bw = _legacy_bw_from_nn(p, "iq")
    func = p.get("activation") or "linear"
    params.update(
        {
            "func": func,
            "input_shape": in_shape,
            "output_shape": out_shape,
            "input_qint": p.get("iq_qint"),
            "input_kif": p.get("iq_kif"),
            "output_qint": p.get("oq_qint"),
            "output_kif": p.get("oq_kif"),
            "activation_quantizer": p.get("aq"),
            "output_quantizer": p.get("oq"),
            "implementation": _activation_impl_guess(func),
            "lut_entries": None,
            "lut_input_qint": None,
            "lut_output_qint": None,
            "in_shape": in_shape,
            "in_bw": in_bw,
            "out_bw": _first_not_none(_legacy_bw_from_nn(p, "oq"), in_bw),
        }
    )
    return params


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


def _apply_node_precision_from_params(sp: dict, params: dict, prim: str):
    in_qints = None
    in_kifs = None
    out_qints = None
    out_kifs = None

    if prim == "dense":
        in_qints = [params.get("input_qint")] if params.get("input_qint") is not None else None
        in_kifs = [params.get("input_kif")] if params.get("input_kif") is not None else None
        out_qints = [params.get("output_qint")] if params.get("output_qint") is not None else None
        out_kifs = [params.get("output_kif")] if params.get("output_kif") is not None else None
    elif prim == "reduce":
        in_qints = [params.get("input_qint")] if params.get("input_qint") is not None else None
        in_kifs = [params.get("input_kif")] if params.get("input_kif") is not None else None
        out_qints = [params.get("output_qint")] if params.get("output_qint") is not None else None
        out_kifs = [params.get("output_kif")] if params.get("output_kif") is not None else None
    elif prim == "elementwise":
        in_qints = params.get("input_qints")
        in_kifs = params.get("input_kifs")
        out_qints = [params.get("output_qint")] if params.get("output_qint") is not None else None
        out_kifs = [params.get("output_kif")] if params.get("output_kif") is not None else None
    elif prim == "activation":
        in_qints = [params.get("input_qint")] if params.get("input_qint") is not None else None
        in_kifs = [params.get("input_kif")] if params.get("input_kif") is not None else None
        out_qints = [params.get("output_qint")] if params.get("output_qint") is not None else None
        out_kifs = [params.get("output_kif")] if params.get("output_kif") is not None else None

    sp["input_qints"] = in_qints
    sp["input_kifs"] = in_kifs
    sp["output_qints"] = out_qints
    sp["output_kifs"] = out_kifs
    sp["precision_source"] = "hgq" if any(x is not None for x in (in_qints, in_kifs, out_qints, out_kifs)) else "unknown"


_EDGE_COPY_FIELDS = [
    "tensor_shape",
    "src_qint",
    "src_kif",
    "src_bitwidth_bits",
    "dst_qint",
    "dst_kif",
    "dst_bitwidth_bits",
    "element_bitwidth_bits",
    "element_kif",
    "element_qint",
    "tensor_width_bits",
    "volume_bits_exact",
    "has_quantization_boundary",
    "producer_quantizer",
    "consumer_quantizer",
    "needs_cast",
    "cast_mode",
    "bitwidth_src",
    "bitwidth_dst",
    "volume_bits",
]


def _copy_edge_precision(nn_ep: dict, sched_ep: dict):
    for key in _EDGE_COPY_FIELDS:
        if key in nn_ep:
            sched_ep[key] = nn_ep.get(key)
    sched_ep["edge_kind"] = "data"
    sched_ep["qint"] = _first_not_none(
        nn_ep.get("element_qint"),
        nn_ep.get("src_qint"),
        nn_ep.get("dst_qint"),
    )
    sched_ep["kif"] = _first_not_none(
        nn_ep.get("element_kif"),
        nn_ep.get("src_kif"),
        nn_ep.get("dst_kif"),
    )
    sched_ep["bitwidth"] = _first_not_none(
        nn_ep.get("element_bitwidth_bits"),
        nn_ep.get("dst_bitwidth_bits"),
        nn_ep.get("src_bitwidth_bits"),
        nn_ep.get("bitwidth_dst"),
        nn_ep.get("bitwidth_src"),
    )
    sched_ep["volume_bits"] = _first_not_none(
        nn_ep.get("volume_bits_exact"),
        nn_ep.get("volume_bits"),
    )


def _sync_op_params_inputs(p: dict):
    params = p.get("op_params") or {}
    op = p.get("op")
    if op == "elementwise":
        params["input_qints"] = p.get("input_qints")
        params["input_kifs"] = p.get("input_kifs")
    elif op in ("reduce", "activation", "dense"):
        if p.get("input_qints"):
            params["input_qint"] = p["input_qints"][0]
        if p.get("input_kifs"):
            params["input_kif"] = p["input_kifs"][0]


def _refresh_node_inputs_from_edges(g: HGraph):
    for vx in g.vertices:
        p = g.pmap[vx]
        preds = g.in_vx(vx)
        if not preds:
            continue

        in_qints = []
        in_kifs = []
        input_widths = []
        for pred in preds:
            ep = g.pmap[(pred, vx)]
            in_qints.append(_first_not_none(ep.get("src_qint"), ep.get("element_qint"), ep.get("dst_qint")))
            in_kifs.append(_first_not_none(ep.get("src_kif"), ep.get("element_kif"), ep.get("dst_kif")))
            input_widths.append(
                _first_not_none(
                    ep.get("tensor_width_bits"),
                    ep.get("volume_bits_exact"),
                    ep.get("volume_bits"),
                )
            )

        if any(x is not None for x in in_qints):
            p["input_qints"] = in_qints
        if any(x is not None for x in in_kifs):
            p["input_kifs"] = in_kifs
        if any(x is not None for x in input_widths):
            p["input_tensor_width_bits"] = input_widths

        _sync_op_params_inputs(p)


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
        _apply_node_precision_from_params(sp, params, prim)

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
        _copy_edge_precision(ep, sp)

    _refresh_node_inputs_from_edges(sg)

    return sg
