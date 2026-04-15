"""NN-IR builder: turn a Keras/HGQ model into a populated `HGraph`.

NN-IR is a *structural* translation — one vertex per Keras layer, edges
following the Keras connectivity, properties copied directly from the layer
and from HGQ metadata. No decomposition, no lowering, no scheduling.

Because the enclosing directory ``IR/NN-IR`` has a hyphen (not a valid
Python package name), this module is loaded via ``importlib`` rather than a
normal ``import`` statement. From a notebook:

    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "nn_ir_builder",
        Path("/path/to/CMIR/IR/NN-IR/builder.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = mod.build_nn_ir(model, name="jedi_gnn")
"""

from __future__ import annotations

from math import prod
from typing import Any

import importlib.util
from pathlib import Path

from heterograph import HGraph


# The enclosing directory ``IR/NN-IR`` has a hyphen so it cannot be a regular
# Python package. Load sibling modules (``schema.py``, ``_hgq.py``) directly
# from their file paths to avoid clashing with the other IR schemas (e.g.
# ``IR/schema.py``, ``IR/Transport-IR/schema.py``) on ``sys.path``.
def _load_sibling(name: str):
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(f"_nn_ir_{name}", here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_schema = _load_sibling("schema")
_hgq = _load_sibling("_hgq")

vinit_nn = _schema.vinit_nn
einit_nn = _schema.einit_nn
ginit_nn = _schema.ginit_nn


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

_OP_KIND = {
    "InputLayer":            "input",
    "QEinsumDenseBatchnorm": "einsum_dense_bn",
    "QEinsumDense":          "einsum_dense",
    "QDense":                "dense",
    "QSum":                  "qsum",
    "QAdd":                  "qadd",
    "Activation":            "activation",
}


def _classify(layer) -> str:
    cls = type(layer).__name__
    if cls not in _OP_KIND:
        raise NotImplementedError(
            f"NN-IR builder does not know how to lower Keras layer class "
            f"{cls!r} (layer name: {getattr(layer, 'name', '?')!r}). "
            f"Add it to _OP_KIND in IR/NN-IR/builder.py."
        )
    return _OP_KIND[cls]


# --------------------------------------------------------------------------- #
# Shape / connectivity helpers (Keras 3 functional API)
# --------------------------------------------------------------------------- #

def _as_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _layer_shapes(layer) -> tuple[list, list]:
    """Return (in_shapes, out_shapes) read from _inbound_nodes."""
    in_shapes: list = []
    out_shapes: list = []
    for node in getattr(layer, "_inbound_nodes", []) or []:
        for t in _as_list(node.input_tensors):
            in_shapes.append(tuple(t.shape))
        for t in _as_list(node.output_tensors):
            out_shapes.append(tuple(t.shape))
    return in_shapes, out_shapes


def _layer_inbound(layer) -> list[str]:
    """Return the names of layers feeding `layer`, in Keras input order."""
    inbound: list[str] = []
    for node in getattr(layer, "_inbound_nodes", []) or []:
        for t in _as_list(node.input_tensors):
            hist = getattr(t, "_keras_history", None)
            if hist is not None:
                inbound.append(hist[0].name)
    return inbound


def _non_batch_volume(shape) -> int | None:
    dims = [d for d in shape if d is not None]
    if len(dims) == 0:
        return None
    return int(prod(dims))


# --------------------------------------------------------------------------- #
# Per-layer record extraction
# --------------------------------------------------------------------------- #

def _extract_record(layer, idx: int) -> dict[str, Any]:
    in_shapes, out_shapes = _layer_shapes(layer)

    equation = getattr(layer, "equation", None)
    act = getattr(layer, "activation", None)
    activation = getattr(act, "__name__", None) if act is not None else None

    kernel = getattr(layer, "kernel", None)
    kernel_shape = tuple(kernel.shape) if kernel is not None else None

    try:
        num_params = int(layer.count_params())
    except Exception:
        num_params = None

    iq = getattr(layer, "iq", None)
    kq = getattr(layer, "kq", None)
    bq = getattr(layer, "bq", None)

    return {
        "layer_name":       layer.name,
        "layer_class":      type(layer).__name__,
        "layer_idx":        idx,
        "op_kind":          _classify(layer),
        "equation":         equation,
        "activation":       activation,
        "kernel_shape":     kernel_shape,
        "in_shapes":        in_shapes,
        "out_shapes":       out_shapes,
        "iq_bw":            _hgq.avg_bw(iq),
        "kq_bw":            _hgq.avg_bw(kq),
        "bq_bw":            _hgq.avg_bw(bq),
        "iq_bw_per_param":  _hgq.bw_array(iq),
        "kq_bw_per_param":  _hgq.bw_array(kq),
        "sparsity":         _hgq.sparsity(kernel),
        "num_params":       num_params,
    }


# --------------------------------------------------------------------------- #
# Edge bitwidth rule
# --------------------------------------------------------------------------- #

def _source_out_bw(src_record: dict) -> float | None:
    """Best locally-derivable output bitwidth for the source layer.

    HGQ does not expose an explicit output quantizer, so for CMVM layers we
    take `max(kq_bw_per_param)` as a conservative proxy (the widest coefficient
    determines the widest product bit). For pure-transport layers we fall back
    to the source's own input bitwidth, which is correct for identity/reshape
    ops and a reasonable proxy otherwise.
    """
    kq_arr = src_record.get("kq_bw_per_param")
    if kq_arr is not None and kq_arr.size > 0:
        return float(kq_arr.max())
    return src_record.get("iq_bw")


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #

_REQUIRED_VX_FIELDS = ("layer_name", "op_kind", "in_shapes", "out_shapes")


def _validate(g: HGraph) -> None:
    for vx in g.vertices:
        p = g.pmap[vx]
        for k in _REQUIRED_VX_FIELDS:
            if p.get(k) is None:
                raise ValueError(
                    f"NN-IR vertex {vx} ({p.get('layer_name')!r}) is missing "
                    f"required field {k!r}"
                )
    for e in g.edges:
        p = g.pmap[e]
        if p.get("tensor_shape") is None:
            raise ValueError(f"NN-IR edge {e} is missing tensor_shape")
        if p.get("bitwidth_src") is None and p.get("bitwidth_dst") is None:
            raise ValueError(
                f"NN-IR edge {e} has no bitwidth on either side"
            )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def build_nn_ir(model, name: str | None = None, validate: bool = True) -> HGraph:
    """Decompose a Keras/HGQ model into a populated NN-IR `HGraph`.

    Args:
        model:    a Keras functional-API model using HGQ-quantized layers.
        name:     optional graph name; defaults to ``model.name``.
        validate: run post-build sanity checks (default True).

    Returns:
        A populated `HGraph` keyed by the NN-IR schema.
    """
    g = HGraph(vinit=vinit_nn, einit=einit_nn, ginit=ginit_nn)

    g.pmap["name"] = name or model.name
    g.pmap["model_source"] = "keras_hgq"
    try:
        g.pmap["n_features"] = int(model.input_shape[-1])
    except Exception:
        g.pmap["n_features"] = None
    try:
        g.pmap["n_classes"] = int(model.output_shape[-1])
    except Exception:
        g.pmap["n_classes"] = None

    # Pass 1 — classify and extract per-layer records.
    records = [_extract_record(layer, i) for i, layer in enumerate(model.layers)]
    rec_by_name = {r["layer_name"]: r for r in records}

    # Pass 2 — allocate vertices.
    name2vx: dict[str, int] = {}
    for rec in records:
        vx = g.add_vx()
        name2vx[rec["layer_name"]] = vx
        g.pmap[vx].update(rec)

    # Pass 3 — wire edges from Keras connectivity.
    for layer in model.layers:
        dst_name = layer.name
        dst_rec = rec_by_name[dst_name]
        dst_vx = name2vx[dst_name]

        for src_name in _layer_inbound(layer):
            if src_name not in name2vx:
                continue  # e.g. a stray tensor with no matching layer record
            src_vx = name2vx[src_name]
            if src_vx == dst_vx:
                continue  # HGraph drops self-loops anyway

            created = g.add_edge(src_vx, dst_vx)
            if not created:
                continue  # edge already existed (parallel keras edges)

            e = created[0]
            src_rec = rec_by_name[src_name]

            # Tensor shape comes from the source layer's output.
            shape = tuple(src_rec["out_shapes"][0]) if src_rec["out_shapes"] else None

            bw_dst = dst_rec.get("iq_bw")
            bw_src = _source_out_bw(src_rec)

            vol = None
            if shape is not None and bw_dst is not None:
                nbv = _non_batch_volume(shape)
                if nbv is not None:
                    vol = float(nbv) * float(bw_dst)

            g.pmap[e]["tensor_shape"] = shape
            g.pmap[e]["bitwidth_src"] = bw_src
            g.pmap[e]["bitwidth_dst"] = bw_dst
            g.pmap[e]["volume_bits"] = vol

    if validate:
        _validate(g)

    return g


