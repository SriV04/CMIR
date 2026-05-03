"""NN-IR builder: turn a Keras/HGQ model into a populated `HGraph`."""

from __future__ import annotations

from math import prod
from typing import Any
import warnings

import importlib.util
from pathlib import Path

from heterograph import HGraph


def _load_sibling(name: str, relpath: str | None = None):
    here = Path(__file__).resolve().parent
    path = here / relpath if relpath is not None else here / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_nn_ir_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_schema = _load_sibling("schema")
_hgq = _load_sibling("hgq_extractor", "hgq2/hgq_extractor.py")

vinit_nn = _schema.vinit_nn
einit_nn = _schema.einit_nn
ginit_nn = _schema.ginit_nn


_OP_KIND = {
    "InputLayer": "input",
    "QEinsumDenseBatchnorm": "einsum_dense_bn",
    "QEinsumDense": "einsum_dense",
    "QDense": "dense",
    "QSum": "qsum",
    "QAdd": "qadd",
    "Activation": "activation",
}


def _classify(layer) -> str:
    cls = type(layer).__name__
    if cls not in _OP_KIND:
        raise NotImplementedError(
            f"NN-IR builder does not know how to lower Keras layer class "
            f"{cls!r} (layer name: {getattr(layer, 'name', '?')!r})."
        )
    return _OP_KIND[cls]


def _as_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _layer_shapes(layer) -> tuple[list, list]:
    in_shapes: list = []
    out_shapes: list = []
    for node in getattr(layer, "_inbound_nodes", []) or []:
        for t in _as_list(node.input_tensors):
            in_shapes.append(tuple(t.shape))
        for t in _as_list(node.output_tensors):
            out_shapes.append(tuple(t.shape))
    return in_shapes, out_shapes


def _layer_inbound(layer) -> list[str]:
    inbound: list[str] = []
    for node in getattr(layer, "_inbound_nodes", []) or []:
        for t in _as_list(node.input_tensors):
            hist = getattr(t, "_keras_history", None)
            if hist is not None:
                inbound.append(hist[0].name)
    return inbound


def _non_batch_volume(shape) -> int | None:
    dims = [d for d in shape[1:] if d is not None] if shape else []
    if len(dims) == 0:
        return None
    return int(prod(dims))


def _quant_summary_fields(summary: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    stats = {
        "avg": None,
        "max": None,
        "min": None,
        "shape": None,
    }
    if summary and summary.get("kif"):
        bits = summary["kif"].get("bits")
        if bits is not None:
            arr = _hgq.safe_array(bits)
            if arr is not None and arr.size > 0:
                stats = {
                    "avg": float(arr.mean()),
                    "max": float(arr.max()),
                    "min": float(arr.min()),
                    "shape": tuple(arr.shape),
                }

    return {
        prefix: summary,
        f"{prefix}_kif": summary.get("kif") if summary else None,
        f"{prefix}_qint": summary.get("qint") if summary else None,
        f"{prefix}_bw_avg": stats["avg"],
        f"{prefix}_bw_max": stats["max"],
        f"{prefix}_bw_min": stats["min"],
        f"{prefix}_bw_shape": stats["shape"],
        f"{prefix}_overflow_mode": summary.get("overflow_mode") if summary else None,
        f"{prefix}_round_mode": summary.get("round_mode") if summary else None,
    }


def _extract_record(
    layer,
    idx: int,
    *,
    include_values: bool,
    include_histograms: bool,
    value_storage: str,
) -> dict[str, Any]:
    in_shapes, out_shapes = _layer_shapes(layer)
    equation = getattr(layer, "equation", None)
    act = getattr(layer, "activation", None)
    activation = getattr(act, "__name__", None) if act is not None else None

    try:
        num_params = int(layer.count_params())
    except Exception:
        num_params = None

    qmeta = _hgq.extract_all_quantizers(layer)
    values = _hgq.extract_layer_values(layer) if include_values else {
        "kernel_values": None,
        "kernel_float_values": None,
        "bias_values": None,
        "batchnorm_values": None,
        "qkernel_values": None,
        "qbias_values": None,
        "uses_qkernel": False,
    }
    stats = _hgq.weight_stats(values["kernel_values"], include_histogram=include_histograms)
    kernel_values = values["kernel_values"]
    kernel_shape = tuple(kernel_values.shape) if kernel_values is not None else None

    record = {
        "layer_name": layer.name,
        "layer_class": type(layer).__name__,
        "layer_idx": idx,
        "op_kind": _classify(layer),
        "equation": equation,
        "activation": activation,
        "kernel_shape": kernel_shape,
        "has_bn": type(layer).__name__ == "QEinsumDenseBatchnorm",
        "bn_folded_into_qkernel": bool(values["qkernel_values"] is not None and type(layer).__name__ == "QEinsumDenseBatchnorm"),
        "in_shapes": in_shapes,
        "out_shapes": out_shapes,
        "kernel_values": values["kernel_values"] if value_storage == "inline" else None,
        "kernel_float_values": values["kernel_float_values"] if value_storage == "inline" else None,
        "bias_values": values["bias_values"] if value_storage == "inline" else None,
        "batchnorm_values": values["batchnorm_values"] if value_storage == "inline" else None,
        "qkernel_values": values["qkernel_values"] if value_storage == "inline" else None,
        "qbias_values": values["qbias_values"] if value_storage == "inline" else None,
        "uses_qkernel": values["uses_qkernel"],
        "num_params": num_params,
        "quantizer_granularity": qmeta["iq"].get("granularity") if qmeta["iq"] else None,
        "quantizer_place": "layer",
        "quantizer_source": "HGQ",
        "kernel_sparsity": stats["sparsity"],
        "kernel_nonzero_count": stats["nonzero_count"],
        "kernel_zero_count": stats["zero_count"],
        "kernel_unique_values": stats["unique_values"] if include_values and value_storage == "inline" else None,
        "kernel_unique_count": stats["unique_count"],
        "kernel_value_histogram": stats["value_histogram"],
        "kernel_min": stats["min"],
        "kernel_max": stats["max"],
        "kernel_dtype": stats["dtype"],
    }
    record.update(_quant_summary_fields(qmeta["iq"], "iq"))
    record.update(_quant_summary_fields(qmeta["kq"], "kq"))
    record.update(_quant_summary_fields(qmeta["bq"], "bq"))
    record.update(_quant_summary_fields(qmeta["oq"], "oq"))
    record["aq"] = qmeta["aq"]

    # Legacy aliases.
    record["iq_bw"] = record["iq_bw_avg"]
    record["kq_bw"] = record["kq_bw_avg"]
    record["bq_bw"] = record["bq_bw_avg"]
    record["iq_bw_per_param"] = (
        qmeta["iq"]["kif"].get("bits") if qmeta["iq"] and qmeta["iq"].get("kif") else None
    )
    record["kq_bw_per_param"] = (
        qmeta["kq"]["kif"].get("bits") if qmeta["kq"] and qmeta["kq"].get("kif") else None
    )
    record["sparsity"] = record["kernel_sparsity"]
    record["weights"] = record["kernel_values"]
    record["biases"] = record["bias_values"]
    return record


def _edge_kif(record: dict[str, Any], side: str) -> dict[str, Any] | None:
    if side == "src":
        return record.get("oq_kif") or (
            record.get("aq", {}).get("kif") if isinstance(record.get("aq"), dict) else None
        )
    return record.get("iq_kif")


def _edge_qint(record: dict[str, Any], side: str) -> dict[str, Any] | None:
    if side == "src":
        return record.get("oq_qint") or (
            record.get("aq", {}).get("qint") if isinstance(record.get("aq"), dict) else None
        )
    return record.get("iq_qint")


_REQUIRED_VX_FIELDS = ("layer_name", "op_kind", "in_shapes", "out_shapes")
_COMPUTE_OPS = {"dense", "einsum_dense", "einsum_dense_bn"}


def _validate(g: HGraph) -> None:
    for vx in g.vertices:
        p = g.pmap[vx]
        for key in _REQUIRED_VX_FIELDS:
            if p.get(key) is None:
                raise ValueError(
                    f"NN-IR vertex {vx} ({p.get('layer_name')!r}) is missing required field {key!r}"
                )
        if p.get("op_kind") in _COMPUTE_OPS and p.get("kernel_values") is None and p.get("qkernel_values") is None:
            raise ValueError(
                f"NN-IR vertex {vx} ({p.get('layer_name')!r}) is missing kernel values"
            )
        if p.get("op_kind") in _COMPUTE_OPS and p.get("kq") is None:
            warnings.warn(
                f"NN-IR vertex {p.get('layer_name')!r} has no kernel quantizer summary",
                stacklevel=2,
            )
        if p.get("op_kind") != "input" and p.get("iq") is None:
            warnings.warn(
                f"NN-IR vertex {p.get('layer_name')!r} has no input quantizer summary",
                stacklevel=2,
            )

    for edge in g.edges:
        p = g.pmap[edge]
        if p.get("tensor_shape") is None:
            raise ValueError(f"NN-IR edge {edge} is missing tensor_shape")
        if not any(
            p.get(key) is not None
            for key in ("dst_kif", "dst_qint", "src_kif", "src_qint", "bitwidth_dst", "bitwidth_src")
        ):
            warnings.warn(f"NN-IR edge {edge} has no precision metadata", stacklevel=2)


def build_nn_ir(
    model,
    name: str | None = None,
    validate: bool = True,
    *,
    include_values: bool = True,
    include_histograms: bool = True,
    value_storage: str = "inline",
) -> HGraph:
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

    records = [
        _extract_record(
            layer,
            i,
            include_values=include_values,
            include_histograms=include_histograms,
            value_storage=value_storage,
        )
        for i, layer in enumerate(model.layers)
    ]
    rec_by_name = {record["layer_name"]: record for record in records}

    name2vx: dict[str, int] = {}
    for record in records:
        vx = g.add_vx()
        name2vx[record["layer_name"]] = vx
        g.pmap[vx].update(record)

    for layer in model.layers:
        dst_name = layer.name
        dst_rec = rec_by_name[dst_name]
        dst_vx = name2vx[dst_name]

        for src_name in _layer_inbound(layer):
            if src_name not in name2vx:
                continue
            src_vx = name2vx[src_name]
            if src_vx == dst_vx:
                continue

            created = g.add_edge(src_vx, dst_vx)
            if not created:
                continue

            src_rec = rec_by_name[src_name]
            edge = created[0]
            shape = tuple(src_rec["out_shapes"][0]) if src_rec["out_shapes"] else None
            src_kif = _edge_kif(src_rec, "src")
            dst_kif = _edge_kif(dst_rec, "dst")
            src_qint = _edge_qint(src_rec, "src")
            dst_qint = _edge_qint(dst_rec, "dst")
            bw_src = _hgq.bitwidth_from_kif(src_kif)
            bw_dst = _hgq.bitwidth_from_kif(dst_kif)
            element_bw = bw_src if bw_src is not None else bw_dst
            tensor_width = None
            if shape is not None and element_bw is not None:
                nbv = _non_batch_volume(shape)
                if nbv is not None:
                    tensor_width = float(nbv * element_bw)

            boundary = False
            if src_kif is not None and dst_kif is not None:
                boundary = repr(src_kif) != repr(dst_kif)

            g.pmap[edge].update(
                {
                    "tensor_shape": shape,
                    "src_qint": src_qint,
                    "src_kif": src_kif,
                    "src_bitwidth_bits": bw_src,
                    "dst_qint": dst_qint,
                    "dst_kif": dst_kif,
                    "dst_bitwidth_bits": bw_dst,
                    "element_bitwidth_bits": element_bw,
                    "element_kif": src_kif if src_kif is not None else dst_kif,
                    "element_qint": src_qint if src_qint is not None else dst_qint,
                    "tensor_width_bits": tensor_width,
                    "volume_bits_exact": tensor_width,
                    "has_quantization_boundary": boundary,
                    "producer_quantizer": src_rec.get("oq") or src_rec.get("aq"),
                    "consumer_quantizer": dst_rec.get("iq"),
                    "needs_cast": boundary,
                    "cast_mode": "requantize" if boundary else None,
                    # Legacy aliases.
                    "bitwidth_src": bw_src,
                    "bitwidth_dst": bw_dst,
                    "volume_bits": tensor_width,
                }
            )

    if validate:
        _validate(g)
    return g
