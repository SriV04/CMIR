"""Sched-IR Phase 1b — propagate bound precision metadata.

This pass runs after BIND. BIND stores DA4ML-derived output precision on
compute vertices; this module copies that producer-side precision onto data
edges and refreshes consumer input metadata without changing topology.
"""

from __future__ import annotations

from heterograph import HGraph


_COMPUTE_OPS = ("dense", "reduce", "elementwise", "activation")
_INFRA_OPS = ("buffer", "mux")


def _kif_bits(kif: dict | None) -> int | None:
    if not kif or not isinstance(kif, dict):
        return None
    bits = kif.get("bits")
    return int(bits) if bits is not None else None


def _as_list(value):
    if value is None:
        return None
    return value if isinstance(value, list) else [value]


def _total_bits_from_kifs(kifs) -> int | None:
    kifs = _as_list(kifs)
    if not kifs:
        return None
    bits = [_kif_bits(k) for k in kifs if k is not None]
    if not bits or len(bits) != len(kifs):
        return None
    return sum(bits)


def _element_bits_from_kifs(kifs) -> int | None:
    kifs = _as_list(kifs)
    if not kifs:
        return None
    bits = [_kif_bits(k) for k in kifs if k is not None]
    if not bits:
        return None
    return bits[0] if len(set(bits)) == 1 else max(bits)


def _producer_output_precision(p: dict) -> dict:
    params = p.get("op_params") or {}

    qints = p.get("output_qints")
    kifs = p.get("output_kifs")
    width = p.get("output_tensor_width_bits")
    source = p.get("precision_source")

    if qints is None:
        oq = params.get("output_qint")
        qints = oq if isinstance(oq, list) else ([oq] if oq is not None else None)

    if kifs is None:
        ok = params.get("output_kif")
        kifs = ok if isinstance(ok, list) else ([ok] if ok is not None else None)

    if width is None:
        width = _total_bits_from_kifs(kifs)

    return {
        "qints": qints,
        "kifs": kifs,
        "tensor_width_bits": width,
        "element_bitwidth_bits": _element_bits_from_kifs(kifs),
        "source": source or "unknown",
    }


def _producer_output_precision_with_warnings(p: dict, vx: int, warnings: list[str], strict: bool) -> dict:
    prod = _producer_output_precision(p)
    params = p.get("op_params") or {}

    if prod["qints"] is None and prod["kifs"] is None:
        legacy_bw = params.get("out_bw")
        if legacy_bw is not None and not strict:
            msg = f"vertex {vx} ({p.get('nn_layer_name')}) fell back to legacy op_params.out_bw"
            warnings.append(msg)
            bits = int(legacy_bw)
            prod["kifs"] = [
                {"k": None, "i": None, "f": None, "bits": bits, "source": "legacy_out_bw"}
            ]
            prod["tensor_width_bits"] = bits
            prod["element_bitwidth_bits"] = bits
            prod["source"] = "legacy_out_bw"

    return prod


def _write_edge_producer_precision(ep: dict, prod: dict) -> None:
    ep["src_qint"] = prod["qints"]
    ep["src_kif"] = prod["kifs"]
    ep["src_bitwidth_bits"] = prod["element_bitwidth_bits"]

    ep["element_qint"] = prod["qints"]
    ep["element_kif"] = prod["kifs"]
    ep["element_bitwidth_bits"] = prod["element_bitwidth_bits"]

    ep["tensor_width_bits"] = prod["tensor_width_bits"]
    ep["volume_bits_exact"] = prod["tensor_width_bits"]

    # Canonical Sched-IR aliases.
    ep["qint"] = prod["qints"]
    ep["kif"] = prod["kifs"]

    # Legacy aliases used by old infra/schedule code.
    ep["bitwidth"] = prod["element_bitwidth_bits"]
    ep["volume_bits"] = prod["tensor_width_bits"]


def _kifs_equal(a, b) -> bool | None:
    if a is None or b is None:
        return None
    a = _as_list(a)
    b = _as_list(b)
    if len(a) != len(b):
        if len(a) == 1 and all(item == a[0] for item in b):
            return True
        if len(b) == 1 and all(item == b[0] for item in a):
            return True
        return False
    return a == b


def _mark_cast_if_needed(ep: dict) -> None:
    eq = _kifs_equal(ep.get("src_kif"), ep.get("dst_kif"))

    if eq is False:
        ep["needs_cast"] = True
        ep["has_quantization_boundary"] = True
        ep["cast_mode"] = "producer_to_consumer_quantizer"
    elif eq is True:
        ep["needs_cast"] = False


def _sync_inputs_to_op_params(p: dict) -> None:
    if p.get("op_params") is None:
        p["op_params"] = {}
    params = p["op_params"]
    op = p.get("op")

    if op == "elementwise":
        params["input_qints"] = p.get("input_qints")
        params["input_kifs"] = p.get("input_kifs")

    elif op in ("dense", "reduce", "activation"):
        if p.get("input_qints"):
            params["input_qint"] = p["input_qints"][0]
        if p.get("input_kifs"):
            params["input_kif"] = p["input_kifs"][0]


def _refresh_consumer_inputs(g: HGraph) -> None:
    for vx in g.vertices:
        p = g.pmap[vx]
        preds = g.in_vx(vx) if hasattr(g, "in_vx") else _predecessors(g, vx)
        if not preds:
            continue

        qints = []
        kifs = []
        widths = []

        for pred in preds:
            ep = g.pmap[(pred, vx)]

            use_dst = ep.get("needs_cast") and ep.get("dst_kif") is not None

            qints.append(ep.get("dst_qint") if use_dst else ep.get("src_qint"))
            kifs.append(ep.get("dst_kif") if use_dst else ep.get("src_kif"))
            widths.append(ep.get("tensor_width_bits") or ep.get("volume_bits_exact"))

        if any(x is not None for x in qints):
            p["input_qints"] = qints
        if any(x is not None for x in kifs):
            p["input_kifs"] = kifs
        if any(x is not None for x in widths):
            p["input_tensor_width_bits"] = widths

        _sync_inputs_to_op_params(p)


def _successors(g: HGraph, vx: int) -> list[int]:
    if hasattr(g, "out_vx"):
        return list(g.out_vx(vx))
    return [dst for src, dst in g.edges if src == vx]


def _predecessors(g: HGraph, vx: int) -> list[int]:
    return [src for src, dst in g.edges if dst == vx]


def _tensor_volume(shape) -> int | None:
    if not shape:
        return None
    dims = [dim for dim in shape if dim is not None]
    if not dims:
        return None
    volume = 1
    for dim in dims:
        volume *= int(dim)
    return volume


def _has_producer_precision(p: dict) -> bool:
    prod = _producer_output_precision(p)
    return prod["qints"] is not None or prod["kifs"] is not None


def propagate_precision(g_sched: HGraph, *, strict: bool = False) -> HGraph:
    """Propagate bound node output precision onto outgoing edges.

    Assumes BIND has already populated output_qints/output_kifs.
    Mutates ``g_sched`` in place and returns it.
    """
    warnings: list[str] = []

    for u in list(g_sched.vertices):
        p = g_sched.pmap[u]

        if p.get("op") in _INFRA_OPS:
            continue

        prod = _producer_output_precision_with_warnings(p, u, warnings, strict)

        if prod["qints"] is None and prod["kifs"] is None:
            msg = f"vertex {u} ({p.get('nn_layer_name')}) has no output precision"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

        for v in _successors(g_sched, u):
            ep = g_sched.pmap[(u, v)]
            _write_edge_producer_precision(ep, prod)
            _mark_cast_if_needed(ep)

    _refresh_consumer_inputs(g_sched)

    more = validate_precision(g_sched, strict=strict)
    warnings.extend(more)

    g_sched.pmap["precision_propagated"] = True
    g_sched.pmap["precision_warnings"] = warnings or None
    return g_sched


def validate_precision(g_sched: HGraph, *, strict: bool = False) -> list[str]:
    """Return precision metadata warnings, raising only when ``strict`` is set."""
    warnings: list[str] = []

    for vx in g_sched.vertices:
        p = g_sched.pmap[vx]
        op = p.get("op")
        if op not in _COMPUTE_OPS:
            continue

        if p.get("cost") is None:
            warnings.append(f"vertex {vx} ({p.get('nn_layer_name')}) has no bound cost")

        if not _has_producer_precision(p):
            warnings.append(f"vertex {vx} ({p.get('nn_layer_name')}) has no output precision")

        if op == "elementwise":
            preds = g_sched.in_vx(vx) if hasattr(g_sched, "in_vx") else _predecessors(g_sched, vx)
            input_kifs = p.get("input_kifs")
            if input_kifs is not None and len(input_kifs) != len(preds):
                warnings.append(
                    f"elementwise vertex {vx} ({p.get('nn_layer_name')}) has "
                    f"{len(input_kifs)} input_kifs for {len(preds)} incoming edges"
                )

    for u, v in g_sched.edges:
        ep = g_sched.pmap[(u, v)]
        if ep.get("edge_kind") not in (None, "data"):
            continue

        if ep.get("tensor_shape") is None:
            warnings.append(f"edge ({u}, {v}) has no tensor_shape")

        producer = g_sched.pmap[u]
        producer_has_precision = producer.get("op") in _INFRA_OPS or _has_producer_precision(producer)
        if producer_has_precision and ep.get("src_qint") is None and ep.get("src_kif") is None:
            warnings.append(f"edge ({u}, {v}) has no producer-side precision")

        if ep.get("src_kif") is not None and ep.get("tensor_width_bits") is None:
            warnings.append(f"edge ({u}, {v}) has src_kif but no tensor_width_bits")

        src_kifs = _as_list(ep.get("src_kif"))
        volume = _tensor_volume(ep.get("tensor_shape"))
        if src_kifs is not None and volume is not None and len(src_kifs) not in (1, volume):
            warnings.append(
                f"edge ({u}, {v}) has {len(src_kifs)} src_kif entries for tensor volume {volume}"
            )

        if ep.get("element_bitwidth_bits") is not None and ep.get("bitwidth") is None:
            warnings.append(f"edge ({u}, {v}) has element_bitwidth_bits but no bitwidth alias")

        if ep.get("needs_cast") is True and ep.get("dst_kif") is None and ep.get("dst_qint") is None:
            warnings.append(f"edge ({u}, {v}) needs cast but has no consumer precision")

    if strict and warnings:
        raise ValueError("precision validation failed:\n" + "\n".join(warnings))

    return warnings
