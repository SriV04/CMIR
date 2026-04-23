"""NN-IR heterograph styling — colors, shapes, labels.

Attaches a Graphviz-style renderer configuration to an NN-IR graph so the
WebView / Graphviz exporter picks up per-vertex colors by ``op_kind`` and
rich labels built from ``in_shapes`` / ``out_shapes`` / quantizer bitwidths.

Usage::

    from styling import apply_nn_style
    apply_nn_style(g_nn)
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #

OP_COLORS = {
    "input":           "#B0BEC5",
    "einsum_dense_bn": "#1E88E5",
    "einsum_dense":    "#42A5F5",
    "dense":           "#64B5F6",
    "qsum":            "#FB8C00",
    "qadd":            "#43A047",
    "activation":      "#E0E0E0",
}

OP_SHAPES = {
    "input":           "ellipse",
    "einsum_dense_bn": "box",
    "einsum_dense":    "box",
    "dense":           "box",
    "qsum":            "invtrapezium",   # fan-in
    "qadd":            "diamond",
    "activation":      "oval",
}

_DARK_FG_OPS = ("einsum_dense_bn", "einsum_dense", "dense", "qsum", "qadd")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fmt_shape(s) -> str:
    if s is None:
        return "?"
    return "x".join("?" if d is None else str(d) for d in s)


# --------------------------------------------------------------------------- #
# Vertex / edge label callbacks
# --------------------------------------------------------------------------- #

def vx_label(g, vx) -> str:
    p = g.pmap[vx]
    name = p["layer_name"]
    kind = p["op_kind"]
    in_s  = ",".join(_fmt_shape(s) for s in (p["in_shapes"] or []))  or "—"
    out_s = ",".join(_fmt_shape(s) for s in (p["out_shapes"] or [])) or "—"
    lines = [name, f"[{kind}]", f"in: {in_s}", f"out: {out_s}"]
    if p.get("equation"):
        lines.append(f"eq: {p['equation']}")
    if p.get("kernel_shape"):
        lines.append(f"W: {_fmt_shape(p['kernel_shape'])}")
    iq, kq, bq = p.get("iq_bw"), p.get("kq_bw"), p.get("bq_bw")
    bits = []
    if iq is not None: bits.append(f"iq={iq:.1f}")
    if kq is not None: bits.append(f"kq={kq:.1f}")
    if bq is not None: bits.append(f"bq={bq:.1f}")
    if bits:
        lines.append(" ".join(bits))
    return "\n".join(lines)


def edge_label(g, e) -> str:
    p = g.pmap[e]
    shape = _fmt_shape(p.get("tensor_shape"))
    bw_dst = p.get("bitwidth_dst")
    bw_src = p.get("bitwidth_src")
    tag = shape
    if bw_src is not None and bw_dst is not None:
        tag += f" @ {bw_src:.1f}->{bw_dst:.1f}b"
    elif bw_dst is not None:
        tag += f" @ {bw_dst:.1f}b"
    vol = p.get("volume_bits")
    if vol is not None:
        tag += f"\n{int(vol)} bits"
    return tag


def edge_penwidth(g, e) -> str:
    import math
    vol = g.pmap[e].get("volume_bits") or 0
    return str(0.6 + 0.6 * math.log1p(vol / 64))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def apply_nn_style(g) -> None:
    """Attach NN-IR styling callbacks to a heterograph in place."""
    g.vstyle["fillcolor"] = lambda g, vx: OP_COLORS.get(g.pmap[vx]["op_kind"], "#FFFFFF")
    g.vstyle["shape"]     = lambda g, vx: OP_SHAPES.get(g.pmap[vx]["op_kind"], "box")
    g.vstyle["style"]     = lambda g, vx: "filled"
    g.vstyle["fontcolor"] = lambda g, vx: (
        "white" if g.pmap[vx]["op_kind"] in _DARK_FG_OPS else "black"
    )
    g.vstyle["label"]     = vx_label

    g.estyle["label"]     = edge_label
    g.estyle["penwidth"]  = edge_penwidth
    g.estyle["fontsize"]  = lambda g, e: "10"
