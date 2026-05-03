"""NN-IR heterograph styling."""

from __future__ import annotations


OP_COLORS = {
    "input": "#B0BEC5",
    "einsum_dense_bn": "#1E88E5",
    "einsum_dense": "#42A5F5",
    "dense": "#64B5F6",
    "qsum": "#FB8C00",
    "qadd": "#43A047",
    "activation": "#E0E0E0",
}

OP_SHAPES = {
    "input": "ellipse",
    "einsum_dense_bn": "box",
    "einsum_dense": "box",
    "dense": "box",
    "qsum": "invtrapezium",
    "qadd": "diamond",
    "activation": "oval",
}

_DARK_FG_OPS = ("einsum_dense_bn", "einsum_dense", "dense", "qsum", "qadd")


def _fmt_shape(shape) -> str:
    if shape is None:
        return "?"
    return "x".join("?" if dim is None else str(dim) for dim in shape)


def vx_label(g, vx) -> str:
    p = g.pmap[vx]
    lines = [
        p["layer_name"],
        f"[{p['op_kind']}]",
        f"in: {','.join(_fmt_shape(s) for s in (p['in_shapes'] or [])) or '—'}",
        f"out: {','.join(_fmt_shape(s) for s in (p['out_shapes'] or [])) or '—'}",
    ]
    if p.get("equation"):
        lines.append(f"eq: {p['equation']}")
    if p.get("kernel_shape"):
        lines.append(f"W: {_fmt_shape(p['kernel_shape'])}")
    bits = []
    for key in ("iq_bw", "kq_bw", "bq_bw"):
        value = p.get(key)
        if value is not None:
            bits.append(f"{key[:-3]}={value:.1f}")
    if bits:
        lines.append(" ".join(bits))
    if p.get("kernel_sparsity") is not None:
        lines.append(f"s={p['kernel_sparsity']:.2f}")
    return "\n".join(lines)


def edge_label(g, e) -> str:
    p = g.pmap[e]
    shape = _fmt_shape(p.get("tensor_shape"))
    bw_src = p.get("bitwidth_src")
    bw_dst = p.get("bitwidth_dst")
    label = shape
    if bw_src is not None and bw_dst is not None:
        label += f" @ {bw_src:.1f}->{bw_dst:.1f}b"
    elif bw_dst is not None:
        label += f" @ {bw_dst:.1f}b"
    volume = p.get("volume_bits")
    if volume is not None:
        label += f"\n{int(volume)} bits"
    return label


def edge_penwidth(g, e) -> str:
    import math

    volume = g.pmap[e].get("volume_bits") or 0
    return str(0.6 + 0.6 * math.log1p(volume / 64))


def apply_nn_style(g) -> None:
    g.vstyle["fillcolor"] = lambda g, vx: OP_COLORS.get(g.pmap[vx]["op_kind"], "#FFFFFF")
    g.vstyle["shape"] = lambda g, vx: OP_SHAPES.get(g.pmap[vx]["op_kind"], "box")
    g.vstyle["style"] = lambda g, vx: "filled"
    g.vstyle["fontcolor"] = lambda g, vx: "white" if g.pmap[vx]["op_kind"] in _DARK_FG_OPS else "black"
    g.vstyle["label"] = vx_label
    g.estyle["label"] = edge_label
    g.estyle["penwidth"] = edge_penwidth
    g.estyle["fontsize"] = lambda g, e: "10"
