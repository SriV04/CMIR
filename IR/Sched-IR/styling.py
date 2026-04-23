"""Sched-IR heterograph styling — N–P–T aware labels.

Labels are built to surface the N–P–T timing fields written by FOLD / SCHEDULE
so the WebView makes the model visually inspectable:

* Compute vertices show  ``N=… P=… T=…``  when inside a fold group.
* Every bound vertex shows  ``L=… II=… L_tot=…``  (pipeline depth, initiation
  interval, total cycles = L + T - 1).
* Scheduled vertices show  ``t=[t_start..t_end]``.
* Critical-path vertices get a red border.
* Edges with a non-zero lifetime are drawn red and annotated with the buffer
  depth they will eventually be replaced with.

Usage::

    from styling import apply_sched_style
    apply_sched_style(g_sched)
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #

SCHED_COLORS = {
    "dense":       "#1E88E5",
    "reduce":      "#FB8C00",
    "elementwise": "#43A047",
    "activation":  "#E0E0E0",
    "buffer":      "#8E24AA",
    "mux":         "#F4511E",
}

SCHED_SHAPES = {
    "dense":       "box",
    "reduce":      "invtrapezium",
    "elementwise": "diamond",
    "activation":  "oval",
    "buffer":      "cylinder",
    "mux":         "trapezium",
}

_DARK_FG_OPS = ("dense", "reduce", "elementwise", "buffer", "mux")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fmt_shape(s) -> str:
    if s is None:
        return "?"
    return "x".join("?" if d is None else str(d) for d in s)


def _fmt_op_params(op: str, pp: dict, lines: list[str]) -> None:
    """Append op-specific parameter lines (in place) to ``lines``."""
    if op == "dense":
        ks = pp.get("kernel_shape")
        if ks is not None:
            lines.append(f"W: {_fmt_shape(ks)}")
        if pp.get("equation"):
            lines.append(f"eq: {pp['equation']}")
        bits = []
        if pp.get("in_bw") is not None: bits.append(f"in={pp['in_bw']:.1f}")
        if pp.get("kq_bw") is not None: bits.append(f"kq={pp['kq_bw']:.1f}")
        if bits:
            lines.append(" ".join(bits))
        if pp.get("activation"):
            lines.append(f"act: {pp['activation']}")
        if pp.get("has_bn"):
            lines.append("+bn")
    elif op == "reduce":
        lines.append(f"{pp.get('mode','?')} axes={pp.get('axes')}")
        if pp.get("reduction_width"):
            lines.append(f"width: {pp['reduction_width']}")
        if pp.get("in_bw") is not None and pp.get("out_bw") is not None:
            lines.append(f"bw: {pp['in_bw']:.1f}->{pp['out_bw']:.1f}")
    elif op == "elementwise":
        lines.append(f"op: {pp.get('op','?')}")
        for i, s in enumerate(pp.get("in_shapes") or []):
            lines.append(f"in{i}: {_fmt_shape(s)}")
        bc = pp.get("broadcast")
        if bc:
            lines.append(f"bcast: {bc}")
        if pp.get("out_bw") is not None:
            lines.append(f"out_bw: {pp['out_bw']:.1f}")
    elif op == "activation":
        lines.append(f"func: {pp.get('func','?')}")
    elif op == "buffer":
        lines.append(f"w={pp.get('width_bits')} d={pp.get('depth')}")
    elif op == "mux":
        lines.append(f"n={pp.get('n_inputs')} w={pp.get('width_bits')}")


# --------------------------------------------------------------------------- #
# Vertex label — N–P–T timing is the headline for compute vertices
# --------------------------------------------------------------------------- #

def sched_vx_label(g, vx) -> str:
    p = g.pmap[vx]
    name = p.get("nn_layer_name") or "?"
    op   = p.get("op") or "?"
    lines = [name, f"[{op}]"]

    fa = p.get("fold_axes")
    if fa:
        lines.append(f"fold axes: {','.join(str(a) for a in fa)}")

    pp = p.get("op_params") or {}
    _fmt_op_params(op, pp, lines)

    # ---- BIND: kernel binding ---- #
    kt = p.get("kernel_type")
    if kt is not None:
        ki = p.get("kernel_instance")
        lines.append(f"kernel: {kt}#{ki}")

    # ---- Cost (without latency — that lives in the N–P–T block below) ---- #
    cost = p.get("cost") or {}
    if cost:
        bits = []
        if cost.get("lut"):  bits.append(f"lut={cost['lut']}")
        if cost.get("ff"):   bits.append(f"ff={cost['ff']}")
        if cost.get("dsp"):  bits.append(f"dsp={cost['dsp']}")
        if cost.get("bram"): bits.append(f"bram={cost['bram']}")
        if bits:
            lines.append(" ".join(bits))

    # ---- FOLD: N–P–T timing ---- #
    N = p.get("parallelism_N")
    P = p.get("lanes_P")
    T = p.get("temporal_steps_T")
    L = p.get("pipeline_latency_L")
    II = p.get("ii")
    L_tot = p.get("latency_total")

    if N is not None and P is not None and T is not None:
        fg = p.get("fold_group")
        tag = f"N={N} P={P} T={T}"
        if fg is not None:
            tag += f"  g{fg}"
        lines.append(tag)
    if L is not None and II is not None and L_tot is not None:
        lines.append(f"L={L} II={II} L_tot={L_tot}")

    rm = p.get("reduce_mode")
    if rm and rm != "spatial":
        lines.append(f"reduce: {rm}")

    # ---- SCHEDULE: cycle range ---- #
    ts = p.get("t_start")
    te = p.get("t_end")
    if ts is not None and te is not None:
        lines.append(f"t=[{ts}..{te}]")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Edge label / width
# --------------------------------------------------------------------------- #

def sched_edge_label(g, e) -> str:
    p = g.pmap[e]
    shape = _fmt_shape(p.get("tensor_shape"))
    bw = p.get("bitwidth")
    tag = shape
    if bw is not None:
        tag += f" @ {bw:.1f}b"
    vol = p.get("volume_bits")
    if vol is not None:
        tag += f"\n{int(vol)} bits"
    lt = p.get("lifetime")
    if lt is not None and lt > 0:
        tag += f"\nbuf={lt} cyc"
    return tag


def sched_edge_penwidth(g, e) -> str:
    import math
    vol = g.pmap[e].get("volume_bits") or 0
    return str(0.6 + 0.6 * math.log1p(vol / 64))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def apply_sched_style(g) -> None:
    """Attach Sched-IR styling callbacks to a heterograph in place."""
    g.vstyle["fillcolor"] = lambda g, vx: SCHED_COLORS.get(g.pmap[vx].get("op", ""), "#FFFFFF")
    g.vstyle["shape"]     = lambda g, vx: SCHED_SHAPES.get(g.pmap[vx].get("op", ""), "box")
    g.vstyle["style"]     = lambda g, vx: "filled"
    g.vstyle["fontcolor"] = lambda g, vx: (
        "white" if g.pmap[vx].get("op") in _DARK_FG_OPS else "black"
    )
    g.vstyle["penwidth"]  = lambda g, vx: "3" if g.pmap[vx].get("critical_path") else "1"
    g.vstyle["color"]     = lambda g, vx: "#E53935" if g.pmap[vx].get("critical_path") else "#333333"
    g.vstyle["label"]     = sched_vx_label

    g.estyle["label"]     = sched_edge_label
    g.estyle["penwidth"]  = sched_edge_penwidth
    g.estyle["fontsize"]  = lambda g, e: "10"
    g.estyle["color"]     = lambda g, e: "#E53935" if (g.pmap[e].get("lifetime") or 0) > 0 else "#666666"
