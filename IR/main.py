"""Build the NN-IR for the JEDI-linear GNN and open it in the heterograph web viewer.

Run from the CMIR repo root (or from anywhere — paths are resolved relative to
this file):

    KERAS_BACKEND=jax conda run -n jedi-linear python IR/main.py

Then open http://localhost:8888 in a browser.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, str(REPO / "JEDI-linear" / "src"))
sys.path.insert(0, str(REPO / "heterograph"))


# --------------------------------------------------------------------------- #
# Load the NN-IR builder (IR/NN-IR has a hyphen, so we go via importlib)
# --------------------------------------------------------------------------- #

def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required for dataclasses / introspection helpers
    spec.loader.exec_module(mod)
    return mod


nn_ir_builder = _load_path("nn_ir_builder", HERE / "NN-IR" / "builder.py")
sched_decomp = _load_path("sched_decomposer", HERE / "Sched-IR" / "decomposer.py")
sched_engine = _load_path("sched_scheduler", HERE / "Sched-IR" / "scheduler.py")
sched_folder = _load_path("sched_folder", HERE / "Sched-IR" / "folder.py")
sched_p3     = _load_path("sched_p3", HERE / "Sched-IR" / "scheduler_p3.py")
build_nn_ir = nn_ir_builder.build_nn_ir
RESOURCE_YAML = HERE / "Sched-IR" / "da4ml-resource.yaml"


# --------------------------------------------------------------------------- #
# Build the model and NN-IR graph
# --------------------------------------------------------------------------- #

from model import get_gnn  # from JEDI-linear/src
from heterograph.webview import WebView

conf = SimpleNamespace(n_constituents=8, pt_eta_phi=True)
model = get_gnn(conf)
print(f"[jedi_gnn] keras layers: {len(model.layers)}")

g = build_nn_ir(model, name="jedi_gnn")
print(f"[jedi_gnn] nn-ir: {g.num_vx} vertices, {g.num_edges} edges")

# --------------------------------------------------------------------------- #
# Sched-IR — Decompose → BIND → FOLD, repeated for K=1 (baseline) and K=4
# (hybrid). Each fold factor produces its own scheduled graph; the webview
# shows them side by side so we can eyeball the area/latency trade.
# --------------------------------------------------------------------------- #

def _build_sched(K: int):
    g_local = sched_decomp.decompose_nn_to_sched(g)
    g_local = sched_engine.bind(g_local, model, RESOURCE_YAML)
    g_local = sched_folder.fold(g_local, factor=K)
    g_local = sched_p3.schedule(g_local)
    return g_local


def _area_weighted_lut(g_local) -> int:
    total = 0
    for v in g_local.vertices:
        p = g_local.pmap[v]
        cost = p.get("cost") or {}
        inst = p.get("physical_instances") or 1
        total += int(cost.get("lut") or 0) * int(inst)
    return total


g_sched = _build_sched(1)              # baseline
g_sched_k4 = _build_sched(4)           # hybrid fold

def _summary(label, gx):
    lut = _area_weighted_lut(gx)
    ms = gx.pmap.get("makespan", "?")
    ii = gx.pmap.get("initiation_interval", "?")
    print(f"[jedi_gnn] {label}: {gx.num_vx} vx / {gx.num_edges} ed, "
          f"LUT={lut}, makespan={ms} cyc, II={ii}")


_summary("sched K=1", g_sched)
_summary("sched K=4", g_sched_k4)


# --------------------------------------------------------------------------- #
# Styling — color by op_kind, label with shapes/bitwidths, edges by volume
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
    "qsum":            "invtrapezium",  # fan-in
    "qadd":            "diamond",
    "activation":      "oval",
}


def _fmt_shape(s):
    if s is None:
        return "?"
    return "x".join("?" if d is None else str(d) for d in s)


def vx_label(g, vx):
    p = g.pmap[vx]
    name = p["layer_name"]
    kind = p["op_kind"]
    in_s = ",".join(_fmt_shape(s) for s in (p["in_shapes"] or [])) or "—"
    out_s = ",".join(_fmt_shape(s) for s in (p["out_shapes"] or [])) or "—"
    lines = [f"{name}", f"[{kind}]", f"in: {in_s}", f"out: {out_s}"]
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


def edge_label(g, e):
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


def edge_penwidth(g, e):
    vol = g.pmap[e].get("volume_bits") or 0
    # log-ish scaling so 48b and 1024b are both visible
    import math
    return str(0.6 + 0.6 * math.log1p(vol / 64))


g.vstyle["fillcolor"] = lambda g, vx: OP_COLORS.get(g.pmap[vx]["op_kind"], "#FFFFFF")
g.vstyle["shape"] = lambda g, vx: OP_SHAPES.get(g.pmap[vx]["op_kind"], "box")
g.vstyle["style"] = lambda g, vx: "filled"
g.vstyle["fontcolor"] = lambda g, vx: (
    "white" if g.pmap[vx]["op_kind"] in ("einsum_dense_bn", "einsum_dense", "dense", "qsum", "qadd") else "black"
)
g.vstyle["label"] = vx_label

g.estyle["label"] = edge_label
g.estyle["penwidth"] = edge_penwidth
g.estyle["fontsize"] = lambda g, e: "10"


# --------------------------------------------------------------------------- #
# Sched-IR styling — color by primitive op, label with op_params and fold axes
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


def sched_vx_label(g, vx):
    p = g.pmap[vx]
    name = p.get("nn_layer_name") or "?"
    op = p.get("op") or "?"
    lines = [name, f"[{op}]"]
    fa = p.get("fold_axes")
    if fa:
        lines.append(f"fold axes: {','.join(str(a) for a in fa)}")
    pp = p.get("op_params") or {}
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
        rm = p.get("reduce_mode")
        if rm:
            lines.append(f"mode: {rm}")
        if pp.get("in_bw") is not None and pp.get("out_bw") is not None:
            lines.append(f"bw: {pp['in_bw']:.1f}->{pp['out_bw']:.1f}")
    elif op == "elementwise":
        lines.append(f"op: {pp.get('op','?')}")
        ins = pp.get("in_shapes") or []
        for i, s in enumerate(ins):
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

    # ---- Phase 1 BIND output: kernel + cost ---- #
    kt = p.get("kernel_type")
    if kt is not None:
        ki = p.get("kernel_instance")
        lines.append(f"kernel: {kt}#{ki}")
    cost = p.get("cost") or {}
    if cost:
        bits = []
        if cost.get("lut"):            bits.append(f"lut={cost['lut']}")
        if cost.get("ff"):             bits.append(f"ff={cost['ff']}")
        if cost.get("dsp"):            bits.append(f"dsp={cost['dsp']}")
        if cost.get("bram"):           bits.append(f"bram={cost['bram']}")
        if cost.get("latency_cycles"): bits.append(f"lat={cost['latency_cycles']}")
        if cost.get("ii", 1) and cost.get("ii") != 1: bits.append(f"ii={cost['ii']}")
        if bits:
            lines.append(" ".join(bits))

    # ---- Phase 2 FOLD output: K, instance count, group id, reduce mode ---- #
    K = p.get("fold_factor")
    inst = p.get("physical_instances")
    fg = p.get("fold_group")
    if K is not None and (K != 1 or inst not in (None, 1) or fg is not None):
        line = f"K={K} inst={inst}"
        if fg is not None:
            line += f" g{fg}"
        lines.append(line)
    rm = p.get("reduce_mode")
    if rm and rm != "spatial":
        lines.append(f"reduce: {rm}")

    # ---- Phase 3 SCHEDULE output: timing ---- #
    ts = p.get("t_start")
    te = p.get("t_end")
    if ts is not None and te is not None:
        lines.append(f"t=[{ts}..{te}]")
    return "\n".join(lines)


def sched_edge_label(g, e):
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


def sched_edge_penwidth(g, e):
    import math
    vol = g.pmap[e].get("volume_bits") or 0
    return str(0.6 + 0.6 * math.log1p(vol / 64))


def _apply_sched_style(gx):
    gx.vstyle["fillcolor"] = lambda g, vx: SCHED_COLORS.get(g.pmap[vx]["op"], "#FFFFFF")
    gx.vstyle["shape"] = lambda g, vx: SCHED_SHAPES.get(g.pmap[vx]["op"], "box")
    gx.vstyle["style"] = lambda g, vx: "filled"
    gx.vstyle["fontcolor"] = lambda g, vx: (
        "white" if g.pmap[vx]["op"] in ("dense", "reduce", "elementwise", "buffer", "mux") else "black"
    )
    gx.vstyle["penwidth"] = lambda g, vx: "3" if g.pmap[vx].get("critical_path") else "1"
    gx.vstyle["color"] = lambda g, vx: "#E53935" if g.pmap[vx].get("critical_path") else "#333333"
    gx.vstyle["label"] = sched_vx_label
    gx.estyle["label"] = sched_edge_label
    gx.estyle["penwidth"] = sched_edge_penwidth
    gx.estyle["fontsize"] = lambda g, e: "10"
    gx.estyle["color"] = lambda g, e: "#E53935" if g.pmap[e].get("lifetime", 0) > 0 else "#666666"


_apply_sched_style(g_sched)
_apply_sched_style(g_sched_k4)


# --------------------------------------------------------------------------- #
# Gantt chart wrapper — feeds raw SVG into the WebView
# --------------------------------------------------------------------------- #

class GanttWrapper:
    """Thin wrapper so WebView.add_graph can render a Gantt SVG."""

    def __init__(self, g_sched):
        self._svg = _render_gantt_svg(g_sched)
        self.style = {}

    def render(self, *, format="svg", pipe=False, **kwargs):
        return self._svg


def _render_gantt_svg(gx):
    """Generate a cycle-accurate Gantt chart as raw SVG bytes."""
    makespan = int(gx.pmap.get("makespan") or 1)
    crit_set = set(gx.pmap.get("critical_path") or [])

    # Collect rows in topological order (source → sink).
    rows = []
    for v in gx.vertices:
        p = gx.pmap[v]
        rows.append({
            "vx": v,
            "name": p.get("nn_layer_name") or "?",
            "op": p.get("op") or "?",
            "t_start": int(p.get("t_start") or 0),
            "t_end": int(p.get("t_end") or 1),
            "K": int(p.get("fold_factor") or 1),
            "crit": v in crit_set,
        })

    # Collect buffer edges (lifetime > 0)
    buf_edges = []
    for u, v in gx.edges:
        ep = gx.pmap[(u, v)]
        lt = ep.get("lifetime", 0)
        if lt and lt > 0:
            buf_edges.append({
                "src": u, "dst": v,
                "t_produce": int(ep.get("t_produce", 0)),
                "t_consume": int(ep.get("t_consume", 0)),
                "lifetime": int(lt),
            })

    # Layout constants
    PX_PER_CYCLE = max(18, min(30, 900 // max(makespan, 1)))
    ROW_H = 36
    LABEL_W = 240
    PAD = 20
    HEADER_H = 30

    chart_w = makespan * PX_PER_CYCLE
    total_w = LABEL_W + chart_w + 2 * PAD
    total_h = HEADER_H + len(rows) * ROW_H + 2 * PAD

    vx_to_row = {r["vx"]: i for i, r in enumerate(rows)}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}pt" height="{total_h}pt" '
        f'viewBox="0 0 {total_w} {total_h}" '
        f'font-family="monospace" font-size="11">',
        f'<g id="graph0" class="graph" transform="translate(0,0)">',
        f'<rect width="{total_w}" height="{total_h}" fill="#1e1e1e"/>',
    ]

    x0 = LABEL_W + PAD
    y0 = HEADER_H + PAD

    # Cycle grid lines
    for c in range(0, makespan + 1, max(1, makespan // 20)):
        x = x0 + c * PX_PER_CYCLE
        parts.append(f'<line x1="{x}" y1="{y0 - 5}" x2="{x}" y2="{y0 + len(rows) * ROW_H}" '
                     f'stroke="#444" stroke-width="0.5"/>')
        parts.append(f'<text x="{x}" y="{y0 - 8}" fill="#aaa" font-size="9" text-anchor="middle">{c}</text>')

    # Rows
    for i, r in enumerate(rows):
        y = y0 + i * ROW_H
        bw = (r["t_end"] - r["t_start"]) * PX_PER_CYCLE
        bx = x0 + r["t_start"] * PX_PER_CYCLE

        color = SCHED_COLORS.get(r["op"], "#888")
        border = "#E53935" if r["crit"] else "#555"
        bw_px = max(bw, 2)

        # Label
        parts.append(f'<text x="{LABEL_W}" y="{y + ROW_H // 2 + 4}" fill="#ccc" '
                     f'text-anchor="end" font-size="10">{r["name"]}</text>')

        # Bar
        parts.append(f'<rect x="{bx}" y="{y + 4}" width="{bw_px}" height="{ROW_H - 8}" '
                     f'rx="3" fill="{color}" stroke="{border}" stroke-width="{"2" if r["crit"] else "1"}"/>')

        # Timing text inside bar
        label = f'{r["t_start"]}..{r["t_end"]}'
        if r["K"] > 1:
            label += f' K={r["K"]}'
        if bw_px > 50:
            parts.append(f'<text x="{bx + bw_px // 2}" y="{y + ROW_H // 2 + 4}" '
                         f'fill="white" font-size="9" text-anchor="middle">{label}</text>')

    # Buffer arrows
    for be in buf_edges:
        ri = vx_to_row.get(be["src"])
        rj = vx_to_row.get(be["dst"])
        if ri is None or rj is None:
            continue
        x1 = x0 + be["t_produce"] * PX_PER_CYCLE
        y1 = y0 + ri * ROW_H + ROW_H // 2
        x2 = x0 + be["t_consume"] * PX_PER_CYCLE
        y2 = y0 + rj * ROW_H + ROW_H // 2
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     f'stroke="#E53935" stroke-width="1.5" stroke-dasharray="4,3" '
                     f'marker-end="url(#arr)"/>')

    # Arrow marker definition — insert after <g> (index 2)
    parts.insert(2,
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#E53935"/></marker></defs>'
    )

    # Title / legend
    ms = gx.pmap.get("makespan", "?")
    ii = gx.pmap.get("initiation_interval", "?")
    parts.append(f'<text x="{PAD}" y="16" fill="#eee" font-size="12" font-weight="bold">'
                 f'Gantt — makespan={ms} cycles, II={ii}</text>')

    parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts).encode("utf-8")


# --------------------------------------------------------------------------- #
# Web view
# --------------------------------------------------------------------------- #

wv = WebView()
wv.add_graph(g, title="JEDI-linear NN-IR")
wv.add_graph(g_sched,    title=f"Sched-IR K=1 (baseline) — LUT={_area_weighted_lut(g_sched)}, makespan={g_sched.pmap.get('makespan')}")
wv.add_graph(g_sched_k4, title=f"Sched-IR K=4 (hybrid) — LUT={_area_weighted_lut(g_sched_k4)}, makespan={g_sched_k4.pmap.get('makespan')}")
wv.add_graph(GanttWrapper(g_sched),    title="Gantt K=1")
wv.add_graph(GanttWrapper(g_sched_k4), title="Gantt K=4")
print("Serving on http://localhost:8888  (Ctrl-C to stop)")
wv.run(host="127.0.0.1", port="8888")
