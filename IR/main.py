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
    spec.loader.exec_module(mod)
    return mod


nn_ir_builder = _load_path("nn_ir_builder", HERE / "NN-IR" / "builder.py")
sched_decomp = _load_path("sched_decomposer", HERE / "Sched-IR" / "decomposer.py")
build_nn_ir = nn_ir_builder.build_nn_ir


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
# Sched-IR First pass:
g_sched = sched_decomp.decompose_nn_to_sched(g)


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
    return tag


def sched_edge_penwidth(g, e):
    import math
    vol = g.pmap[e].get("volume_bits") or 0
    return str(0.6 + 0.6 * math.log1p(vol / 64))


g_sched.vstyle["fillcolor"] = lambda g, vx: SCHED_COLORS.get(g.pmap[vx]["op"], "#FFFFFF")
g_sched.vstyle["shape"] = lambda g, vx: SCHED_SHAPES.get(g.pmap[vx]["op"], "box")
g_sched.vstyle["style"] = lambda g, vx: "filled"
g_sched.vstyle["fontcolor"] = lambda g, vx: (
    "white" if g.pmap[vx]["op"] in ("dense", "reduce", "elementwise", "buffer", "mux") else "black"
)
g_sched.vstyle["label"] = sched_vx_label

g_sched.estyle["label"] = sched_edge_label
g_sched.estyle["penwidth"] = sched_edge_penwidth
g_sched.estyle["fontsize"] = lambda g, e: "10"


# --------------------------------------------------------------------------- #
# Web view
# --------------------------------------------------------------------------- #

wv = WebView()
wv.add_graph(g, title="JEDI-linear NN-IR")
wv.add_graph(g_sched, title="JEDI-linear Sched-IR (1st pass)")
print("Serving on http://localhost:8888  (Ctrl-C to stop)")
wv.run(host="127.0.0.1", port="8888")
