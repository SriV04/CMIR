"""Build the NN-IR + Sched-IR for the JEDI-linear GNN and open it in the
heterograph web viewer.

Run from the CMIR repo root (or from anywhere — paths are resolved relative
to this file)::

    KERAS_BACKEND=jax conda run -n jedi-linear python IR/main.py

Then open http://localhost:8888 in a browser.

Styling + Gantt rendering have been moved into dedicated modules alongside
the IR they belong to (``IR/NN-IR/styling.py``, ``IR/Sched-IR/styling.py``,
``IR/Sched-IR/gantt.py``). This file is now just an orchestration script.
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
# Hyphenated-directory module loader
# --------------------------------------------------------------------------- #

def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# IR pipeline modules.
nn_ir_builder = _load_path("nn_ir_builder",     HERE / "NN-IR"    / "builder.py")
sched_decomp  = _load_path("sched_decomposer",  HERE / "Sched-IR" / "decomposer.py")
sched_engine  = _load_path("sched_scheduler",   HERE / "Sched-IR" / "binder.py")
sched_folder  = _load_path("sched_folder",      HERE / "Sched-IR" / "folder.py")
sched_p3      = _load_path("sched_p3",          HERE / "Sched-IR" / "scheduler_p3.py")
sched_infra   = _load_path("sched_infra",       HERE / "Sched-IR" / "infrastructure.py")

# Styling + visualisation modules.
nn_styling    = _load_path("nn_ir_styling",     HERE / "NN-IR"    / "styling.py")
sched_styling = _load_path("sched_ir_styling",  HERE / "Sched-IR" / "styling.py")
sched_gantt   = _load_path("sched_ir_gantt",    HERE / "Sched-IR" / "gantt.py")

build_nn_ir       = nn_ir_builder.build_nn_ir
apply_nn_style    = nn_styling.apply_nn_style
apply_sched_style = sched_styling.apply_sched_style
GanttWrapper      = sched_gantt.GanttWrapper
RESOURCE_YAML     = HERE / "Sched-IR" / "da4ml-resource.yaml"


# --------------------------------------------------------------------------- #
# Build the model and NN-IR graph
# --------------------------------------------------------------------------- #

from model import get_gnn                    # from JEDI-linear/src
from heterograph.webview import WebView

conf = SimpleNamespace(n_constituents=8, pt_eta_phi=True)
model = get_gnn(conf)
print(f"[jedi_gnn] keras layers: {len(model.layers)}")

g = build_nn_ir(model, name="jedi_gnn")
print(f"[jedi_gnn] nn-ir: {g.num_vx} vertices, {g.num_edges} edges")


# --------------------------------------------------------------------------- #
# Sched-IR pipeline — decompose → bind → fold(K) → schedule → infrastructure
# --------------------------------------------------------------------------- #

TARGET_FMAX = 300e6  # 300 MHz — typical VU13P clock


def _build_bind():
    g_local = sched_decomp.decompose_nn_to_sched(g)
    g_local = sched_engine.bind(g_local, model, RESOURCE_YAML)
    return g_local


def _build_unscheduled(K: int):
    g_local = _build_bind()
    g_local = sched_folder.fold(g_local, factor=K)
    return g_local


def _build_sched_p3(K: int):
    g_local = _build_unscheduled(K)
    g_local = sched_p3.schedule(g_local)
    g_local = sched_p3.steady_state(g_local, fmax=TARGET_FMAX)
    return g_local


def _build_sched(K: int):
    g_local = _build_sched_p3(K)
    g_local = sched_infra.insert_buffers(g_local)
    return g_local


g_bind        = _build_bind()           # decompose + bind (no fold/schedule)
g_unsched     = _build_unscheduled(1)   # K=1 pre-schedule
g_unsched_k4  = _build_unscheduled(4)   # K=4 pre-schedule
g_sched_p3    = _build_sched_p3(1)      # baseline schedule output
g_sched_p3_k4 = _build_sched_p3(4)      # hybrid schedule output
g_sched       = _build_sched(1)         # baseline + infrastructure
g_sched_k4    = _build_sched(4)         # hybrid fold + infrastructure


def _summary(label, gx):
    ms  = gx.pmap.get("makespan", "?")
    ii  = gx.pmap.get("initiation_interval", "?")
    tp  = gx.pmap.get("sustained_throughput_hz")
    bif = gx.pmap.get("batches_in_flight", "?")
    tp_s    = f"{tp/1e6:.0f} MHz" if tp else "?"
    t_lut   = gx.pmap.get("total_luts",  "?")
    t_ff    = gx.pmap.get("total_ffs",   "?")
    t_bram  = gx.pmap.get("total_brams", "?")
    n_buf   = sum(1 for v in gx.vertices if gx.pmap[v].get("op") == "buffer")
    print(
        f"[jedi_gnn] {label}: {gx.num_vx} vx ({n_buf} bufs), "
        f"LUT={t_lut} FF={t_ff} BRAM={t_bram}, "
        f"makespan={ms} cyc, II={ii}, throughput={tp_s}, in-flight={bif}"
    )


_summary("sched K=1",    g_sched)
_summary("sched K=4",    g_sched_k4)
_summary("sched p3 K=1", g_sched_p3)
_summary("sched p3 K=4", g_sched_p3_k4)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

apply_nn_style(g)

for gx in (g_bind, g_unsched, g_unsched_k4, g_sched_p3, g_sched_p3_k4, g_sched, g_sched_k4):
    apply_sched_style(gx)


# --------------------------------------------------------------------------- #
# Web view
# --------------------------------------------------------------------------- #

def _tab_title(label, gx):
    t_lut = gx.pmap.get("total_luts", "?")
    t_ff  = gx.pmap.get("total_ffs", 0)
    ms    = gx.pmap.get("makespan", "?")
    ii    = gx.pmap.get("initiation_interval", "?")
    tp    = gx.pmap.get("sustained_throughput_hz")
    tp_s  = f", {tp/1e6:.0f} MHz" if tp else ""
    ff_s  = f" FF={t_ff}" if t_ff else ""
    return f"{label} — LUT={t_lut}{ff_s}, {ms} cyc, II={ii}{tp_s}"


wv = WebView()
wv.add_graph(g,                       title="JEDI-linear NN-IR")
wv.add_graph(g_bind,                  title="Sched BIND (unscheduled)")
wv.add_graph(g_unsched,               title="Sched K=1 (unscheduled)")
wv.add_graph(g_unsched_k4,            title="Sched K=4 (unscheduled)")
wv.add_graph(g_sched_p3,              title=_tab_title("Sched P3 K=1", g_sched_p3))
wv.add_graph(g_sched_p3_k4,           title=_tab_title("Sched P3 K=4", g_sched_p3_k4))
wv.add_graph(g_sched,                 title=_tab_title("Sched K=1",    g_sched))
wv.add_graph(g_sched_k4,              title=_tab_title("Sched K=4",    g_sched_k4))
wv.add_graph(GanttWrapper(g_sched),   title=_tab_title("Gantt K=1",    g_sched))
wv.add_graph(GanttWrapper(g_sched_k4),title=_tab_title("Gantt K=4",    g_sched_k4))
print("Serving on http://localhost:8888  (Ctrl-C to stop)")
wv.run(host="127.0.0.1", port="8888")
