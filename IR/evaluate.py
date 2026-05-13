"""CMIR Evaluation Metrics — sweep fold factors and report all metrics.

Builds Sched-IR graphs for K ∈ {1, 2, 4, 8}, computes primary / secondary /
diagnostic metrics from the literature (hls4ml, da4ml, JEDI-linear, FINN),
and prints publication-ready tables + a Pareto plot data summary.

Run from the CMIR repo root:

    KERAS_BACKEND=jax conda run -n jedi-linear python IR/evaluate.py
"""

from __future__ import annotations

import glob
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
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
sys.path.insert(0, str(HERE))   # so `import plot` resolves


# --------------------------------------------------------------------------- #
# Load modules (hyphenated directories → importlib)
# --------------------------------------------------------------------------- #

def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nn_ir_builder = _load_path("nn_ir_builder", HERE / "NN-IR" / "builder.py")
sched_decomp  = _load_path("sched_decomposer", HERE / "Sched-IR" / "decomposer.py")
sched_engine  = _load_path("sched_scheduler", HERE / "Sched-IR" / "binder.py")
sched_folder  = _load_path("sched_folder", HERE / "Sched-IR" / "folder.py")
sched_p3      = _load_path("sched_p3", HERE / "Sched-IR" / "scheduler_p3.py")
sched_infra   = _load_path("sched_infra", HERE / "Sched-IR" / "infrastructure.py")
build_nn_ir   = nn_ir_builder.build_nn_ir
RESOURCE_YAML = HERE / "Sched-IR" / "da4ml-resource.yaml"

from model import get_gnn  # from JEDI-linear/src
from heterograph import HGraph

import yaml as _yaml

# --------------------------------------------------------------------------- #
# Build model + NN-IR (shared across all sweeps)
# --------------------------------------------------------------------------- #

# Single source of truth: read fpga config from the resource YAML and resolve
# `latency_cutoff: auto` once. Both path A (end-to-end ground truth) and path B
# (per-layer Sched-IR via BIND) read the same value from this dict.
_RESOURCE_CFG  = _yaml.safe_load(RESOURCE_YAML.read_text())
_FPGA_CFG      = sched_engine.normalize_fpga(_RESOURCE_CFG.get("fpga") or {})

TARGET_FMAX             = float(_FPGA_CFG.get("target_fmax_hz") or 300e6)
PIPELINE_LATENCY_CUTOFF = int(_FPGA_CFG.get("latency_cutoff", -1))
N_CONSTITUENTS = 8
USE_PERMINV = True            # uq1 variant — matches JEDI-linear paper's permutation-invariant model
LOAD_TRAINED_WEIGHTS = True   # required for realistic DA cost (random weights yield trivially low cost)

# JEDI-linear's official_models tarball stores trained checkpoints we should reuse.
_VARIANT_DIR = "3-feature-perminv" if USE_PERMINV else "3-feature"
_CHECKPOINT_GLOB = (
    REPO / "official_models" / _VARIANT_DIR / f"jet_classifier_large_{N_CONSTITUENTS}"
    / "models" / "*.keras"
)


def _load_trained_model():
    """Load the official JEDI-linear checkpoint matching (variant, N).

    Returns the loaded Keras model. Falls back to a freshly-built model only
    if no checkpoint is found — but emits a loud warning, since the DA cost
    queries depend on the trained weights to produce realistic estimates.
    """
    import keras
    import hgq  # noqa: F401  registers HGQ custom layers for keras.models.load_model

    if LOAD_TRAINED_WEIGHTS:
        ckpts = sorted(glob.glob(str(_CHECKPOINT_GLOB)))
        if ckpts:
            print(f"Loading trained model: {Path(ckpts[0]).name}")
            return keras.models.load_model(ckpts[0])
        print(f"WARN: no checkpoint at {_CHECKPOINT_GLOB} — falling back to fresh weights")

    conf = SimpleNamespace(n_constituents=N_CONSTITUENTS, pt_eta_phi=True)
    return get_gnn(conf, uq1=USE_PERMINV)


model = _load_trained_model()
g_nnir = build_nn_ir(model, name="jedi_gnn")

print(f"\n{'='*80}")
print(f"  CMIR EVALUATION — JEDI-linear GNN ({len(model.layers)} Keras layers)")
print(f"  Variant: {_VARIANT_DIR}, N={N_CONSTITUENTS}, weights={'trained' if LOAD_TRAINED_WEIGHTS else 'fresh'}")
print(f"  NN-IR: {g_nnir.num_vx} vertices, {g_nnir.num_edges} edges")
print(f"  Target: VU13P @ {TARGET_FMAX/1e6:.0f} MHz, latency_cutoff={PIPELINE_LATENCY_CUTOFF}")
print(f"{'='*80}\n")


# --------------------------------------------------------------------------- #
# Build scheduled graphs for each fold factor
# --------------------------------------------------------------------------- #

FOLD_FACTORS = [1, 2, 4, 8]


def build_scheduled(K: int) -> HGraph:
    """Decompose → Fold-plan(K) → Bind+Propagate → Timing → Schedule → Steady_state → Insert_buffers."""
    g = sched_decomp.decompose_nn_to_sched(g_nnir)
    g = sched_folder.stamp_fold_plan(g, factor=K)
    g = sched_engine.bind_and_propagate(g, model, RESOURCE_YAML)
    g = sched_folder.apply_timing_from_costs(g)
    g = sched_p3.schedule(g)
    g = sched_p3.steady_state(g, fmax=TARGET_FMAX)
    g = sched_infra.insert_buffers(g)
    return g


# --------------------------------------------------------------------------- #
# End-to-end DA ground truth (K=1 only)
#
# The per-layer Sched-IR rollup undercounts because it (a) loses cross-layer
# bitwidth/quantizer info by collapsing to scalar averages, and (b) doesn't
# pipeline-register-count compute kernels. For the *fully unfolded* dataflow,
# da4ml's full-model trace + to_pipeline gives the authoritative cost — it's
# exactly the path the JEDI-linear paper reports.
# --------------------------------------------------------------------------- #

def end_to_end_ground_truth() -> dict | None:
    """Trace the entire model through da4ml and report (LUT, FF, latency).

    Mirrors `JEDI-linear/src/syn_test.py::syn_test_verilog`: trace the model,
    pipeline at the same latency cutoff, then read `cost` (LUTs) and
    `reg_bits` (FFs) off the resulting CascadedSolution.
    """
    try:
        from da4ml.converter.hgq2.parser import trace_model
        from da4ml.trace import comb_trace, HWConfig
        from da4ml.trace.pipeline import to_pipeline
    except Exception as e:
        print(f"  end-to-end ground truth unavailable: {e}")
        return None

    inp, out = trace_model(model, solver_options={"hard_dc": 2}, hwconf=HWConfig(1, -1, -1))
    sol = comb_trace(inp, out)
    pipe = to_pipeline(sol, PIPELINE_LATENCY_CUTOFF, retiming=True, verbose=False)

    lat_min, lat_max = pipe.latency
    return {
        "lut": int(round(float(pipe.cost))),
        "ff": int(pipe.reg_bits),
        "stages": int(len(pipe.solutions)),
        "logic_latency_min": float(lat_min),
        "logic_latency_max": float(lat_max),
    }


GROUND_TRUTH = end_to_end_ground_truth()
if GROUND_TRUTH is not None:
    gt = GROUND_TRUTH
    gt_ns = gt["stages"] * (1e9 / TARGET_FMAX)
    print(
        f"  Ground truth (end-to-end DA, K=1):\n"
        f"    LUT={gt['lut']:,}  FF={gt['ff']:,}  "
        f"latency={gt['stages']} cyc ({gt_ns:.1f} ns)  II=1\n"
    )


print("Building scheduled graphs...")
graphs: dict[int, HGraph] = {}
for K in FOLD_FACTORS:
    try:
        graphs[K] = build_scheduled(K)
        print(f"  K={K}: ✓")
    except Exception as e:
        print(f"  K={K}: ✗ ({e})")

if not graphs:
    print("ERROR: No graphs could be built. Aborting.")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Metric extraction helpers
# --------------------------------------------------------------------------- #

def _gv(g: HGraph, key: str, default=0):
    """Get graph-level pmap value."""
    return g.pmap.get(key, default) or default


def _vertex_iter(g: HGraph, op: str | None = None):
    """Iterate over vertices, optionally filtering by op type."""
    for vx in g.vertices:
        p = g.pmap[vx]
        if op is None or p.get("op") == op:
            yield vx, p


def compute_metrics(g: HGraph, K: int) -> dict:
    """Extract all evaluation metrics from a scheduled + buffered graph."""
    m = {}

    # ---- Graph-level basics ---- #
    m["K"]                  = K
    m["num_vertices"]       = g.num_vx
    m["num_edges"]          = g.num_edges
    m["makespan"]           = int(_gv(g, "makespan"))
    m["II"]                 = int(_gv(g, "initiation_interval", 1))
    m["pipeline_depth"]     = int(_gv(g, "pipeline_depth"))
    m["throughput_hz"]      = _gv(g, "sustained_throughput_hz")
    m["throughput_mhz"]     = m["throughput_hz"] / 1e6 if m["throughput_hz"] else None
    m["batches_in_flight"]  = int(_gv(g, "batches_in_flight", 1))

    # ---- Area totals ---- #
    m["total_luts"]  = int(_gv(g, "total_luts"))
    m["total_ffs"]   = int(_gv(g, "total_ffs"))
    m["total_dsps"]  = int(_gv(g, "total_dsps"))
    m["total_brams"] = int(_gv(g, "total_brams"))

    # For K=1 (fully unfolded) the per-layer Sched-IR rollup is a lower bound
    # — it loses the cross-layer bitwidth tracking and pipeline-register
    # accounting that the end-to-end DA flow performs. Override with the DA
    # ground truth so the K=1 row matches the JEDI-linear paper.
    if K == 1 and GROUND_TRUTH is not None:
        m["total_luts"]     = GROUND_TRUTH["lut"]
        m["total_ffs"]      = GROUND_TRUTH["ff"]
        m["makespan"]       = GROUND_TRUTH["stages"]
        m["pipeline_depth"] = GROUND_TRUTH["stages"]
        m["batches_in_flight"] = GROUND_TRUTH["stages"]   # II=1 → one batch/stage
        m["ground_truth"]   = True

    m["makespan_ns"] = m["makespan"] * (1e9 / TARGET_FMAX) if TARGET_FMAX else None

    # ---- Infrastructure breakdown ---- #
    compute_luts = 0
    compute_ffs = 0
    buffer_luts = 0
    buffer_ffs = 0
    buffer_brams = 0
    mux_luts = 0
    n_buffers = 0
    n_muxes = 0

    # Multiplier convention (matches Sched-IR/infrastructure.py:_rollup): dense
    # costs are per-lane and scale with P; reduce/elementwise/buffer/mux costs
    # cover the full hardware structure and are not replicated.
    def _mult_for(op: str, inst: int) -> int:
        if op in ("reduce", "elementwise", "buffer", "mux"):
            return 1
        return inst

    for vx, p in _vertex_iter(g):
        cost = p.get("cost") or {}
        inst = int(p.get("physical_instances") or 1)
        op = p.get("op", "")
        mult = _mult_for(op, inst)

        if op == "buffer":
            n_buffers += 1
            buffer_luts  += int(cost.get("lut", 0)) * mult
            buffer_ffs   += int(cost.get("ff", 0)) * mult
            buffer_brams += int(cost.get("bram", 0)) * mult
        elif op == "mux":
            n_muxes += 1
            mux_luts += int(cost.get("lut", 0)) * mult
        else:
            compute_luts += int(cost.get("lut", 0)) * mult
            compute_ffs  += int(cost.get("ff", 0)) * mult

    m["compute_luts"]   = compute_luts
    m["compute_ffs"]    = compute_ffs
    m["buffer_luts"]    = buffer_luts
    m["buffer_ffs"]     = buffer_ffs
    m["buffer_brams"]   = buffer_brams
    m["mux_luts"]       = mux_luts
    m["n_buffers"]      = n_buffers
    m["n_muxes"]        = n_muxes
    m["infra_luts"]     = buffer_luts + mux_luts
    m["infra_ffs"]      = buffer_ffs

    # ---- Buffer overhead ratio (S1) ---- #
    m["buffer_overhead_ratio"] = (
        m["infra_luts"] / m["total_luts"] if m["total_luts"] > 0 else 0.0
    )

    # ---- Critical path utilisation (S3) ---- #
    m["critical_path_utilisation"] = (
        m["pipeline_depth"] / m["makespan"] if m["makespan"] > 0 else 0.0
    )

    # ---- Per-vertex slack distribution (S4) ---- #
    slacks = []
    for vx, p in _vertex_iter(g):
        if p.get("op") in ("buffer", "mux"):
            continue
        ts = p.get("t_start")
        tr = p.get("t_ready")
        if ts is not None and tr is not None:
            slacks.append(int(ts) - int(tr))
    m["slack_min"]  = min(slacks) if slacks else 0
    m["slack_max"]  = max(slacks) if slacks else 0
    m["slack_mean"] = sum(slacks) / len(slacks) if slacks else 0.0
    m["slack_zero_pct"] = (
        sum(1 for s in slacks if s == 0) / len(slacks) * 100 if slacks else 0.0
    )

    # ---- Schedule compactness (D1) ---- #
    total_occupied = sum(
        int(p.get("t_end", 0)) - int(p.get("t_start", 0))
        for _, p in _vertex_iter(g)
        if p.get("op") not in ("buffer", "mux")
    )
    n_compute_vx = sum(1 for _, p in _vertex_iter(g) if p.get("op") not in ("buffer", "mux"))
    m["schedule_compactness"] = (
        total_occupied / (n_compute_vx * m["makespan"])
        if n_compute_vx > 0 and m["makespan"] > 0
        else 0.0
    )

    # ---- Edge lifetime distribution (D2) ---- #
    lifetimes = []
    for u, v in g.edges:
        ep = g.pmap[(u, v)]
        lt = ep.get("lifetime")
        if lt is not None:
            lifetimes.append(int(lt))
    m["lifetime_max"]  = max(lifetimes) if lifetimes else 0
    m["lifetime_mean"] = sum(lifetimes) / len(lifetimes) if lifetimes else 0.0
    m["lifetime_nonzero_count"] = sum(1 for lt in lifetimes if lt > 0)

    # ---- Per-op-type cost breakdown ---- #
    op_costs: dict[str, dict[str, int]] = defaultdict(lambda: {"lut": 0, "ff": 0, "dsp": 0, "bram": 0, "count": 0})
    for vx, p in _vertex_iter(g):
        op = p.get("op", "unknown")
        cost = p.get("cost") or {}
        inst = int(p.get("physical_instances") or 1)
        mult = _mult_for(op, inst)
        op_costs[op]["lut"]  += int(cost.get("lut", 0)) * mult
        op_costs[op]["ff"]   += int(cost.get("ff", 0)) * mult
        op_costs[op]["dsp"]  += int(cost.get("dsp", 0)) * mult
        op_costs[op]["bram"] += int(cost.get("bram", 0)) * mult
        op_costs[op]["count"] += 1
    m["op_breakdown"] = dict(op_costs)

    # ---- Fold-group balance (S5) ---- #
    fold_plan = g.pmap.get("fold_plan") or []
    group_iis = [entry.get("factor", 1) for entry in fold_plan]
    if group_iis:
        m["fold_group_balance"] = max(group_iis) / max(min(group_iis), 1)
    else:
        m["fold_group_balance"] = 1.0

    # ---- Throughput bottleneck (D3) ---- #
    bottleneck = g.pmap.get("throughput_bottleneck")
    if bottleneck:
        m["throughput_bottleneck"] = [
            g.pmap[v].get("nn_layer_name", str(v)) for v in bottleneck
        ]
    else:
        m["throughput_bottleneck"] = []

    # ---- Area-Latency Product (P3) ---- #
    m["area_latency_product"] = m["total_luts"] * m["makespan"]

    # ---- Area Efficiency (P5) ---- #
    m["area_efficiency"] = (
        m["throughput_hz"] / m["total_luts"] if m["total_luts"] > 0 and m["throughput_hz"] else 0.0
    )

    return m


# --------------------------------------------------------------------------- #
# Compute metrics for all fold factors
# --------------------------------------------------------------------------- #

print("\nComputing metrics...\n")
all_metrics: dict[int, dict] = {}
for K, g in graphs.items():
    all_metrics[K] = compute_metrics(g, K)


# --------------------------------------------------------------------------- #
# Compute relative metrics (vs K=1 baseline)
# --------------------------------------------------------------------------- #

baseline = all_metrics.get(1)
if baseline:
    for K, m in all_metrics.items():
        # P1: Latency Reduction Ratio (>1 means baseline is faster)
        m["latency_ratio"] = m["makespan"] / baseline["makespan"] if baseline["makespan"] > 0 else 1.0
        # P2: Area Reduction Ratio (>1 means K saves area vs baseline)
        m["area_reduction_ratio"] = baseline["total_luts"] / m["total_luts"] if m["total_luts"] > 0 else 1.0
        # ALP normalised
        m["alp_normalised"] = m["area_latency_product"] / baseline["area_latency_product"] if baseline["area_latency_product"] > 0 else 1.0
        # DSP savings (S7)
        m["dsp_saved"] = baseline["total_dsps"] - m["total_dsps"]


# --------------------------------------------------------------------------- #
# Pretty-print tables
# --------------------------------------------------------------------------- #

def _sep(w=80):
    print("─" * w)


def _header(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


# ---- TABLE 1: Primary Metrics ---- #
_header("TABLE 1: Primary Metrics (Paper-Ready)")

cols = ["K", "LUTs", "FFs", "DSPs", "BRAMs", "Makespan\n(cyc)", "Makespan\n(ns)", "II",
        "Throughput\n(MHz)", "Batches\nin Flight", "ALP\n(norm)", "AE\n(inf/s/LUT)"]
print(f"{'K':>3} │ {'LUTs':>8} │ {'FFs':>8} │ {'DSPs':>5} │ {'BRAMs':>5} │ "
      f"{'Mksp(cyc)':>9} │ {'Mksp(ns)':>8} │ {'II':>3} │ "
      f"{'Tput(MHz)':>9} │ {'BIF':>3} │ {'ALP(n)':>7} │ {'AE':>12}")
_sep()

for K in FOLD_FACTORS:
    m = all_metrics.get(K)
    if m is None:
        continue
    mksp_ns = f"{m['makespan_ns']:.1f}" if m['makespan_ns'] else "?"
    tput = f"{m['throughput_mhz']:.1f}" if m['throughput_mhz'] else "?"
    alp = f"{m.get('alp_normalised', 1.0):.3f}" if baseline else "—"
    ae = f"{m['area_efficiency']:.2f}" if m["area_efficiency"] else "?"
    print(f"{K:>3} │ {m['total_luts']:>8,} │ {m['total_ffs']:>8,} │ {m['total_dsps']:>5} │ "
          f"{m['total_brams']:>5} │ {m['makespan']:>9} │ {mksp_ns:>8} │ {m['II']:>3} │ "
          f"{tput:>9} │ {m['batches_in_flight']:>3} │ {alp:>7} │ {ae:>12}")


# ---- TABLE 2: Relative to Baseline ---- #
if baseline:
    _header("TABLE 2: Relative Metrics (vs K=1 Baseline)")

    print(f"{'K':>3} │ {'Area Red.':>9} │ {'Lat. Ratio':>10} │ {'ALP(norm)':>10} │ "
          f"{'DSP Saved':>9} │ {'LUT Saved':>9} │ {'FF Saved':>9}")
    _sep()

    for K in FOLD_FACTORS:
        m = all_metrics.get(K)
        if m is None:
            continue
        lut_saved = baseline["total_luts"] - m["total_luts"]
        ff_saved = baseline["total_ffs"] - m["total_ffs"]
        print(f"{K:>3} │ {m['area_reduction_ratio']:>9.3f} │ {m['latency_ratio']:>10.3f} │ "
              f"{m['alp_normalised']:>10.3f} │ {m['dsp_saved']:>9} │ "
              f"{lut_saved:>+9,} │ {ff_saved:>+9,}")


# ---- TABLE 3: Infrastructure Overhead Breakdown ---- #
_header("TABLE 3: Infrastructure Overhead Breakdown")

print(f"{'K':>3} │ {'Compute':>10} │ {'Buffer':>8} │ {'Buffer':>8} │ {'Buffer':>6} │ "
      f"{'Mux':>6} │ {'Infra %':>7} │ {'#Buf':>5} │ {'#Mux':>5}")
print(f"{'':>3} │ {'LUTs':>10} │ {'LUTs':>8} │ {'FFs':>8} │ {'BRAMs':>6} │ "
      f"{'LUTs':>6} │ {'of Total':>7} │ {'':>5} │ {'':>5}")
_sep()

for K in FOLD_FACTORS:
    m = all_metrics.get(K)
    if m is None:
        continue
    infra_pct = f"{m['buffer_overhead_ratio']*100:.1f}%"
    print(f"{K:>3} │ {m['compute_luts']:>10,} │ {m['buffer_luts']:>8,} │ "
          f"{m['buffer_ffs']:>8,} │ {m['buffer_brams']:>6} │ {m['mux_luts']:>6} │ "
          f"{infra_pct:>7} │ {m['n_buffers']:>5} │ {m['n_muxes']:>5}")


# ---- TABLE 4: Schedule Quality Diagnostics ---- #
_header("TABLE 4: Schedule Quality Diagnostics")

print(f"{'K':>3} │ {'Crit Path':>10} │ {'Sched':>7} │ {'Slack':>6} │ {'Slack':>6} │ "
      f"{'Slack=0':>7} │ {'LT max':>7} │ {'LT mean':>7} │ {'FG Bal':>7}")
print(f"{'':>3} │ {'Util':>10} │ {'Compact':>7} │ {'min':>6} │ {'max':>6} │ "
      f"{'%':>7} │ {'(cyc)':>7} │ {'(cyc)':>7} │ {'':>7}")
_sep()

for K in FOLD_FACTORS:
    m = all_metrics.get(K)
    if m is None:
        continue
    print(f"{K:>3} │ {m['critical_path_utilisation']:>10.3f} │ {m['schedule_compactness']:>7.3f} │ "
          f"{m['slack_min']:>6} │ {m['slack_max']:>6} │ {m['slack_zero_pct']:>6.1f}% │ "
          f"{m['lifetime_max']:>7} │ {m['lifetime_mean']:>7.1f} │ {m['fold_group_balance']:>7.1f}")


# ---- TABLE 5: Per-Op-Type Cost Breakdown (for baseline K=1 and best folded) ---- #
_header("TABLE 5: Per-Op-Type Cost Breakdown")

for K in [1, max(k for k in FOLD_FACTORS if k in all_metrics)]:
    m = all_metrics.get(K)
    if m is None:
        continue
    print(f"\n  K = {K}:")
    print(f"  {'Op Type':>15} │ {'Count':>5} │ {'LUTs':>8} │ {'FFs':>8} │ {'DSPs':>5} │ {'BRAMs':>5} │ {'% of LUTs':>9}")
    _sep(70)
    total_lut = m["total_luts"] or 1
    for op in sorted(m["op_breakdown"].keys()):
        c = m["op_breakdown"][op]
        pct = c["lut"] / total_lut * 100
        print(f"  {op:>15} │ {c['count']:>5} │ {c['lut']:>8,} │ {c['ff']:>8,} │ "
              f"{c['dsp']:>5} │ {c['bram']:>5} │ {pct:>8.1f}%")


# ---- TABLE 6: Throughput Bottleneck Identification ---- #
_header("TABLE 6: Throughput Bottleneck Identification")

for K in FOLD_FACTORS:
    m = all_metrics.get(K)
    if m is None:
        continue
    bottleneck = m.get("throughput_bottleneck", [])
    bn_str = ", ".join(bottleneck) if bottleneck else "(none — II=1, no bottleneck)"
    print(f"  K={K}: II={m['II']}, bottleneck → {bn_str}")


# ---- PARETO ANALYSIS ---- #
_header("PARETO FRONTIER ANALYSIS")

points = [(m["total_luts"], m["makespan"], K) for K, m in all_metrics.items()]
points.sort(key=lambda p: p[0])  # sort by area

# Find Pareto-optimal points (non-dominated in area vs latency)
pareto = []
min_lat = float("inf")
for lut, mksp, K in sorted(points, key=lambda p: p[0]):
    if mksp <= min_lat:
        pareto.append((lut, mksp, K))
        min_lat = mksp

print("  All design points (Area vs Latency):\n")
print(f"  {'K':>3} │ {'LUTs':>10} │ {'Makespan':>10} │ {'Pareto':>7}")
_sep(40)
for lut, mksp, K in points:
    is_pareto = "  ★" if (lut, mksp, K) in pareto else ""
    print(f"  {K:>3} │ {lut:>10,} │ {mksp:>10} │ {is_pareto:>7}")

print(f"\n  Pareto-optimal: K ∈ {{{', '.join(str(p[2]) for p in pareto)}}}")



# ---- SUMMARY ---- #
_header("EVALUATION SUMMARY")

if baseline:
    best_K = max(all_metrics.keys())
    best = all_metrics[best_K]
    print(f"  Baseline (K=1):")
    print(f"    Area:       {baseline['total_luts']:>10,} LUTs, {baseline['total_ffs']:>10,} FFs")
    print(f"    Latency:    {baseline['makespan']:>10} cycles ({baseline['makespan_ns']:.1f} ns)")
    print(f"    Throughput: {baseline['throughput_mhz']:.1f} MHz")
    print()
    print(f"  Most Aggressive (K={best_K}):")
    print(f"    Area:       {best['total_luts']:>10,} LUTs ({best['area_reduction_ratio']:.2f}× reduction)")
    print(f"    Latency:    {best['makespan']:>10} cycles ({best['latency_ratio']:.2f}× slowdown)")
    print(f"    Throughput: {best['throughput_mhz']:.1f} MHz")
    print(f"    ALP (norm): {best['alp_normalised']:.3f} ({'better' if best['alp_normalised'] < 1 else 'worse'} than baseline)")
    print(f"    Infra overhead: {best['buffer_overhead_ratio']*100:.1f}% of total LUTs")
    print()

    # Verdict
    if best["alp_normalised"] < 1.0:
        print(f"  ✅ SCHEDULING IS EFFECTIVE: ALP improved by {(1 - best['alp_normalised'])*100:.1f}%")
    elif best["area_reduction_ratio"] > 1.5:
        print(f"  ⚠️  SCHEDULING TRADES LATENCY FOR AREA: {best['area_reduction_ratio']:.1f}× area reduction")
        print(f"     at {best['latency_ratio']:.1f}× latency cost. Net ALP is {'better' if best['alp_normalised'] < 1 else 'worse'}.")
    else:
        print(f"  ❌ SCHEDULING SHOWS LIMITED BENEFIT at K={best_K}")


# ---- Save JSON for programmatic use ---- #
output_path = HERE / "evaluation_results.json"
serialisable = {}
for K, m in all_metrics.items():
    m_copy = dict(m)
    # Convert defaultdict to dict for JSON
    if "op_breakdown" in m_copy:
        m_copy["op_breakdown"] = {k: dict(v) for k, v in m_copy["op_breakdown"].items()}
    serialisable[str(K)] = m_copy

with open(output_path, "w") as f:
    json.dump(serialisable, f, indent=2, default=str)
print(f"\n  Results saved to: {output_path}")


# --------------------------------------------------------------------------- #
# Matplotlib charts (plot.py is loaded here, after results are persisted, so a
# plotting failure can't trash a successful sweep).
# --------------------------------------------------------------------------- #

try:
    import plot as _plot  # IR/plot.py — added to sys.path at top of file

    plot_dir = HERE / "evaluation_plots"
    saved = _plot.plot_all({int(K): m for K, m in serialisable.items()}, plot_dir)
    print(f"\n  Plots saved to: {plot_dir} ({len(saved)} files)")
    for p in saved:
        print(f"    - {p.name}")
except Exception as e:
    print(f"\n  WARN: plot generation failed: {e}")


print(f"\n{'='*80}")
print(f"  END OF EVALUATION")
print(f"{'='*80}\n")
