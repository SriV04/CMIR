"""CMIR Evaluation Plots — matplotlib charts derived from evaluation_results.json.

Importable from notebooks without triggering the heavy model build in
`evaluate.py`. Each function accepts a `results: dict[int, dict]` (mapping K
to its metric dict) and an optional `ax` for compositing.

Usage from a notebook:

    from plot import load_results, plot_dashboard, plot_pareto
    results = load_results("IR/evaluation_results.json")
    plot_dashboard(results)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Style + palettes
# --------------------------------------------------------------------------- #

_RC = {
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

_K_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

OP_COLORS = {
    "dense":       "#1f77b4",
    "reduce":      "#ff7f0e",
    "elementwise": "#2ca02c",
    "buffer":      "#d62728",
    "mux":         "#9467bd",
    "unattributed": "#7f7f7f",
}


def _apply_style():
    plt.rcParams.update(_RC)


def _ks(results) -> list[int]:
    return sorted(results.keys())


def _k_color(K: int, all_ks: list[int]) -> str:
    return _K_PALETTE[all_ks.index(K) % len(_K_PALETTE)]


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

def load_results(path) -> dict[int, dict]:
    """Load `evaluation_results.json` with int K keys."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# Individual plots
# --------------------------------------------------------------------------- #

def plot_pareto(results, ax=None):
    """Area (LUTs) vs Latency (ns) Pareto frontier.

    Each design point is labeled with its K and (★) if Pareto-optimal. The
    frontier is connected with a dashed line so the trade-off curve is
    visible at a glance.
    """
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ks = _ks(results)
    luts = np.array([results[K]["total_luts"] for K in ks], dtype=float)
    lats = np.array([results[K]["makespan_ns"] for K in ks], dtype=float)

    pareto = []
    for i, (l, m) in enumerate(zip(luts, lats)):
        dominated = any(
            (lo <= l) and (mo <= m) and ((lo, mo) != (l, m))
            for lo, mo in zip(luts, lats)
        )
        pareto.append(not dominated)

    colors = [_k_color(K, ks) for K in ks]
    ax.scatter(luts, lats, s=220, c=colors, edgecolor="black",
               linewidth=1.5, zorder=3)

    for K, l, m, is_p in zip(ks, luts, lats, pareto):
        star = " ★" if is_p else ""
        ax.annotate(f"K={K}{star}", (l, m), fontsize=11, fontweight="bold",
                    xytext=(12, 4), textcoords="offset points", va="center")

    pareto_pts = sorted(
        [(l, m) for l, m, p in zip(luts, lats, pareto) if p]
    )
    if len(pareto_pts) > 1:
        ax.plot([p[0] for p in pareto_pts], [p[1] for p in pareto_pts],
                "k--", alpha=0.5, lw=1.5, zorder=1, label="Pareto frontier")
        ax.legend(loc="upper right")

    ax.set_xscale("log")
    ax.set_xlabel("Area (LUTs, log scale)")
    ax.set_ylabel("Latency (ns)")
    ax.set_title("Area vs Latency — Pareto Frontier")
    return ax


def plot_alp_comparison(results, ax=None):
    """Normalised area-latency product (ALP) per K.

    ALP = LUTs × makespan, normalised so K=1 = 1.0. Bars below the red
    baseline line represent net improvement vs the unfolded design.
    """
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ks = _ks(results)
    alps = [results[K].get("alp_normalised", 1.0) for K in ks]
    colors = [_k_color(K, ks) for K in ks]

    bars = ax.bar([str(K) for K in ks], alps, color=colors,
                  edgecolor="black", linewidth=1.0)
    ax.axhline(1.0, color="#d62728", linestyle="--", lw=1.5, alpha=0.7,
               label="K=1 baseline")

    for bar, val in zip(bars, alps):
        gain_pct = (1 - val) * 100
        label = f"{val:.3f}\n({gain_pct:+.0f}%)" if val < 1 else f"{val:.3f}"
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Fold factor K")
    ax.set_ylabel("ALP (normalised, lower = better)")
    ax.set_title("Area-Latency Product vs K")
    ax.set_ylim(0, max(1.15, max(alps) * 1.15))
    ax.legend(loc="upper right")
    return ax


def plot_resource_scaling(results, ax=None):
    """LUTs and latency vs K on twin axes — shows the trade-off curve directly."""
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ks = _ks(results)
    xs = [str(K) for K in ks]
    luts = [results[K]["total_luts"] for K in ks]
    lats = [results[K]["makespan_ns"] for K in ks]

    color_lut = "#1f77b4"
    color_lat = "#d62728"

    line1 = ax.plot(xs, luts, "o-", color=color_lut, lw=2, ms=10,
                    markeredgecolor="black", markeredgewidth=1.0,
                    label="LUTs")
    ax.set_yscale("log")
    ax.set_xlabel("Fold factor K")
    ax.set_ylabel("LUTs (log scale)", color=color_lut)
    ax.tick_params(axis="y", labelcolor=color_lut)
    ax.grid(True, which="both", alpha=0.3)

    for K, l in zip(ks, luts):
        ax.annotate(f"{l:,}", (str(K), l), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=color_lut)

    ax2 = ax.twinx()
    line2 = ax2.plot(xs, lats, "s-", color=color_lat, lw=2, ms=10,
                     markeredgecolor="black", markeredgewidth=1.0,
                     label="Latency (ns)")
    ax2.set_ylabel("Latency (ns)", color=color_lat)
    ax2.tick_params(axis="y", labelcolor=color_lat)
    ax2.grid(False)

    for K, m in zip(ks, lats):
        ax2.annotate(f"{m:.0f}", (str(K), m), xytext=(0, -14),
                     textcoords="offset points", ha="center",
                     fontsize=8, color=color_lat)

    lines = line1 + line2
    ax.legend(lines, [l.get_label() for l in lines], loc="upper center")
    ax.set_title("Resource & Latency Scaling vs K")
    return ax


def plot_throughput_efficiency(results, ax=None):
    """Throughput (MHz) and area efficiency (inferences/sec/LUT) vs K."""
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ks = _ks(results)
    xs = [str(K) for K in ks]
    tput = [results[K].get("throughput_mhz") or 0 for K in ks]
    ae = [results[K].get("area_efficiency") or 0 for K in ks]

    color_tput = "#2ca02c"
    color_ae = "#9467bd"

    line1 = ax.plot(xs, tput, "o-", color=color_tput, lw=2, ms=10,
                    markeredgecolor="black", markeredgewidth=1.0,
                    label="Throughput (MHz)")
    ax.set_xlabel("Fold factor K")
    ax.set_ylabel("Throughput (MHz)", color=color_tput)
    ax.tick_params(axis="y", labelcolor=color_tput)

    for K, t in zip(ks, tput):
        ax.annotate(f"{t:.0f}", (str(K), t), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=color_tput)

    ax2 = ax.twinx()
    line2 = ax2.plot(xs, ae, "s-", color=color_ae, lw=2, ms=10,
                     markeredgecolor="black", markeredgewidth=1.0,
                     label="Area efficiency")
    ax2.set_ylabel("Inferences / sec / LUT", color=color_ae)
    ax2.tick_params(axis="y", labelcolor=color_ae)
    ax2.grid(False)

    best = max(range(len(ae)), key=lambda i: ae[i])
    ax2.annotate(f"peak\nK={ks[best]}",
                 (str(ks[best]), ae[best]), xytext=(8, -4),
                 textcoords="offset points", fontsize=9,
                 fontweight="bold", color=color_ae)

    lines = line1 + line2
    ax.legend(lines, [l.get_label() for l in lines], loc="best")
    ax.set_title("Throughput & Area Efficiency vs K")
    return ax


def plot_lut_composition(results, ax=None, percent=True):
    """Stacked bar of LUT cost by op type for each K.

    `percent=True` normalises each bar to 100% so composition is comparable
    across K. `percent=False` shows absolute LUT counts.

    For K=1, total_luts comes from da4ml's end-to-end ground truth and may
    exceed the per-op rollup; the residual is shown as 'unattributed'.
    """
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ks = _ks(results)
    op_order = ["dense", "reduce", "elementwise", "mux", "buffer", "unattributed"]

    # Collect per-K, per-op LUT counts
    rows = []
    for K in ks:
        m = results[K]
        ob = m.get("op_breakdown", {})
        row = {op: int(ob.get(op, {}).get("lut", 0)) for op in op_order}
        # Reconcile rollup vs ground-truth total
        rollup_sum = sum(row.values())
        total = int(m.get("total_luts", rollup_sum))
        if total > rollup_sum:
            row["unattributed"] = total - rollup_sum
        rows.append(row)

    # Drop op categories that are zero everywhere
    active_ops = [op for op in op_order if any(r[op] > 0 for r in rows)]

    xs = np.arange(len(ks))
    bottoms = np.zeros(len(ks))

    for op in active_ops:
        vals = np.array([r[op] for r in rows], dtype=float)
        if percent:
            totals = np.array([sum(r.values()) for r in rows], dtype=float)
            vals = np.where(totals > 0, vals / totals * 100, 0)
        ax.bar(xs, vals, bottom=bottoms, label=op,
               color=OP_COLORS.get(op, "#cccccc"),
               edgecolor="black", linewidth=0.7)
        # Label segments large enough to fit text
        for i, v in enumerate(vals):
            if v > (5 if percent else max(vals.max() * 0.05, 1)):
                txt = f"{v:.0f}%" if percent else f"{int(v):,}"
                ax.text(xs[i], bottoms[i] + v / 2, txt,
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(xs)
    ax.set_xticklabels([f"K={K}" for K in ks])
    ax.set_xlabel("Fold factor")
    if percent:
        ax.set_ylabel("Share of LUTs (%)")
        ax.set_ylim(0, 105)
        ax.set_title("LUT Composition by Op Type (% of total)")
    else:
        ax.set_ylabel("LUTs")
        ax.set_yscale("log")
        ax.set_title("LUT Cost by Op Type")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_ff_growth(results, ax=None):
    """Buffer FFs vs compute FFs as K grows — surfaces the cost of folding.

    Buffer FFs hold inter-stage operands across the longer schedules that
    folding creates; this chart shows them dominating at high K.
    """
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ks = _ks(results)
    xs = np.arange(len(ks))
    compute_ffs = np.array([results[K].get("compute_ffs", 0) for K in ks])
    buffer_ffs = np.array([results[K].get("buffer_ffs", 0) for K in ks])
    totals = np.array([results[K]["total_ffs"] for K in ks])
    residual = np.maximum(totals - compute_ffs - buffer_ffs, 0)

    ax.bar(xs, compute_ffs, label="Compute FFs", color="#1f77b4",
           edgecolor="black", linewidth=0.7)
    ax.bar(xs, buffer_ffs, bottom=compute_ffs, label="Buffer FFs",
           color="#d62728", edgecolor="black", linewidth=0.7)
    if residual.sum() > 0:
        ax.bar(xs, residual, bottom=compute_ffs + buffer_ffs,
               label="DA pipeline FFs", color="#7f7f7f",
               edgecolor="black", linewidth=0.7, hatch="//")

    for i, total in enumerate(totals):
        ax.annotate(f"{int(total):,}", (xs[i], total),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"K={K}" for K in ks])
    ax.set_xlabel("Fold factor")
    ax.set_ylabel("Flip-flops")
    ax.set_title("FF Cost: Compute vs Buffer vs Pipeline")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_schedule_quality(results, ax=None):
    """Critical-path utilisation, schedule compactness, and slack=0 share.

    These are the diagnostic metrics from the Sched-IR rollup. Critical-path
    utilisation drops as K grows because the schedule has more idle slots
    inside each II window.
    """
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ks = _ks(results)
    xs = np.arange(len(ks))
    width = 0.35

    util = [results[K].get("critical_path_utilisation", 0) for K in ks]
    compact = [results[K].get("schedule_compactness", 0) for K in ks]

    bars1 = ax.bar(xs - width / 2, util, width, label="Critical-path utilisation",
                   color="#1f77b4", edgecolor="black", linewidth=0.7)
    bars2 = ax.bar(xs + width / 2, compact, width, label="Schedule compactness",
                   color="#ff7f0e", edgecolor="black", linewidth=0.7)

    for bars, vals in ((bars1, util), (bars2, compact)):
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}",
                        (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"K={K}" for K in ks])
    ax.set_xlabel("Fold factor")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.1)
    ax.set_title("Schedule Quality Diagnostics")
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_op_heatmap(results, ax=None):
    """Heatmap of LUT cost by op type × K. Log-scale colour."""
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ks = _ks(results)
    op_order = ["dense", "reduce", "elementwise", "buffer", "mux"]

    matrix = np.zeros((len(op_order), len(ks)))
    for j, K in enumerate(ks):
        ob = results[K].get("op_breakdown", {})
        for i, op in enumerate(op_order):
            matrix[i, j] = int(ob.get(op, {}).get("lut", 0))

    # Drop rows that are zero across all K
    nonzero = matrix.sum(axis=1) > 0
    matrix = matrix[nonzero]
    labels = [op for op, keep in zip(op_order, nonzero) if keep]

    # Log-scale colour mapping
    plot_matrix = np.where(matrix > 0, np.log10(matrix + 1), 0)
    im = ax.imshow(plot_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"K={K}" for K in ks])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("LUT Cost Heatmap (Op × K)")

    # Annotate with raw counts
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = int(matrix[i, j])
            if v == 0:
                txt = "—"
            elif v >= 1000:
                txt = f"{v / 1000:.1f}k"
            else:
                txt = str(v)
            # White text only on the darkest cells (top ~20% of log range);
            # everything else gets black for legibility on the YlOrRd light
            # end.
            cell_norm = plot_matrix[i, j] / plot_matrix.max() if plot_matrix.max() > 0 else 0
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if cell_norm > 0.85 else "black")

    cbar = plt.colorbar(im, ax=ax, label="log10(LUTs+1)")
    cbar.ax.tick_params(labelsize=8)
    ax.grid(False)
    return ax


# --------------------------------------------------------------------------- #
# Composite dashboard
# --------------------------------------------------------------------------- #

def plot_dashboard(results, output_path=None, figsize=(18, 11)):
    """Combined 2×4 dashboard with all eight charts."""
    _apply_style()
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    plot_pareto(results,                axes[0, 0])
    plot_alp_comparison(results,        axes[0, 1])
    plot_resource_scaling(results,      axes[0, 2])
    plot_throughput_efficiency(results, axes[0, 3])
    plot_lut_composition(results,       axes[1, 0], percent=True)
    plot_ff_growth(results,             axes[1, 1])
    plot_schedule_quality(results,      axes[1, 2])
    plot_op_heatmap(results,            axes[1, 3])

    fig.suptitle("CMIR Fold-Factor Sweep — JEDI-linear GNN",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    return fig


# --------------------------------------------------------------------------- #
# Save-all helper
# --------------------------------------------------------------------------- #

_PLOTS = {
    "pareto":               plot_pareto,
    "alp":                  plot_alp_comparison,
    "resource_scaling":     plot_resource_scaling,
    "throughput_efficiency": plot_throughput_efficiency,
    "lut_composition_pct":  lambda r, ax=None: plot_lut_composition(r, ax, percent=True),
    "ff_growth":            plot_ff_growth,
    "schedule_quality":     plot_schedule_quality,
    "op_heatmap":           plot_op_heatmap,
}


def plot_all(results, output_dir):
    """Save every individual plot + the combined dashboard to `output_dir`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, fn in _PLOTS.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        fn(results, ax)
        fig.tight_layout()
        path = output_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    dashboard_path = output_dir / "dashboard.png"
    fig = plot_dashboard(results, output_path=dashboard_path)
    plt.close(fig)
    saved.append(dashboard_path)
    return saved


# --------------------------------------------------------------------------- #
# CLI: `python plot.py [results.json] [out_dir]`
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys
    here = Path(__file__).resolve().parent
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "evaluation_results.json"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "evaluation_plots"
    results = load_results(json_path)
    paths = plot_all(results, out_dir)
    print(f"Saved {len(paths)} plot(s) to {out_dir}")
    for p in paths:
        print(f"  - {p.name}")
