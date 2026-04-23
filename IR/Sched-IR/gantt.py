"""Cycle-accurate Gantt renderer for a scheduled Sched-IR graph.

Produces a raw SVG that the heterograph WebView can embed alongside the
normal graph views, showing

* one horizontal bar per vertex, spanning ``[t_start, t_end]``,
* a dashed ghost bar for the next batch, offset by the graph II (so the
  N–P–T pipelining is visible at a glance),
* a red dashed arrow for each edge whose lifetime > 0 (inserted buffers
  at Phase 4),
* N–P–T info (N, P, T) embedded in the bar for folded vertices with room.

Usage::

    from gantt import GanttWrapper
    wv.add_graph(GanttWrapper(g_sched), title="Gantt K=4")
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


# --------------------------------------------------------------------------- #
# Sibling styling module — SCHED_COLORS
# --------------------------------------------------------------------------- #

def _load_sibling(name: str):
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(f"_sched_ir_{name}", here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_styling = _load_sibling("styling")
SCHED_COLORS = _styling.SCHED_COLORS


# --------------------------------------------------------------------------- #
# Public wrapper
# --------------------------------------------------------------------------- #

class GanttWrapper:
    """Thin wrapper so ``WebView.add_graph`` can render a Gantt SVG."""

    def __init__(self, g_sched):
        self._svg = render_gantt_svg(g_sched)
        self.style = {}

    def render(self, *, format="svg", pipe=False, **kwargs):
        return self._svg


# --------------------------------------------------------------------------- #
# SVG generation
# --------------------------------------------------------------------------- #

def render_gantt_svg(gx) -> bytes:
    """Render a cycle-accurate Gantt chart as raw SVG bytes."""
    makespan = int(gx.pmap.get("makespan") or 1)
    crit_set = set(gx.pmap.get("critical_path") or [])

    # Collect rows in topological order (source → sink).
    rows = []
    for v in gx.vertices:
        p = gx.pmap[v]
        rows.append({
            "vx":      v,
            "name":    p.get("nn_layer_name") or "?",
            "op":      p.get("op") or "?",
            "t_start": int(p.get("t_start") or 0),
            "t_end":   int(p.get("t_end") or 1),
            "N":       p.get("parallelism_N"),
            "P":       p.get("lanes_P"),
            "T":       int(p.get("temporal_steps_T") or p.get("fold_factor") or 1),
            "L":       p.get("pipeline_latency_L"),
            "ii":      p.get("ii"),
            "crit":    v in crit_set,
        })

    # Collect buffer edges (lifetime > 0).
    buf_edges = []
    for u, v in gx.edges:
        ep = gx.pmap[(u, v)]
        lt = ep.get("lifetime", 0)
        if lt and lt > 0:
            buf_edges.append({
                "src":         u, "dst": v,
                "t_produce":   int(ep.get("t_produce", 0)),
                "t_consume":   int(ep.get("t_consume", 0)),
                "t_producer":  int(ep.get("t_producer", ep.get("t_produce", 0))),
                "t_consumer":  int(ep.get("t_consumer", ep.get("t_consume", 0))),
                "lifetime":    int(lt),
            })

    # Layout constants
    PX_PER_CYCLE = max(18, min(30, 900 // max(makespan, 1)))
    ROW_H    = 36
    LABEL_W  = 240
    PAD      = 20
    HEADER_H = 30

    graph_ii_pre = int(gx.pmap.get("initiation_interval") or 1)
    chart_cycles = makespan + (graph_ii_pre if graph_ii_pre < makespan else 0) + 2
    chart_w  = chart_cycles * PX_PER_CYCLE
    total_w  = LABEL_W + chart_w + 2 * PAD
    total_h  = HEADER_H + len(rows) * ROW_H + 2 * PAD

    vx_to_row = {r["vx"]: i for i, r in enumerate(rows)}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}pt" height="{total_h}pt" '
        f'viewBox="0 0 {total_w} {total_h}" '
        f'font-family="monospace" font-size="11">',
        # Arrow marker for buffer edges.
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#E53935"/></marker></defs>',
        '<g id="graph0" class="graph" transform="translate(0,0)">',
        f'<rect width="{total_w}" height="{total_h}" fill="#1e1e1e"/>',
    ]

    x0 = LABEL_W + PAD
    y0 = HEADER_H + PAD

    # Cycle grid lines.
    for c in range(0, makespan + 1, max(1, makespan // 20)):
        x = x0 + c * PX_PER_CYCLE
        parts.append(
            f'<line x1="{x}" y1="{y0 - 5}" x2="{x}" y2="{y0 + len(rows) * ROW_H}" '
            f'stroke="#444" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x}" y="{y0 - 8}" fill="#aaa" font-size="9" '
            f'text-anchor="middle">{c}</text>'
        )

    # Rows.
    for i, r in enumerate(rows):
        y  = y0 + i * ROW_H
        bw = (r["t_end"] - r["t_start"]) * PX_PER_CYCLE
        bx = x0 + r["t_start"] * PX_PER_CYCLE

        color  = SCHED_COLORS.get(r["op"], "#888")
        border = "#E53935" if r["crit"] else "#555"
        bw_px  = max(bw, 2)

        # Label (with T inline when folded).
        name_tag = r["name"]
        if r["T"] > 1 and r["N"] and r["P"]:
            name_tag += f"  N={r['N']} P={r['P']} T={r['T']}"
        parts.append(
            f'<text x="{LABEL_W}" y="{y + ROW_H // 2 + 4}" fill="#ccc" '
            f'text-anchor="end" font-size="10">{name_tag}</text>'
        )

        # Bar.
        parts.append(
            f'<rect x="{bx}" y="{y + 4}" width="{bw_px}" height="{ROW_H - 8}" '
            f'rx="3" fill="{color}" stroke="{border}" '
            f'stroke-width="{"2" if r["crit"] else "1"}"/>'
        )

        # Timing text inside the bar (if wide enough).
        label = f'{r["t_start"]}..{r["t_end"]}'
        if r["T"] > 1:
            label += f'  T={r["T"]}'
        if bw_px > 50:
            parts.append(
                f'<text x="{bx + bw_px // 2}" y="{y + ROW_H // 2 + 4}" '
                f'fill="white" font-size="9" text-anchor="middle">{label}</text>'
            )

    # Ghost bars — second batch offset by graph II to visualise pipelining.
    graph_ii = int(gx.pmap.get("initiation_interval") or 1)
    if graph_ii > 0 and graph_ii < makespan:
        for i, r in enumerate(rows):
            y  = y0 + i * ROW_H
            bw = (r["t_end"] - r["t_start"]) * PX_PER_CYCLE
            bx = x0 + (r["t_start"] + graph_ii) * PX_PER_CYCLE
            color = SCHED_COLORS.get(r["op"], "#888")
            bw_px = max(bw, 2)
            if bx + bw_px <= x0 + chart_w + 40:
                parts.append(
                    f'<rect x="{bx}" y="{y + 4}" width="{bw_px}" height="{ROW_H - 8}" '
                    f'rx="3" fill="{color}" fill-opacity="0.3" stroke="#888" '
                    f'stroke-width="0.5" stroke-dasharray="3,2"/>'
                )

    # Buffer arrows.
    for be in buf_edges:
        ri = vx_to_row.get(be["src"])
        rj = vx_to_row.get(be["dst"])
        if ri is None or rj is None:
            continue
        x1 = x0 + be["t_producer"] * PX_PER_CYCLE
        y1 = y0 + ri * ROW_H + ROW_H // 2
        x2 = x0 + be["t_consumer"] * PX_PER_CYCLE
        y2 = y0 + rj * ROW_H + ROW_H // 2
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#E53935" stroke-width="1.5" stroke-dasharray="4,3" '
            f'marker-end="url(#arr)"/>'
        )

    # Title / legend.
    ms  = gx.pmap.get("makespan", "?")
    ii  = gx.pmap.get("initiation_interval", "?")
    tp  = gx.pmap.get("sustained_throughput_hz")
    bif = gx.pmap.get("batches_in_flight", "?")
    tp_s = f", throughput={tp/1e6:.0f} MHz" if tp else ""
    parts.append(
        f'<text x="{PAD}" y="16" fill="#eee" font-size="12" font-weight="bold">'
        f'Gantt — makespan={ms} cyc, II={ii}, in-flight={bif}{tp_s}'
        f'</text>'
    )

    parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts).encode("utf-8")
