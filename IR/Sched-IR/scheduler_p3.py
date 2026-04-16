"""Sched-IR Phase 3 — SCHEDULE (single-batch latency).

Takes a *bound + folded* Sched-IR graph and assigns cycle-accurate timing
to every vertex (``t_ready``, ``t_start``, ``t_end``) and every edge
(``t_produce``, ``t_consume``, ``lifetime``). Computes graph-level
``makespan``, ``initiation_interval``, ``pipeline_depth``, and marks the
``critical_path``.

This is a single-batch model: one input at cycle 0, propagate forward.
No batch-overlap, no resource-conflict modelling.

Usage::

    g_sched = schedule(g_sched)      # mutates in place
"""

from __future__ import annotations

from collections import deque
from heterograph import HGraph


# --------------------------------------------------------------------------- #
# Topological sort (Kahn's algorithm)
# --------------------------------------------------------------------------- #

def _topo_sort(g: HGraph) -> list[int]:
    in_deg = {v: g.num_in_vx(v) for v in g.vertices}
    queue = deque(v for v, d in in_deg.items() if d == 0)
    order: list[int] = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in g.out_vx(v):
            in_deg[u] -= 1
            if in_deg[u] == 0:
                queue.append(u)
    if len(order) != g.num_vx:
        raise ValueError("Sched-IR graph has a cycle — cannot topologically sort")
    return order


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #

def _latency(p: dict) -> int:
    """Pipeline depth L for a vertex, clamped to >= 1."""
    cost = p.get("cost") or {}
    return max(int(cost.get("latency_cycles") or 0), 1)


def _fold_factor(p: dict) -> int:
    return max(int(p.get("fold_factor") or 1), 1)


def _first_output_cycle(p: dict) -> int:
    """Cycle at which the first pipeline result is valid."""
    return int(p["t_start"]) + _latency(p)


def _is_temporalised_reduce(p: dict) -> bool:
    return p.get("op") == "reduce" and p.get("reduce_mode") in ("temporal_accumulate", "hybrid")


def _same_fold_group(pa: dict, pb: dict) -> bool:
    ga = pa.get("fold_group")
    gb = pb.get("fold_group")
    return ga is not None and ga == gb


def _first_output_for_consumer(g: HGraph, pred: int, consumer: int) -> int:
    """When does `pred`'s output become usable by `consumer`?

    * **Same fold group, consumer is a temporalised reduce**: overlap —
      the reduce can start as soon as the first pipeline result arrives
      from the producer.
    * **Same fold group, normal consumer**: first pipeline result. The
      consumer is rate-matched and processes items in lockstep.
    * **Cross-fold (different groups or one is unfolded)**: the consumer
      waits for `pred.t_end` — it needs all K items before it can start.
    * **Producer is unfolded (K=1)**: same as `first_output_cycle`.
    """
    pp = g.pmap[pred]
    pc = g.pmap[consumer]
    pred_K = _fold_factor(pp)

    if pred_K <= 1:
        return _first_output_cycle(pp)

    if _same_fold_group(pp, pc):
        return _first_output_cycle(pp)

    if _is_temporalised_reduce(pc):
        return _first_output_cycle(pp)

    # Cross-fold: consumer needs the full stream.
    return int(pp["t_end"])


# --------------------------------------------------------------------------- #
# Critical-path extraction
# --------------------------------------------------------------------------- #

def _find_critical_path(g: HGraph, order: list[int]) -> list[int]:
    """Backward walk from the vertex with max t_end."""
    if not order:
        return []
    last = max(order, key=lambda v: int(g.pmap[v]["t_end"]))
    path = [last]
    while True:
        preds = g.in_vx(path[-1])
        if not preds:
            break
        best = max(preds, key=lambda v: int(g.pmap[v]["t_end"]))
        path.append(best)
    path.reverse()
    return path


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def schedule(g_sched: HGraph) -> HGraph:
    """Phase 3 — assign single-batch timing to every vertex and edge.

    Mutates ``g_sched`` in place and returns it.
    """
    order = _topo_sort(g_sched)

    # ---- vertex timing ------------------------------------------------ #
    for vx in order:
        p = g_sched.pmap[vx]
        L = _latency(p)
        K = _fold_factor(p)

        preds = g_sched.in_vx(vx)
        if not preds:
            t_ready = 0
        else:
            t_ready = max(_first_output_for_consumer(g_sched, pred, vx) for pred in preds)

        p["t_ready"] = t_ready
        p["t_start"] = t_ready
        p["t_end"]   = t_ready + L + max(K - 1, 0)

    # ---- edge timing -------------------------------------------------- #
    for u, v in g_sched.edges:
        pu = g_sched.pmap[u]
        pv = g_sched.pmap[v]
        ep = g_sched.pmap[(u, v)]
        ep["t_produce"] = _first_output_cycle(pu)
        ep["t_consume"] = int(pv["t_start"])
        ep["lifetime"]  = max(ep["t_consume"] - ep["t_produce"], 0)

    # ---- graph-level -------------------------------------------------- #
    makespan = max(int(g_sched.pmap[v]["t_end"]) for v in g_sched.vertices)
    g_sched.pmap["makespan"] = makespan

    fold_plan = g_sched.pmap.get("fold_plan") or []
    graph_ii = max((entry.get("factor", 1) for entry in fold_plan), default=1)
    g_sched.pmap["initiation_interval"] = int(graph_ii)

    crit = _find_critical_path(g_sched, order)
    g_sched.pmap["critical_path"] = crit
    g_sched.pmap["pipeline_depth"] = sum(_latency(g_sched.pmap[v]) for v in crit)

    for vx in g_sched.vertices:
        g_sched.pmap[vx]["critical_path"] = vx in crit

    _validate_schedule(g_sched)
    return g_sched


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _validate_schedule(g: HGraph) -> None:
    for vx in g.vertices:
        p = g.pmap[vx]
        if p.get("t_start") is None or p.get("t_end") is None:
            raise ValueError(f"SCHEDULE left vertex {vx} without timing")
        if p["t_end"] < p["t_start"] + 1:
            raise ValueError(f"SCHEDULE vertex {vx}: t_end < t_start + 1")
        if p["t_start"] < p.get("t_ready", 0):
            raise ValueError(f"SCHEDULE vertex {vx}: t_start < t_ready")

    for u, v in g.edges:
        ep = g.pmap[(u, v)]
        if ep.get("t_produce") is None or ep.get("t_consume") is None:
            raise ValueError(f"SCHEDULE left edge ({u},{v}) without timing")
        if ep["lifetime"] < 0:
            raise ValueError(f"SCHEDULE edge ({u},{v}): negative lifetime")
