"""Sched-IR Phase 4 — INSERT INFRASTRUCTURE + Phase 5 — COST ROLL-UP.

Walks every data edge with ``lifetime > 0`` and inserts an explicit
``buffer`` vertex between producer and consumer. Buffer type is chosen
from the resource YAML:

* ``register_buffer`` — flip-flop based (depth × width < 36 864 bits).
* ``bram_buffer``     — block RAM (depth × width ≥ 36 864 bits).

Each inserted buffer vertex is bound to its kernel and costed via the
closed-form queries already in ``kernels.py``.

Phase 5 (COST ROLL-UP) is bundled in: after buffer insertion, the total
area is summed across all vertices (including buffers) and written to the
graph-level pmap fields ``total_luts``, ``total_ffs``, ``total_dsps``,
``total_brams``.

Usage::

    g_sched = insert_buffers(g_sched)   # mutates in place
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import yaml

from heterograph import HGraph


# --------------------------------------------------------------------------- #
# Sibling-module load
# --------------------------------------------------------------------------- #

def _load_sibling(name: str):
    here = Path(__file__).resolve().parent
    full_name = f"_sched_ir_{name}"
    spec = importlib.util.spec_from_file_location(full_name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


_kernels = _load_sibling("kernels")
_schema  = _load_sibling("schema")
vinit_sched = _schema.vinit_sched


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BRAM_BITS = 36_864        # 36 Kb per BRAM block


# --------------------------------------------------------------------------- #
# Buffer sizing
# --------------------------------------------------------------------------- #

def _edge_width_bits(ep: dict) -> int:
    """Total bit-width of data on an edge (prod of non-batch dims × bitwidth)."""
    shape = ep.get("tensor_shape")
    bw = ep.get("bitwidth")
    if shape is None or bw is None:
        return 0
    # Drop symbolic batch dim (None) and multiply the rest.
    spatial = [d for d in shape if d is not None]
    if not spatial:
        return 0
    vol = 1
    for d in spatial:
        vol *= int(d)
    return int(vol * math.ceil(float(bw)))


def _choose_buffer_kernel(width_bits: int, depth: int) -> str:
    total = width_bits * depth
    if total >= BRAM_BITS:
        return "bram_buffer"
    return "register_buffer"


# --------------------------------------------------------------------------- #
# Mux detection (stub — fires for future topologies, not JEDI-linear)
# --------------------------------------------------------------------------- #

def _needs_mux(g: HGraph, vx: int) -> bool:
    """True if a folded vertex receives data from distinct sources across
    fold iterations and therefore needs an input multiplexer.

    In JEDI-linear every folded vertex is fed by a single predecessor along
    the fold axis, so this always returns False. Real mux insertion becomes
    relevant for attention-style topologies with multiple heads sharing a
    compute block.
    """
    p = g.pmap[vx]
    if (p.get("fold_factor") or 1) <= 1:
        return False
    # TODO: detect heterogeneous input sources across fold iterations.
    return False


# --------------------------------------------------------------------------- #
# Buffer insertion
# --------------------------------------------------------------------------- #

def _insert_buffer(
    g: HGraph,
    u: int,
    v: int,
    ep: dict,
    fpga: dict,
) -> int:
    """Replace edge (u, v) with u → buf → v and return the buffer vertex id."""
    depth = int(ep.get("lifetime") or 1)
    width = _edge_width_bits(ep)
    kernel_name = _choose_buffer_kernel(width, depth)

    # Create buffer vertex.
    buf = g.add_vx()
    bp = g.pmap[buf]
    bp["nn_layer_idx"]      = None
    bp["nn_layer_name"]     = f"buf_{g.pmap[u].get('nn_layer_name', u)}_{g.pmap[v].get('nn_layer_name', v)}"
    bp["nn_op_kind"]        = None
    bp["decomp_index"]      = None
    bp["inserted_by"]       = "scheduler"
    bp["op"]                = "buffer"
    bp["op_params"]         = {
        "width_bits": width,
        "depth":      depth,
        "total_bits": width * depth,
    }
    bp["kernel_type"]       = kernel_name
    bp["kernel_instance"]   = 0
    bp["fold_factor"]       = 1
    bp["fold_group"]        = None
    bp["physical_instances"] = 1

    # Cost via the closed-form query.
    cost_fn = _kernels.REGISTRY[kernel_name + "_cost"]
    bp["cost"] = cost_fn(bp, _kernels.WeightProvider(None), fpga)

    # Timing: the buffer sits in the gap — it doesn't extend the schedule.
    bp["t_start"]       = int(g.pmap[u].get("t_end") or 0)
    bp["t_ready"]       = bp["t_start"]
    buf_lat             = int(bp["cost"].get("latency_cycles") or 1)
    bp["t_end"]         = bp["t_start"] + buf_lat
    bp["critical_path"] = False

    # Rewire: remove (u, v), add (u, buf) and (buf, v).
    # Copy edge properties to both new edges.
    g.rm_edge((u, v))

    created_ub = g.add_edge(u, buf)
    if created_ub:
        e_ub = created_ub[0]
        g.pmap[e_ub]["tensor_shape"] = ep.get("tensor_shape")
        g.pmap[e_ub]["bitwidth"]     = ep.get("bitwidth")
        g.pmap[e_ub]["volume_bits"]  = ep.get("volume_bits")
        g.pmap[e_ub]["edge_kind"]    = "data"
        g.pmap[e_ub]["t_produce"]    = ep.get("t_produce")
        g.pmap[e_ub]["t_consume"]    = bp["t_start"]
        g.pmap[e_ub]["t_producer"]   = ep.get("t_producer", ep.get("t_produce"))
        g.pmap[e_ub]["t_consumer"]   = bp["t_start"]
        g.pmap[e_ub]["lifetime"]     = 0   # producer → buffer is immediate

    created_bv = g.add_edge(buf, v)
    if created_bv:
        e_bv = created_bv[0]
        g.pmap[e_bv]["tensor_shape"] = ep.get("tensor_shape")
        g.pmap[e_bv]["bitwidth"]     = ep.get("bitwidth")
        g.pmap[e_bv]["volume_bits"]  = ep.get("volume_bits")
        g.pmap[e_bv]["edge_kind"]    = "data"
        g.pmap[e_bv]["t_produce"]    = bp["t_end"]
        g.pmap[e_bv]["t_consume"]    = ep.get("t_consume")
        g.pmap[e_bv]["t_producer"]   = bp["t_end"]
        g.pmap[e_bv]["t_consumer"]   = ep.get("t_consumer", ep.get("t_consume"))
        g.pmap[e_bv]["lifetime"]     = 0   # buffer → consumer is immediate

    return buf


# --------------------------------------------------------------------------- #
# Phase 5 — cost roll-up
# --------------------------------------------------------------------------- #

def _rollup(g: HGraph) -> None:
    totals = {"total_luts": 0, "total_ffs": 0, "total_dsps": 0, "total_brams": 0}
    key_map = {"lut": "total_luts", "ff": "total_ffs", "dsp": "total_dsps", "bram": "total_brams"}

    for vx in g.vertices:
        p = g.pmap[vx]
        cost = p.get("cost") or {}
        inst = int(p.get("physical_instances") or 1)
        for ck, gk in key_map.items():
            totals[gk] += int(cost.get(ck) or 0) * inst

    for k, v in totals.items():
        g.pmap[k] = v


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def insert_buffers(g_sched: HGraph) -> HGraph:
    """Phase 4 + 5 — insert buffer vertices on edges with lifetime > 0,
    then roll up total area.

    Mutates ``g_sched`` in place and returns it.
    """
    # Load fpga config from the resource YAML stamped on the graph.
    yaml_path = g_sched.pmap.get("resource_yaml")
    cfg = yaml.safe_load(Path(yaml_path).read_text()) if yaml_path else {}
    fpga = cfg.get("fpga") or {}

    # Collect edges that need buffers BEFORE mutating the graph (we can't
    # iterate edges while inserting / removing).
    to_buffer: list[tuple[int, int, dict]] = []
    for u, v in g_sched.edges:
        ep = g_sched.pmap[(u, v)]
        lt = ep.get("lifetime") or 0
        if lt > 0:
            to_buffer.append((u, v, dict(ep)))

    # Insert buffers.
    inserted: list[int] = []
    for u, v, ep_copy in to_buffer:
        buf_vx = _insert_buffer(g_sched, u, v, ep_copy, fpga)
        inserted.append(buf_vx)

    # Mux stub — detect and insert muxes if needed (no-op for JEDI).
    for vx in list(g_sched.vertices):
        if _needs_mux(g_sched, vx):
            pass  # TODO: insert mux vertex

    # Phase 5 — roll up area.
    _rollup(g_sched)

    _validate_infra(g_sched, inserted)
    return g_sched


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _validate_infra(g: HGraph, inserted: list[int]) -> None:
    for buf in inserted:
        p = g.pmap[buf]
        if p.get("op") != "buffer":
            raise ValueError(f"inserted vertex {buf} is not a buffer")
        if p.get("kernel_type") is None:
            raise ValueError(f"buffer vertex {buf} has no kernel_type")
        cost = p.get("cost") or {}
        if not cost:
            raise ValueError(f"buffer vertex {buf} has no cost")

    # No edge should have lifetime > 0 after insertion (all were replaced).
    for u, v in g.edges:
        ep = g.pmap[(u, v)]
        lt = ep.get("lifetime") or 0
        if lt > 0:
            raise ValueError(
                f"edge ({u}, {v}) still has lifetime={lt} after buffer insertion"
            )
