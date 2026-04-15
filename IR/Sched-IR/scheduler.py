"""Sched-IR scheduler — Phase 1 (BIND).

Reads the resource YAML, walks an unscheduled Sched-IR graph (produced by
``decomposer.py``), and for every vertex:

* picks the first kernel that claims the vertex's primitive and whose
  constraints are satisfied by ``op_params``;
* writes ``kernel_type`` + a fresh ``kernel_instance`` id;
* runs the kernel's ``cost_query`` (against the original Keras model for
  weight access) and stores the canonical
  ``cost = {lut, ff, dsp, bram, latency_cycles, ii}`` dict.

It does **not** touch any folding (`fold_*`) or timing (`t_*`) fields —
those belong to later phases.

Usage::

    g_sched_bound = bind(g_sched, keras_model, "IR/Sched-IR/da4ml-resource.yaml")
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import sys

import yaml

from heterograph import HGraph


# --------------------------------------------------------------------------- #
# Sibling-module load (IR/Sched-IR has a hyphen, so we go via importlib)
# --------------------------------------------------------------------------- #

def _load_sibling(name: str):
    here = Path(__file__).resolve().parent
    full_name = f"_sched_ir_{name}"
    spec = importlib.util.spec_from_file_location(full_name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod  # required for dataclasses / introspection
    spec.loader.exec_module(mod)
    return mod


_kernels = _load_sibling("kernels")
WeightProvider = _kernels.WeightProvider
REGISTRY: dict[str, Callable[..., dict[str, Any]]] = _kernels.REGISTRY


# --------------------------------------------------------------------------- #
# Kernel-library construction
# --------------------------------------------------------------------------- #

class KernelEntry:
    """One row of the resource YAML's ``kernels:`` mapping."""

    __slots__ = ("name", "primitives", "constraints", "cost_query_name",
                 "inserted_by", "instances", "max_instances")

    def __init__(self, name, primitives, constraints, cost_query_name,
                 inserted_by, instances, max_instances):
        self.name = name
        self.primitives = primitives
        self.constraints = constraints
        self.cost_query_name = cost_query_name
        self.inserted_by = inserted_by
        self.instances = instances
        self.max_instances = max_instances

    def cost_query(self, p: dict, weights: WeightProvider, fpga: dict) -> dict[str, Any]:
        fn = REGISTRY[self.cost_query_name]
        return fn(p, weights, fpga)


def build_kernel_library(cfg: dict) -> dict[str, list[KernelEntry]]:
    """Group kernel entries by the primitive they support."""
    library: dict[str, list[KernelEntry]] = {}
    for kname, spec in (cfg.get("kernels") or {}).items():
        entry = KernelEntry(
            name=kname,
            primitives=tuple(spec.get("supported_ops") or ()),
            constraints=dict(spec.get("constraints") or {}),
            cost_query_name=spec.get("cost_query"),
            inserted_by=spec.get("inserted_by"),
            instances=str(spec.get("instances") or "unlimited"),
            max_instances=spec.get("max_instances"),
        )
        if entry.cost_query_name and entry.cost_query_name not in REGISTRY:
            raise ValueError(
                f"Resource YAML kernel {kname!r} references unknown cost_query "
                f"{entry.cost_query_name!r}; add it to kernels.REGISTRY"
            )
        for prim in entry.primitives:
            library.setdefault(prim, []).append(entry)
    return library


def _select_kernel(p: dict, candidates: list[KernelEntry]) -> KernelEntry:
    """Return the first kernel whose constraints are satisfied by `p`.

    Skips scheduler-inserted kernels — those are only chosen during the
    later INSERT-INFRASTRUCTURE phase, never during BIND of decomposer-emitted
    vertices.
    """
    op_params = p.get("op_params") or {}
    for entry in candidates:
        if entry.inserted_by == "scheduler" and p.get("inserted_by") != "scheduler":
            continue
        if _constraints_ok(entry.constraints, op_params):
            return entry
    raise ValueError(
        f"no kernel in the resource YAML matches vertex "
        f"{p.get('nn_layer_name')!r} (op={p.get('op')!r}, op_params={op_params})"
    )


def _constraints_ok(constraints: dict[str, Any], op_params: dict[str, Any]) -> bool:
    for key, want in constraints.items():
        if key == "weight_source":
            # We only ever produce constant-weight kernels at BIND time.
            if want not in ("constant", None):
                return False
            continue
        if key == "max_depth":
            depth = op_params.get("depth")
            if depth is not None and depth > want:
                return False
            continue
        if key == "min_depth":
            depth = op_params.get("depth")
            if depth is not None and depth < want:
                return False
            continue
        # Unknown constraints: fail closed so we notice misspellings.
        return False
    return True


# --------------------------------------------------------------------------- #
# BIND entry point
# --------------------------------------------------------------------------- #

_NEEDS_BIND = ("dense", "reduce", "elementwise", "activation")


def bind(
    g_sched: HGraph,
    keras_model,
    resource_yaml_path: str | Path,
) -> HGraph:
    """Phase 1 — assign a kernel + cost to every Sched-IR vertex.

    Mutates ``g_sched`` in place and returns it.
    """
    cfg_path = Path(resource_yaml_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    fpga = cfg.get("fpga") or {}
    library = build_kernel_library(cfg)
    weights = WeightProvider(keras_model)

    g_sched.pmap["resource_yaml"] = str(cfg_path)
    g_sched.pmap["target_device"] = fpga.get("device")

    next_instance: dict[str, int] = {}

    for vx in g_sched.vertices:
        p = g_sched.pmap[vx]
        prim = p.get("op")
        if prim not in _NEEDS_BIND:
            # buffer / mux are scheduler-inserted; nothing to bind here.
            continue

        candidates = library.get(prim) or []
        if not candidates:
            raise ValueError(f"resource YAML has no kernel for primitive {prim!r}")

        chosen = _select_kernel(p, candidates)
        cost = chosen.cost_query(p, weights, fpga)

        p["kernel_type"]     = chosen.name
        p["kernel_instance"] = next_instance.setdefault(chosen.name, 0)
        next_instance[chosen.name] += 1
        p["cost"]            = cost

    _validate_bind(g_sched)
    return g_sched


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

_REQUIRED_COST_KEYS = ("lut", "ff", "dsp", "bram", "latency_cycles", "ii")


def _validate_bind(g_sched: HGraph) -> None:
    for vx in g_sched.vertices:
        p = g_sched.pmap[vx]
        prim = p.get("op")
        if prim not in _NEEDS_BIND:
            continue
        if p.get("kernel_type") is None:
            raise ValueError(f"BIND left vertex {vx} ({p.get('nn_layer_name')!r}) without a kernel_type")
        if p.get("kernel_instance") is None:
            raise ValueError(f"BIND left vertex {vx} without a kernel_instance")
        cost = p.get("cost")
        if not isinstance(cost, dict):
            raise ValueError(f"BIND left vertex {vx} without a cost dict")
        missing = [k for k in _REQUIRED_COST_KEYS if k not in cost]
        if missing:
            raise ValueError(f"BIND vertex {vx} cost dict missing keys: {missing}")
