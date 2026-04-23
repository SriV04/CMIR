"""Standalone validation of the N–P–T timing model.

Covers the four cases mandated by the refactor spec:

    Case 1: N=8, P=8  → T=1, II=1        (fully unfolded / da4ml baseline)
    Case 2: N=8, P=4  → T=2, II=2        (partial folding)
    Case 3: N=8, P=1  → T=8, II=8        (fully temporal)
    Case 4: N=8, P=4  → T=2, II=2        (reduction hybrid)

The tests exercise FOLD + SCHEDULE end-to-end on a tiny two-vertex
dense→reduce Sched-IR graph. They don't need a Keras model: both cost
queries are stubbed with a constant pipeline depth L, so the assertions
exercise only the N–P–T / II / latency_total invariants.

Run from the repo root::

    conda run -n jedi-linear python IR/Sched-IR/test_npt_model.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Paths / module loading
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "heterograph"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


schema_m  = _load("_sched_ir_schema", HERE / "schema.py")
folder_m  = _load("_sched_ir_folder", HERE / "folder.py")
sched_m   = _load("_sched_ir_scheduler_p3", HERE / "scheduler_p3.py")

from heterograph import HGraph  # noqa: E402


L_DENSE  = 3    # stubbed pipeline depth for the dense vertex
L_REDUCE = 2    # stubbed pipeline depth for the reduce vertex (spatial part)


# --------------------------------------------------------------------------- #
# Tiny two-vertex graph: dense → reduce over an N-sized fold axis
# --------------------------------------------------------------------------- #

def _build_graph(N: int) -> HGraph:
    g = HGraph(
        vinit=schema_m.vinit_sched,
        einit=schema_m.einit_sched,
        ginit=schema_m.ginit_sched,
    )

    # Dense vertex on axis 1 of a (B, N, C) tensor.
    v_dense = g.add_vx()
    pd = g.pmap[v_dense]
    pd["op"]            = "dense"
    pd["nn_layer_name"] = "fake_dense"
    pd["fold_axes"]     = [1]
    pd["op_params"]     = {"in_bw": 8.0, "kernel_shape": (4, 4)}
    pd["cost"]          = {"lut": 0, "ff": 0, "dsp": 0, "bram": 0,
                           "latency_cycles": L_DENSE, "ii": 1}

    # Reduce vertex that consumes the same fold axis.
    v_red = g.add_vx()
    pr = g.pmap[v_red]
    pr["op"]            = "reduce"
    pr["nn_layer_name"] = "fake_reduce"
    pr["fold_axes"]     = [1]
    pr["op_params"]     = {
        "mode":     "sum",
        "axes":     [1],
        "in_shape": (None, N, 4),
        "in_bw":    8.0,
        "out_bw":   10.0,
    }
    pr["reduce_mode"]   = "spatial"
    pr["cost"]          = {"lut": 0, "ff": 0, "dsp": 0, "bram": 0,
                           "latency_cycles": L_REDUCE, "ii": 1}

    # dense -> reduce edge carrying the fold axis.
    e = g.add_edge(v_dense, v_red)[0]
    ep = g.pmap[e]
    ep["tensor_shape"] = (None, N, 4)
    ep["bitwidth"]     = 8.0
    ep["volume_bits"]  = N * 4 * 8
    ep["edge_kind"]    = "data"

    # FOLD needs a resource YAML on the graph to recover fpga config.
    g.pmap["resource_yaml"] = str(HERE / "da4ml-resource.yaml")

    return g, v_dense, v_red


# --------------------------------------------------------------------------- #
# Single case runner
# --------------------------------------------------------------------------- #

def _run_case(label: str, N: int, P: int, T_expected: int, II_expected: int) -> None:
    g, v_dense, v_red = _build_graph(N)

    folder_m.fold(g, lanes=P)
    sched_m.schedule(g)

    for vx, name in ((v_dense, "dense"), (v_red, "reduce")):
        p = g.pmap[vx]
        assert p["parallelism_N"]    == N,           f"{label} {name}: N mismatch"
        assert p["lanes_P"]          == P,           f"{label} {name}: P mismatch"
        assert p["temporal_steps_T"] == T_expected,  f"{label} {name}: T mismatch"
        assert p["ii"]               == II_expected, f"{label} {name}: II mismatch"
        assert p["elements_per_cycle"] == P,         f"{label} {name}: P_per_cycle mismatch"

        # Hard invariants.
        L = int(p["pipeline_latency_L"])
        assert p["ii"] == math.ceil(N / max(P, 1)), (
            f"{label} {name}: II={p['ii']} != ceil(N/P)={math.ceil(N / max(P, 1))}"
        )
        assert p["latency_total"] == L + (p["ii"] - 1), (
            f"{label} {name}: latency_total={p['latency_total']} != L+(II-1)={L + (p['ii'] - 1)}"
        )

        # t_end consistency.
        assert p["t_end"] == p["t_start"] + p["latency_total"], (
            f"{label} {name}: t_end != t_start + latency_total"
        )

    # Reduction mode flips correctly under the different (N, P) regimes.
    pr = g.pmap[v_red]
    if P == N:
        assert pr["reduce_mode"] == "spatial",            f"{label}: reduce_mode should be spatial"
    elif P == 1:
        assert pr["reduce_mode"] == "temporal_accumulate", f"{label}: reduce_mode should be temporal_accumulate"
    else:
        assert pr["reduce_mode"] == "hybrid",              f"{label}: reduce_mode should be hybrid"

    # Graph-level II == max(node.ii).
    assert g.pmap["initiation_interval"] == II_expected, (
        f"{label}: graph II={g.pmap['initiation_interval']} != expected {II_expected}"
    )

    print(f"  {label:<32s}  N={N} P={P}  →  T={T_expected}  II={II_expected}  ✓")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    print("N–P–T timing model — validation cases:\n")
    _run_case("Case 1  (fully unfolded)", N=8, P=8, T_expected=1, II_expected=1)
    _run_case("Case 2  (partial folding)", N=8, P=4, T_expected=2, II_expected=2)
    _run_case("Case 3  (fully temporal)", N=8, P=1, T_expected=8, II_expected=8)
    _run_case("Case 4  (hybrid reduce)",  N=8, P=4, T_expected=2, II_expected=2)
    print("\nAll N–P–T invariants hold.")


if __name__ == "__main__":
    main()
