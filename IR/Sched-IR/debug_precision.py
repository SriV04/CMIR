from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "heterograph"))


def _sample(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _count(value) -> int:
    if value is None:
        return 0
    return len(value) if isinstance(value, list) else 1


def print_precision(g) -> None:
    print("Nodes:")
    print("vx | op | layer | precision_source | out_width | #out_kifs | out_bw legacy")
    for vx in g.vertices:
        p = g.pmap[vx]
        params = p.get("op_params") or {}
        out_kifs = p.get("output_kifs")
        print(
            f"{vx} | {p.get('op')} | {p.get('nn_layer_name')} | "
            f"{p.get('precision_source')} | {p.get('output_tensor_width_bits')} | "
            f"{_count(out_kifs)} | {params.get('out_bw')}"
        )

    print()
    print("Edges:")
    print("u -> v | shape | bitwidth | tensor_width_bits | needs_cast | src_kif sample | dst_kif sample")
    for u, v in g.edges:
        ep = g.pmap[(u, v)]
        print(
            f"{u} -> {v} | {ep.get('tensor_shape')} | {ep.get('bitwidth')} | "
            f"{ep.get('tensor_width_bits')} | {ep.get('needs_cast')} | "
            f"{_sample(ep.get('src_kif'))} | {_sample(ep.get('dst_kif'))}"
        )


if __name__ == "__main__":
    print("Import this helper from a notebook/script and call print_precision(g_sched).")
