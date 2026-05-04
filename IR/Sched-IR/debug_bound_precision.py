from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "heterograph"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def print_bound_precision(g) -> None:
    print("node | op | lut | L | precision_source | #out_qints | out_bits_sum | out_kif_sample")
    for vx in g.vertices:
        p = g.pmap[vx]
        out_qints = p.get("output_qints") or []
        out_kifs = p.get("output_kifs") or []
        sample = out_kifs[0] if out_kifs else None
        lut = (p.get("cost") or {}).get("lut")
        L = (p.get("cost") or {}).get("latency_cycles")
        print(
            f"{vx} | {p.get('op')} | {lut} | {L} | {p.get('precision_source')} | "
            f"{len(out_qints)} | {p.get('output_tensor_width_bits')} | {sample}"
        )


if __name__ == "__main__":
    print("Import this helper from a notebook/script and call print_bound_precision(g_bound).")
