from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, str(REPO / "JEDI-linear" / "src"))
sys.path.insert(0, str(REPO / "heterograph"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


builder = _load("nn_ir_builder_debug", HERE / "builder.py")

from model import get_gnn  # noqa: E402


def _fmt_bw(value):
    return "?" if value is None else f"{value:.2f}"


def _fmt_shape(shape):
    if shape is None:
        return "?"
    return "x".join("?" if dim is None else str(dim) for dim in shape)


def _fmt_kif(kif):
    if not kif:
        return "?"
    bits = kif.get("bits")
    shape = kif.get("shape")
    if bits is None:
        return f"shape={shape}"
    if hasattr(bits, "shape"):
        return f"shape={shape} max={float(bits.max()):.2f}"
    return str(bits)


def main() -> None:
    conf = SimpleNamespace(n_constituents=8, pt_eta_phi=True)
    model = get_gnn(conf)
    graph = builder.build_nn_ir(model, name="jedi_gnn")

    print(
        "layer | op | iq_kif | kq_kif | qkernel_shape | sparsity | unique_values"
    )
    print("-" * 100)
    for vx in graph.vertices:
        p = graph.pmap[vx]
        print(
            f"{p['layer_name']} | {p['op_kind']} | "
            f"{_fmt_kif(p.get('iq_kif'))} | {_fmt_kif(p.get('kq_kif'))} | "
            f"{_fmt_shape(p.get('kernel_shape'))} | {_fmt_bw(p.get('kernel_sparsity'))} | "
            f"{p.get('kernel_unique_count')}"
        )


if __name__ == "__main__":
    main()
