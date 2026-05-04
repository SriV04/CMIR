from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_da4ml = _load("_sched_ir_da4ml_debug", HERE / "_da4ml.py")


def main() -> None:
    kernel = np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
    in_qint = _da4ml.qint_from_bw(4)

    dense = _da4ml.solve_dense_result(kernel, in_qint, latency_cutoff=2)
    print("dense.cost", dense["cost"])
    print("dense.input_kifs", dense["input_kifs"])
    print("dense.output_kifs", dense["output_kifs"])
    print("dense.output_tensor_width_bits", dense["output_tensor_width_bits"])
    print("dense.precision_source", dense["precision_source"])

    trace = _da4ml.trace_lambda_result(
        input_shapes=[(4,)],
        input_qints=[in_qint],
        body=lambda x: np.sum(x),
        latency_cutoff=2,
    )
    print("trace.cost", trace["cost"])
    print("trace.input_kifs", trace["input_kifs"])
    print("trace.output_kifs", trace["output_kifs"])
    print("trace.output_tensor_width_bits", trace["output_tensor_width_bits"])
    print("trace.precision_source", trace["precision_source"])


if __name__ == "__main__":
    main()
