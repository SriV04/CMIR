from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


da4ml_adapter = _load("sched_da4ml_test", HERE / "_da4ml.py")


class FakeQInt:
    def __init__(self, qmin: float, qmax: float, step: float):
        self.min = qmin
        self.max = qmax
        self.step = step

    @property
    def precision(self):
        class _Prec(tuple):
            def __new__(cls, k, i, f):
                return super().__new__(cls, (k, i, f))

            @property
            def keep_negative(self):
                return self[0]

            @property
            def integers(self):
                return self[1]

            @property
            def fractional(self):
                return self[2]

        return _Prec(self.min < 0, 2, 1)


class FakePipe:
    def __init__(self):
        self.cost = 12.5
        self.reg_bits = 21
        self.solutions = [object(), object(), object()]
        self.shape = (2, 3)
        self.latency = (1.0, 3.0)
        self.out_latencies = [2.0, 2.5, 3.0]
        self.inp_qint = [FakeQInt(-1.0, 1.5, 0.5), FakeQInt(0.0, 3.5, 0.5)]
        self.out_qint = [
            FakeQInt(-2.0, 1.5, 0.5),
            FakeQInt(0.0, 7.5, 0.5),
            FakeQInt(0.0, 1.5, 0.5),
        ]


class DA4MLResultHelpersTests(unittest.TestCase):
    def test_kifs_payload_to_dicts_handles_transposed_array(self):
        payload = np.array([[1, 0, 1], [2, 3, 1], [1, 0, 2]])
        out = da4ml_adapter.kifs_payload_to_dicts(payload)

        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["bits"], 4)
        self.assertEqual(out[1]["bits"], 3)
        self.assertEqual(out[2]["bits"], 4)

    def test_solution_to_result_preserves_cost_and_precision_metadata(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_ensure = da4ml_adapter._ensure_pipeline
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter._ensure_pipeline = lambda sol, latency_cutoff: FakePipe()
            result = da4ml_adapter.solution_to_result(object(), latency_cutoff=2)
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter._ensure_pipeline = prev_ensure

        self.assertEqual(result["cost"]["lut"], 12)
        self.assertEqual(result["cost"]["ff"], 21)
        self.assertEqual(result["cost"]["latency_cycles"], 3)
        self.assertEqual(result["cost"]["pipeline_stages"], 3)
        self.assertEqual(result["precision_source"], "da4ml")
        self.assertEqual(len(result["input_qints"]), 2)
        self.assertEqual(len(result["output_qints"]), 3)
        self.assertEqual(len(result["input_kifs"]), 2)
        self.assertEqual(len(result["output_kifs"]), 3)
        self.assertEqual(result["input_tensor_width_bits"], 7)
        self.assertEqual(result["output_tensor_width_bits"], 10)
        self.assertEqual(result["da4ml"]["n_inputs"], 2)
        self.assertEqual(result["da4ml"]["n_outputs"], 3)
        self.assertEqual(result["da4ml"]["pipeline_stages"], 3)


if __name__ == "__main__":
    unittest.main()
