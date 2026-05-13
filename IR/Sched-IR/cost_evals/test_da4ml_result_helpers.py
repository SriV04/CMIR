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


da4ml_adapter = _load("sched_da4ml_cost_evals_test", HERE / "_da4ml.py")


class FakeQInt:
    def __init__(self, qmin: float, qmax: float, step: float, precision=(1, 2, 1)):
        self.min = qmin
        self.max = qmax
        self.step = step
        self.precision = precision


class FakePipe:
    def __init__(self):
        self.cost = 12.5
        self.reg_bits = 21
        self.solutions = [object(), object(), object()]
        self.shape = (99, 77)
        self.latency = (1.0, 3.0)
        self.out_latencies = [2.0, 2.5, 3.0]
        self.inp_qint = [FakeQInt(-1.0, 1.5, 0.5), FakeQInt(0.0, 3.5, 0.5)]
        self.out_qint = [
            FakeQInt(-2.0, 1.5, 0.5),
            FakeQInt(0.0, 7.5, 0.5),
            FakeQInt(0.0, 1.5, 0.5),
        ]
        self.inp_kifs = [(1, 3, 0), (0, 2, 2)]
        self.out_kifs = [(1, 2, 1), (1, 4, 0), (0, 1, 3)]


class FakeQIntervalCtor:
    @staticmethod
    def from_kif(k, i, f):
        return ("from_kif", k, i, f)

    def __call__(self, qmin, qmax, step):
        return ("from_dict", qmin, qmax, step)


class FakeQIntervalType:
    @staticmethod
    def from_kif(k, i, f):
        return ("from_kif", k, i, f)

    def __new__(cls, qmin, qmax, step):
        return ("from_dict", qmin, qmax, step)


class DA4MLResultHelpersTests(unittest.TestCase):
    def test_qint_to_dict_handles_tuple_like_payload(self):
        self.assertEqual(
            da4ml_adapter.qint_to_dict((-1.0, 3.5, 0.25)),
            {"min": -1.0, "max": 3.5, "step": 0.25},
        )

    def test_kifs_payload_to_dicts_handles_array_valued_kif_dict(self):
        payload = {
            "k": np.array([1, 0]),
            "i": np.array([2, 3]),
            "f": np.array([1, 0]),
        }

        out = da4ml_adapter.kifs_payload_to_dicts(payload)

        self.assertEqual(
            out,
            [
                {"k": True, "i": 2, "f": 1, "bits": 4},
                {"k": False, "i": 3, "f": 0, "bits": 3},
            ],
        )

    def test_solution_to_result_prefers_pipe_kifs_and_counts_logical_ios(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_ensure = da4ml_adapter._ensure_pipeline
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter._ensure_pipeline = lambda sol, latency_cutoff: FakePipe()
            result = da4ml_adapter.solution_to_result(object(), latency_cutoff=2)
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter._ensure_pipeline = prev_ensure

        self.assertEqual(result["input_kifs"][0], {"k": True, "i": 3, "f": 0, "bits": 4})
        self.assertEqual(result["output_kifs"][1], {"k": True, "i": 4, "f": 0, "bits": 5})
        self.assertEqual(result["input_tensor_width_bits"], 8)
        self.assertEqual(result["output_tensor_width_bits"], 13)
        self.assertEqual(result["da4ml"]["n_inputs"], 2)
        self.assertEqual(result["da4ml"]["n_outputs"], 3)
        self.assertEqual(result["da4ml"]["shape"], (99, 77))

    def test_qint_from_kif_dict_uses_integer_keep_negative(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_qinterval = da4ml_adapter.QInterval
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter.QInterval = FakeQIntervalCtor()
            out = da4ml_adapter.qint_from_kif_dict({"k": True, "i": 3, "f": 1})
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter.QInterval = prev_qinterval

        self.assertEqual(out, ("from_kif", 1, 3, 1))

    def test_array_qint_global_precision_collapses_to_scalar(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_qinterval = da4ml_adapter.QInterval
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter.QInterval = FakeQIntervalType
            payload = {
                "min": np.full((1, 8, 3), -16.0),
                "max": np.full((1, 8, 3), 15.75),
                "step": np.full((1, 8, 3), 0.25),
            }

            out = da4ml_adapter.qints_from_precision_payload(
                payload,
                feature_count=3,
                context="dense vertex 'q_dense'",
            )
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter.QInterval = prev_qinterval

        self.assertEqual(out, ("from_dict", -16.0, 15.75, 0.25))

    def test_array_qint_per_feature_precision_collapses_to_feature_list(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_qinterval = da4ml_adapter.QInterval
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter.QInterval = FakeQIntervalType
            payload = {
                "min": np.broadcast_to(np.array([-16.0, -8.0, -4.0]), (1, 8, 3)),
                "max": np.broadcast_to(np.array([15.75, 7.75, 3.75]), (1, 8, 3)),
                "step": np.broadcast_to(np.array([0.25, 0.125, 0.0625]), (1, 8, 3)),
            }

            out = da4ml_adapter.qints_from_precision_payload(
                payload,
                feature_count=3,
                context="dense vertex 'q_dense'",
            )
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter.QInterval = prev_qinterval

        self.assertEqual(
            out,
            [
                ("from_dict", -16.0, 15.75, 0.25),
                ("from_dict", -8.0, 7.75, 0.125),
                ("from_dict", -4.0, 3.75, 0.0625),
            ],
        )

    def test_array_qint_rejects_non_feature_axis_variation(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_qinterval = da4ml_adapter.QInterval
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter.QInterval = FakeQIntervalType
            min_vals = np.broadcast_to(np.array([-16.0, -8.0, -4.0]), (1, 8, 3)).copy()
            min_vals[0, 5, 0] = -32.0
            payload = {
                "min": min_vals,
                "max": np.broadcast_to(np.array([15.75, 7.75, 3.75]), (1, 8, 3)),
                "step": np.broadcast_to(np.array([0.25, 0.125, 0.0625]), (1, 8, 3)),
            }

            with self.assertRaisesRegex(ValueError, "varies across non-feature axes"):
                da4ml_adapter.qints_from_precision_payload(
                    payload,
                    feature_count=3,
                    context="dense vertex 'q_dense'",
                )
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter.QInterval = prev_qinterval

    def test_array_qint_rejects_feature_count_mismatch(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_qinterval = da4ml_adapter.QInterval
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter.QInterval = FakeQIntervalType
            payload = {
                "min": np.broadcast_to(np.array([-16.0, -8.0, -4.0, -2.0]), (1, 8, 4)),
                "max": np.broadcast_to(np.array([15.75, 7.75, 3.75, 1.75]), (1, 8, 4)),
                "step": np.broadcast_to(np.array([0.25, 0.125, 0.0625, 0.03125]), (1, 8, 4)),
            }

            with self.assertRaisesRegex(ValueError, "last dimension == feature_count=3"):
                da4ml_adapter.qints_from_precision_payload(
                    payload,
                    feature_count=3,
                    context="dense vertex 'q_dense'",
                )
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter.QInterval = prev_qinterval

    def test_array_kif_global_precision_collapses_to_scalar_qint(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_qinterval = da4ml_adapter.QInterval
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter.QInterval = FakeQIntervalType
            payload = {
                "k": np.ones((1, 8, 3), dtype=np.uint8),
                "i": np.full((1, 8, 3), 4),
                "f": np.full((1, 8, 3), 2),
            }

            out = da4ml_adapter.qints_from_precision_payload(
                None,
                payload,
                feature_count=3,
                context="dense vertex 'q_dense'",
            )
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter.QInterval = prev_qinterval

        self.assertEqual(out, ("from_kif", 1, 4, 2))

    def test_array_kif_per_feature_precision_collapses_to_feature_list(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_qinterval = da4ml_adapter.QInterval
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter.QInterval = FakeQIntervalType
            payload = {
                "k": np.broadcast_to(np.array([1, 1, 0]), (1, 8, 3)),
                "i": np.broadcast_to(np.array([4, 3, 2]), (1, 8, 3)),
                "f": np.broadcast_to(np.array([2, 1, 0]), (1, 8, 3)),
            }

            out = da4ml_adapter.qints_from_precision_payload(
                None,
                payload,
                feature_count=3,
                context="dense vertex 'q_dense'",
            )
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter.QInterval = prev_qinterval

        self.assertEqual(
            out,
            [
                ("from_kif", 1, 4, 2),
                ("from_kif", 1, 3, 1),
                ("from_kif", 0, 2, 0),
            ],
        )

    def test_array_kif_rejects_non_feature_axis_variation(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_qinterval = da4ml_adapter.QInterval
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter.QInterval = FakeQIntervalType
            i_vals = np.broadcast_to(np.array([4, 3, 2]), (1, 8, 3)).copy()
            i_vals[0, 2, 1] = 5
            payload = {
                "k": np.broadcast_to(np.array([1, 1, 0]), (1, 8, 3)),
                "i": i_vals,
                "f": np.broadcast_to(np.array([2, 1, 0]), (1, 8, 3)),
            }

            with self.assertRaisesRegex(ValueError, "varies across non-feature axes"):
                da4ml_adapter.qints_from_precision_payload(
                    None,
                    payload,
                    feature_count=3,
                    context="dense vertex 'q_dense'",
                )
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter.QInterval = prev_qinterval

    def test_legacy_cost_to_result_accepts_explicit_precision_source(self):
        result = da4ml_adapter.legacy_cost_to_result(
            0.0,
            (0.0, 0.0),
            output_qints=[{"min": 0.0, "max": 1.0, "step": 0.5}],
            output_kifs=[{"k": False, "i": 1, "f": 1, "bits": 2}],
            precision_source="inherited",
        )

        self.assertEqual(result["precision_source"], "inherited")
        self.assertEqual(result["output_tensor_width_bits"], 2)

    def test_trace_lambda_result_rejects_per_element_qint_length_mismatch(self):
        prev_ok = da4ml_adapter._DA4ML_OK
        prev_hwconfig = da4ml_adapter.HWConfig
        try:
            da4ml_adapter._DA4ML_OK = True
            da4ml_adapter.HWConfig = lambda *args: object()
            with self.assertRaisesRegex(ValueError, "expected 4"):
                da4ml_adapter.trace_lambda_result(
                    input_shapes=[(2, 2)],
                    input_qints=[[(0.0, 1.0, 0.5), (0.0, 1.0, 0.5), (0.0, 1.0, 0.5)]],
                    body=lambda x: x,
                )
        finally:
            da4ml_adapter._DA4ML_OK = prev_ok
            da4ml_adapter.HWConfig = prev_hwconfig


if __name__ == "__main__":
    unittest.main()
