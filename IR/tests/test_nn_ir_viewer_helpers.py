from __future__ import annotations

import importlib.util
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
IR_DIR = HERE.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


viewer_helpers = _load("nn_ir_jedi_viewer", IR_DIR / "nn_ir_jedi_viewer.py")


class ViewerHelpersTests(unittest.TestCase):
    def test_json_safe_handles_generic_mapping_objects(self):
        class FakeMapping(Mapping):
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        value = FakeMapping({"place": "datalane", "bits": np.array([4, 5])})
        safe = viewer_helpers._json_safe(value)

        self.assertEqual(safe["place"], "datalane")
        self.assertEqual(safe["bits"], [4, 5])

    def test_summarize_array_truncates_and_reports_shape(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        summary = viewer_helpers._summarize_array(arr, max_items=5)

        self.assertEqual(summary["shape"], [3, 4])
        self.assertEqual(summary["dtype"], "float32")
        self.assertEqual(summary["size"], 12)
        self.assertEqual(len(summary["preview"]), 5)
        self.assertTrue(summary["truncated"])

    def test_node_details_extracts_quantizer_and_weight_stats(self):
        graph = type("Graph", (), {})()
        graph.pmap = {
            3: {
                "layer_name": "dense_1",
                "layer_class": "QDense",
                "op_kind": "dense",
                "kernel_shape": (4, 2),
                "weights": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
                "qkernel_values": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
                "bias_values": np.array([0.5, -0.5], dtype=np.float32),
                "iq": {"class_name": "FakeIQ", "granularity": "per-activation"},
                "kq": {"class_name": "FakeKQ", "granularity": "per-weight"},
                "iq_kif": {"bits": np.array([4, 4, 4, 4])},
                "kq_kif": {"bits": np.array([[3, 2], [2, 3]])},
                "kernel_sparsity": 0.5,
                "kernel_unique_count": 3,
                "iq_bw_avg": 4.0,
                "kq_bw_avg": 2.5,
                "in_shapes": [(None, 4)],
                "out_shapes": [(None, 2)],
            }
        }

        details = viewer_helpers._node_details(graph, 3)

        self.assertEqual(details["kind"], "node")
        self.assertEqual(details["summary"]["layer_name"], "dense_1")
        self.assertEqual(details["weights"]["kernel_sparsity"], 0.5)
        self.assertEqual(details["weights"]["kernel_unique_count"], 3)
        self.assertEqual(details["weights"]["weights"]["shape"], [2, 2])
        self.assertEqual(details["quantizers"]["iq"]["class_name"], "FakeIQ")


if __name__ == "__main__":
    unittest.main()
