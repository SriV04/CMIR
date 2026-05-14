from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent


class FakeHGraph:
    def __init__(self, *, vinit=None, einit=None, ginit=None):
        self._vinit = vinit
        self._einit = einit
        self._ginit = ginit
        self.vertices = []
        self.edges = []
        self.pmap = {}
        if self._ginit is not None:
            self._ginit(self)

    def add_vx(self):
        vx = len(self.vertices)
        self.vertices.append(vx)
        if self._vinit is not None:
            self._vinit(self, vx)
        return vx

    def add_edge(self, src, dst):
        edge = (src, dst)
        if edge in self.edges:
            return []
        self.edges.append(edge)
        if self._einit is not None:
            self._einit(self, edge)
        return [edge]

    def in_vx(self, vx):
        return [src for src, dst in self.edges if dst == vx]

    def out_vx(self, vx):
        return [dst for src, dst in self.edges if src == vx]


fake_heterograph = ModuleType("heterograph")
fake_heterograph.HGraph = FakeHGraph
sys.modules.setdefault("heterograph", fake_heterograph)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


schema = _load("sched_schema_fold_precision_test", HERE / "schema.py")
kernels = _load("sched_kernels_fold_precision_test", HERE / "cost_evals" / "kernels.py")
fold_precision = _load("sched_fold_precision_test", HERE / "fold_precision.py")


class FoldPrecisionTests(unittest.TestCase):
    def _reduce_pmap(self) -> dict:
        return {
            "op": "reduce",
            "nn_layer_name": "reduce_sum",
            "reduce_mode": "hybrid",
            "parallelism_N": 8,
            "lanes_P": 4,
            "temporal_steps_T": 2,
            "op_params": {
                "mode": "sum",
                "axes": [1],
                "in_shape": (None, 8, 4),
                "input_qint": {"min": -16.0, "max": 15.75, "step": 0.25},
                "input_kif": {"k": True, "i": 4, "f": 2, "bits": 7},
                "in_bw": 2.0,
            },
            "cost": {"lut": 999, "ff": 999, "dsp": 0, "bram": 0, "latency_cycles": 9, "ii": 1},
        }

    def test_folded_reduce_result_uses_kif_derived_accumulator_width(self):
        prev_trace = kernels._da4ml.trace_lambda_result
        try:
            kernels._da4ml.trace_lambda_result = lambda *args, **kwargs: {
                "cost": {
                    "lut": 20,
                    "ff": 10,
                    "dsp": 0,
                    "bram": 0,
                    "latency_cycles": 2,
                    "ii": 1,
                }
            }

            result = kernels.da4ml_reduce_folded_result(
                self._reduce_pmap(),
                kernels.WeightProvider(None),
                {},
                parallelism=8,
                factor=2,
            )
        finally:
            kernels._da4ml.trace_lambda_result = prev_trace

        self.assertEqual(result["cost"]["ii"], 2)
        self.assertEqual(result["cost"]["latency_cycles"], 3)
        self.assertEqual(result["cost"]["lut"], 60)
        self.assertEqual(result["cost"]["ff"], 50)
        self.assertEqual(result["kernel_meta"]["partial_sum_kif"]["bits"], 9)
        self.assertEqual(result["kernel_meta"]["accumulator_kif"]["bits"], 10)
        self.assertEqual(result["kernel_meta"]["partial_sum_qint"], {"min": -64.0, "max": 63.0, "step": 0.25})
        self.assertEqual(result["output_qints"][0], {"min": -128.0, "max": 126.0, "step": 0.25})
        self.assertEqual(result["output_tensor_width_bits"], 40)
        self.assertEqual(result["precision_source"], "fold_aware_derived")

    def test_folded_reduce_result_collapses_array_qint_conservatively(self):
        p = self._reduce_pmap()
        p["reduce_mode"] = "temporal_accumulate"
        p["lanes_P"] = 1
        p["temporal_steps_T"] = 8
        p["op_params"]["input_qint"] = {
            "min": [[[-16.0], [-8.0], [-4.0], [-2.0], [-16.0], [-8.0], [-4.0], [-2.0]]],
            "max": [[[15.75], [7.75], [3.75], [1.75], [15.75], [7.75], [3.75], [1.75]]],
            "step": [[[0.25], [0.125], [0.0625], [0.03125], [0.25], [0.125], [0.0625], [0.03125]]],
        }

        result = kernels.da4ml_reduce_folded_result(
            p,
            kernels.WeightProvider(None),
            {},
            parallelism=8,
            factor=8,
        )

        self.assertEqual(result["kernel_meta"]["precision_warning"], "used conservative folded-reduce precision for array input qint")
        self.assertEqual(result["output_qints"][0], {"min": -128.0, "max": 126.0, "step": 0.03125})
        self.assertEqual(result["kernel_meta"]["accumulator_kif"]["bits"], 13)

    def test_apply_fold_aware_precision_updates_reduce_and_repropagates_edges(self):
        g = FakeHGraph(vinit=schema.vinit_sched, einit=schema.einit_sched, ginit=schema.ginit_sched)
        pred = g.add_vx()
        red = g.add_vx()
        dst = g.add_vx()
        g.add_edge(pred, red)
        g.add_edge(red, dst)

        g.pmap[pred]["op"] = "dense"
        g.pmap[pred]["nn_layer_name"] = "producer"
        g.pmap[pred]["output_kifs"] = [{"k": True, "i": 4, "f": 2, "bits": 7}]
        g.pmap[pred]["output_qints"] = [{"min": -16.0, "max": 15.75, "step": 0.25}]
        g.pmap[pred]["output_tensor_width_bits"] = 224
        g.pmap[pred]["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}

        red_p = g.pmap[red]
        red_p.update(self._reduce_pmap())
        red_p["output_kifs"] = [{"k": True, "i": 9, "f": 0, "bits": 10}]
        red_p["output_tensor_width_bits"] = 40

        g.pmap[dst]["op"] = "dense"
        g.pmap[dst]["nn_layer_name"] = "consumer"
        g.pmap[dst]["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}
        g.pmap[dst]["op_params"] = {}
        g.pmap[dst]["output_qints"] = [{"min": -1.0, "max": 1.0, "step": 0.25}]
        g.pmap[dst]["output_kifs"] = [{"k": True, "i": 2, "f": 1, "bits": 4}]
        g.pmap[dst]["output_tensor_width_bits"] = 4
        g.pmap[(pred, red)]["tensor_shape"] = (None, 8, 4)
        g.pmap[(red, dst)]["tensor_shape"] = (None, 4)

        fake_result = {
            "cost": {"lut": 60, "ff": 50, "dsp": 0, "bram": 0, "latency_cycles": 3, "ii": 2},
            "input_qints": [{"min": -16.0, "max": 15.75, "step": 0.25}],
            "input_kifs": [{"k": True, "i": 4, "f": 2, "bits": 7}],
            "output_qints": [{"min": -128.0, "max": 126.0, "step": 0.25}] * 4,
            "output_kifs": [{"k": True, "i": 7, "f": 2, "bits": 10}] * 4,
            "input_tensor_width_bits": 7,
            "output_tensor_width_bits": 40,
            "precision_source": "fold_aware_derived",
            "kernel_meta": {
                "partial_sum_qint": {"min": -64.0, "max": 63.0, "step": 0.25},
                "partial_sum_kif": {"k": True, "i": 6, "f": 2, "bits": 9},
                "accumulator_qint": {"min": -128.0, "max": 126.0, "step": 0.25},
                "accumulator_kif": {"k": True, "i": 7, "f": 2, "bits": 10},
            },
        }

        prev = fold_precision._kernels.da4ml_reduce_folded_result
        try:
            fold_precision._kernels.da4ml_reduce_folded_result = lambda *args, **kwargs: fake_result
            fold_precision.apply_fold_aware_precision(g, strict=True)
        finally:
            fold_precision._kernels.da4ml_reduce_folded_result = prev

        self.assertEqual(red_p["cost"], fake_result["cost"])
        self.assertEqual(red_p["precision_source"], "fold_aware_derived")
        self.assertEqual(red_p["op_params"]["spatial_width_P"], 4)
        self.assertEqual(red_p["op_params"]["temporal_steps_T"], 2)
        self.assertEqual(red_p["op_params"]["partial_sum_kif"]["bits"], 9)
        self.assertEqual(red_p["op_params"]["accumulator_kif"]["bits"], 10)
        self.assertEqual(g.pmap[(red, dst)]["src_kif"], fake_result["output_kifs"])
        self.assertEqual(g.pmap[(red, dst)]["tensor_width_bits"], 40)


if __name__ == "__main__":
    unittest.main()
