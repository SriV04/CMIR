from __future__ import annotations

import importlib.util
import sys
import tempfile
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


fake_heterograph = ModuleType("heterograph")
fake_heterograph.HGraph = FakeHGraph
sys.modules.setdefault("heterograph", fake_heterograph)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


schema = _load("sched_schema_bind_test", HERE / "schema.py")
binder = _load("sched_binder_bind_test", HERE / "binder.py")


class _Model:
    def get_layer(self, layer_name):
        raise LookupError(layer_name)


class BindKernelResultTests(unittest.TestCase):
    def _write_resource_yaml(self, cost_query_name: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(
            f"""
fpga:
  device: VU13P
  latency_cutoff: 2
kernels:
  fake_dense:
    supported_ops: [dense]
    constraints:
      weight_source: constant
    instances: unlimited
    cost_query: {cost_query_name}
"""
        )
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def _build_graph(self) -> FakeHGraph:
        g = FakeHGraph(vinit=schema.vinit_sched, einit=schema.einit_sched, ginit=schema.ginit_sched)
        vx = g.add_vx()
        p = g.pmap[vx]
        p["op"] = "dense"
        p["nn_layer_name"] = "dense_0"
        p["inserted_by"] = "decomposer"
        p["precision_source"] = "hgq"
        p["output_tensor_width_bits"] = 4
        p["op_params"] = {
            "in_bw": 4.0,
            "input_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "input_kif": {"k": True, "i": 2, "f": 1, "bits": 4},
            "output_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "output_kif": {"k": True, "i": 2, "f": 1, "bits": 4},
            "out_bw": 4.0,
        }
        return g

    def test_bind_stores_full_kernel_result_and_updates_output_precision(self):
        def _fake_full_result(p, weights, fpga):
            return {
                "cost": {
                    "lut": 17,
                    "ff": 9,
                    "dsp": 0,
                    "bram": 0,
                    "latency_cycles": 3,
                    "ii": 1,
                },
                "input_qints": [{"min": -1.0, "max": 1.0, "step": 0.5}],
                "input_kifs": [{"k": True, "i": 2, "f": 1, "bits": 4}],
                "output_qints": [{"min": -4.0, "max": 3.0, "step": 1.0}],
                "output_kifs": [{"k": True, "i": 3, "f": 0, "bits": 4}],
                "input_tensor_width_bits": 4,
                "output_tensor_width_bits": 4,
                "precision_source": "da4ml",
                "da4ml": {"solution_type": "FakePipeline"},
            }

        binder.REGISTRY["fake_dense_result"] = _fake_full_result
        cfg = self._write_resource_yaml("fake_dense_result")
        g = self._build_graph()

        out = binder.bind(g, _Model(), cfg)
        p = out.pmap[0]

        self.assertEqual(p["kernel_type"], "fake_dense")
        self.assertEqual(p["kernel_instance"], 0)
        self.assertEqual(p["cost"]["lut"], 17)
        self.assertEqual(p["cost"]["latency_cycles"], 3)
        self.assertEqual(p["precision_source"], "da4ml")
        self.assertEqual(p["kernel_result"]["da4ml"]["solution_type"], "FakePipeline")
        self.assertEqual(p["output_qints"][0]["step"], 1.0)
        self.assertEqual(p["output_kifs"][0]["bits"], 4)
        self.assertEqual(p["output_tensor_width_bits"], 4)
        self.assertEqual(p["op_params"]["output_qint"]["step"], 1.0)
        self.assertEqual(p["op_params"]["output_kif"]["bits"], 4)
        self.assertEqual(p["op_params"]["out_bw"], 4)

    def test_bind_normalizes_legacy_cost_dict(self):
        def _fake_cost_only(p, weights, fpga):
            return {
                "lut": 5,
                "ff": 2,
                "dsp": 0,
                "bram": 0,
                "latency_cycles": 1,
                "ii": 1,
            }

        binder.REGISTRY["fake_cost_only"] = _fake_cost_only
        cfg = self._write_resource_yaml("fake_cost_only")
        g = self._build_graph()

        out = binder.bind(g, _Model(), cfg)
        p = out.pmap[0]

        self.assertEqual(p["cost"]["lut"], 5)
        self.assertEqual(p["kernel_result"]["cost"]["ff"], 2)
        self.assertEqual(p["kernel_result"]["precision_source"], "closed_form")
        self.assertIsNone(p["kernel_result"]["output_qints"])
        self.assertEqual(p["precision_source"], "closed_form")


if __name__ == "__main__":
    unittest.main()
