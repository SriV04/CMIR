from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np


HERE = Path(__file__).resolve().parent
NN_IR = HERE.parent
REPO = NN_IR.parent.parent

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

    @property
    def num_vx(self):
        return len(self.vertices)

    @property
    def num_edges(self):
        return len(self.edges)


fake_heterograph = ModuleType("heterograph")
fake_heterograph.HGraph = FakeHGraph
sys.modules.setdefault("heterograph", fake_heterograph)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hgq_extractor = _load("nn_ir_hgq_extractor", NN_IR / "hgq2" / "hgq_extractor.py")
builder = _load("nn_ir_builder_test", NN_IR / "builder.py")


class FakeVariable:
    def __init__(self, name: str, value):
        self.name = name
        self.value = np.array(value)


class FakeQuantizer:
    def __init__(
        self,
        *,
        name: str = "fake_quantizer",
        k=None,
        i=None,
        f=None,
        b=None,
        overflow_mode: str | None = "WRAP",
        round_mode: str | None = "TRN",
        extra_config: dict | None = None,
    ):
        self.name = name
        self.overflow_mode = overflow_mode
        self.round_mode = round_mode
        self.variables = []
        if k is not None:
            self.k = np.array(k)
            self.variables.append(FakeVariable(f"{name}/k", k))
        if i is not None:
            self.i = np.array(i)
            self.variables.append(FakeVariable(f"{name}/i", i))
        if f is not None:
            self.f = np.array(f)
            self.variables.append(FakeVariable(f"{name}/f", f))
        if b is not None:
            self.b = np.array(b)
            self.variables.append(FakeVariable(f"{name}/b", b))
        self._extra_config = extra_config or {}

    @property
    def kif(self):
        if hasattr(self, "k") and hasattr(self, "i") and hasattr(self, "f"):
            return self.k, self.i, self.f
        raise AttributeError("kif unavailable")

    def get_config(self):
        return {
            "name": self.name,
            "overflow_mode": self.overflow_mode,
            "round_mode": self.round_mode,
            **self._extra_config,
        }


class FakeTensor:
    def __init__(self, shape, layer):
        self.shape = shape
        self._keras_history = (layer, 0, 0)


class FakeNode:
    def __init__(self, input_tensors, output_tensors):
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors


InputLayer = type("InputLayer", (), {})
QDense = type("QDense", (), {})


class HGQMetadataTests(unittest.TestCase):
    def test_quantizer_summary_extracts_kif_qint_and_modes(self):
        quantizer = FakeQuantizer(
            name="iq",
            k=[1, 0],
            i=[2, 3],
            f=[1, 0],
            overflow_mode="SAT_SYM",
            round_mode="RND",
            extra_config={"default_q_type": "fixed"},
        )

        summary = hgq_extractor.quantizer_summary(quantizer, place="input")

        self.assertTrue(summary["exists"])
        self.assertEqual(summary["overflow_mode"], "SAT_SYM")
        self.assertEqual(summary["round_mode"], "RND")
        self.assertEqual(summary["granularity"], "per-activation")
        self.assertEqual(summary["shape"], (2,))
        np.testing.assert_array_equal(summary["kif"]["k"], np.array([1, 0]))
        np.testing.assert_array_equal(summary["kif"]["i"], np.array([2, 3]))
        np.testing.assert_array_equal(summary["kif"]["f"], np.array([1, 0]))
        np.testing.assert_array_equal(summary["kif"]["bits"], np.array([4, 3]))
        np.testing.assert_allclose(summary["qint"]["step"], np.array([0.5, 1.0]))

    def test_extract_layer_values_prefers_qkernel_and_keeps_float_kernel(self):
        layer = QDense()
        layer.kernel = np.array([[0.25, -0.5], [1.5, 0.0]], dtype=np.float32)
        layer.qkernel = np.array([[0.0, -1.0], [2.0, 0.0]], dtype=np.float32)
        layer.bias = np.array([0.5, -0.25], dtype=np.float32)

        values = hgq_extractor.extract_layer_values(layer)

        self.assertTrue(values["uses_qkernel"])
        np.testing.assert_array_equal(values["qkernel_values"], layer.qkernel)
        np.testing.assert_array_equal(values["kernel_values"], layer.qkernel)
        np.testing.assert_array_equal(values["kernel_float_values"], layer.kernel)
        np.testing.assert_array_equal(values["bias_values"], layer.bias)

    def test_build_nn_ir_populates_structured_vertex_and_edge_metadata(self):
        input_layer = InputLayer()
        input_layer.name = "input_1"
        input_layer.activation = None
        input_layer._inbound_nodes = []
        input_tensor = FakeTensor((None, 4), input_layer)
        input_layer._inbound_nodes = [FakeNode([], [input_tensor])]

        dense_layer = QDense()
        dense_layer.name = "dense_1"
        dense_layer.activation = None
        dense_layer.kernel = np.array([[0.5, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, -1.0]], dtype=np.float32)
        dense_layer.qkernel = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, -2.0]], dtype=np.float32)
        dense_layer.bias = np.array([0.0, 0.5], dtype=np.float32)
        dense_layer.iq = FakeQuantizer(name="dense_iq", k=[1, 1, 1, 1], i=[2, 2, 2, 2], f=[1, 1, 1, 1])
        dense_layer.kq = FakeQuantizer(name="dense_kq", k=np.ones((4, 2)), i=np.ones((4, 2)), f=np.ones((4, 2)))
        dense_layer.bq = FakeQuantizer(name="dense_bq", k=[1, 1], i=[3, 3], f=[1, 1])
        dense_layer._inbound_nodes = [FakeNode([input_tensor], [FakeTensor((None, 2), dense_layer)])]
        dense_layer.count_params = lambda: 10

        model = type("FakeModel", (), {})()
        model.name = "fake_model"
        model.layers = [input_layer, dense_layer]
        model.input_shape = (None, 4)
        model.output_shape = (None, 2)

        graph = builder.build_nn_ir(model, name="fake_nn_ir")

        dense_vx = next(vx for vx in graph.vertices if graph.pmap[vx]["layer_name"] == "dense_1")
        input_vx = next(vx for vx in graph.vertices if graph.pmap[vx]["layer_name"] == "input_1")
        edge = (input_vx, dense_vx)

        dense_p = graph.pmap[dense_vx]
        edge_p = graph.pmap[edge]

        self.assertTrue(dense_p["uses_qkernel"])
        self.assertIsNotNone(dense_p["iq"])
        self.assertIsNotNone(dense_p["kq"])
        self.assertIsNotNone(dense_p["iq_kif"])
        self.assertIsNotNone(dense_p["iq_qint"])
        self.assertEqual(dense_p["kernel_shape"], (4, 2))
        self.assertEqual(dense_p["iq_bw"], dense_p["iq_bw_avg"])
        self.assertEqual(dense_p["kq_bw"], dense_p["kq_bw_avg"])
        np.testing.assert_array_equal(dense_p["weights"], dense_p["kernel_values"])
        self.assertGreaterEqual(dense_p["kernel_unique_count"], 1)

        self.assertEqual(edge_p["tensor_shape"], (None, 4))
        self.assertIsNotNone(edge_p["dst_kif"])
        self.assertIsNotNone(edge_p["dst_qint"])
        self.assertIsNone(edge_p["src_kif"])
        self.assertEqual(edge_p["bitwidth_dst"], 4.0)
        self.assertEqual(edge_p["element_bitwidth_bits"], 4.0)
        self.assertEqual(edge_p["tensor_width_bits"], 16.0)
        self.assertEqual(edge_p["volume_bits"], 16.0)


if __name__ == "__main__":
    unittest.main()
