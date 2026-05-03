from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np


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


fake_heterograph = ModuleType("heterograph")
fake_heterograph.HGraph = FakeHGraph
sys.modules.setdefault("heterograph", fake_heterograph)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


decomposer = _load("sched_decomposer_test", HERE / "decomposer.py")


class DecomposerMetadataTests(unittest.TestCase):
    def test_dense_lowering_copies_rich_precision_and_values(self):
        nn_g = FakeHGraph()
        nn_g.pmap["name"] = "fake_nn"

        src = nn_g.add_vx()
        dense = nn_g.add_vx()
        nn_g.vertices = [src, dense]

        nn_g.pmap[src] = {
            "layer_idx": 0,
            "layer_name": "src_dense",
            "op_kind": "dense",
            "in_shapes": [(None, 4)],
            "out_shapes": [(None, 4)],
            "kernel_shape": (4, 4),
            "qkernel_values": np.eye(4, dtype=np.float32),
            "kernel_values": np.eye(4, dtype=np.float32),
            "weights": np.eye(4, dtype=np.float32),
            "iq_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "iq_kif": {"bits": 4, "k": 1, "i": 2, "f": 1},
            "iq": {"class_name": "FakeIQ"},
            "kq_qint": {"min": -1.0, "max": 1.0, "step": 1.0},
            "kq_kif": {"bits": 2, "k": 1, "i": 1, "f": 0},
            "kq": {"class_name": "FakeKQ"},
            "iq_bw_avg": 4.0,
            "iq_bw": 4.0,
            "kq_bw_avg": 2.0,
            "kq_bw": 2.0,
            "kernel_sparsity": 0.0,
            "sparsity": 0.0,
            "activation": None,
        }

        nn_g.pmap[dense] = {
            "layer_idx": 1,
            "layer_name": "dense_1",
            "op_kind": "einsum_dense_bn",
            "equation": "bnc,cC->bnC",
            "in_shapes": [(None, 4)],
            "out_shapes": [(None, 2)],
            "kernel_shape": (4, 2),
            "qkernel_values": np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, -2.0]], dtype=np.float32),
            "kernel_values": np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, -2.0]], dtype=np.float32),
            "kernel_float_values": np.array([[0.5, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, -1.0]], dtype=np.float32),
            "bias_values": np.array([0.0, 0.5], dtype=np.float32),
            "qbias_values": np.array([0.0, 1.0], dtype=np.float32),
            "uses_qkernel": True,
            "iq": {"class_name": "FakeIQ"},
            "kq": {"class_name": "FakeKQ"},
            "bq": {"class_name": "FakeBQ"},
            "oq": {"class_name": "FakeOQ"},
            "aq": {"class_name": "FakeAQ", "qint": {"min": 0.0, "max": 1.0, "step": 0.25}, "kif": {"bits": 3}},
            "iq_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "iq_kif": {"bits": 4, "k": 1, "i": 2, "f": 1},
            "kq_qint": {"min": -1.0, "max": 1.0, "step": 1.0},
            "kq_kif": {"bits": 2, "k": 1, "i": 1, "f": 0},
            "bq_qint": {"min": -2.0, "max": 1.0, "step": 1.0},
            "bq_kif": {"bits": 3, "k": 1, "i": 2, "f": 0},
            "oq_qint": {"min": -4.0, "max": 3.0, "step": 1.0},
            "oq_kif": {"bits": 4, "k": 1, "i": 3, "f": 0},
            "iq_bw_avg": 4.0,
            "iq_bw": 4.0,
            "kq_bw_avg": 2.0,
            "kq_bw": 2.0,
            "bq_bw_avg": 3.0,
            "kernel_sparsity": 0.5,
            "kernel_nonzero_count": 4,
            "kernel_zero_count": 4,
            "kernel_unique_values": np.array([-2.0, 0.0, 1.0, 2.0], dtype=np.float32),
            "kernel_unique_count": 4,
            "kernel_min": -2.0,
            "kernel_max": 2.0,
            "kernel_dtype": "float32",
            "has_bn": True,
            "bn_folded_into_qkernel": True,
            "activation": "relu",
            "sparsity": 0.5,
            "weights": np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, -2.0]], dtype=np.float32),
            "biases": np.array([0.0, 0.5], dtype=np.float32),
        }

        edge = (src, dense)
        nn_g.edges = [edge]
        nn_g.pmap[edge] = {
            "tensor_shape": (None, 4),
            "src_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "src_kif": {"bits": 4, "k": 1, "i": 2, "f": 1},
            "src_bitwidth_bits": 4.0,
            "dst_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "dst_kif": {"bits": 4, "k": 1, "i": 2, "f": 1},
            "dst_bitwidth_bits": 4.0,
            "element_bitwidth_bits": 4.0,
            "element_kif": {"bits": 4, "k": 1, "i": 2, "f": 1},
            "element_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "tensor_width_bits": 16.0,
            "volume_bits_exact": 16.0,
            "volume_bits": 16.0,
            "has_quantization_boundary": False,
            "producer_quantizer": {"class_name": "FakeSrcQ"},
            "consumer_quantizer": {"class_name": "FakeDstQ"},
            "needs_cast": False,
            "cast_mode": None,
            "bitwidth_src": 4.0,
            "bitwidth_dst": 4.0,
        }

        sg = decomposer.decompose_nn_to_sched(nn_g, name="fake_sched")
        dense_vx = next(vx for vx in sg.vertices if sg.pmap[vx]["nn_layer_name"] == "dense_1")
        dense_p = sg.pmap[dense_vx]
        params = dense_p["op_params"]
        sched_edge = sg.pmap[(0, 1)]

        np.testing.assert_array_equal(params["qkernel_values"], nn_g.pmap[dense]["qkernel_values"])
        np.testing.assert_array_equal(params["kernel_values"], nn_g.pmap[dense]["qkernel_values"])
        np.testing.assert_array_equal(params["kernel_float_values"], nn_g.pmap[dense]["kernel_float_values"])
        self.assertEqual(params["input_quantizer"]["class_name"], "FakeIQ")
        self.assertEqual(params["kernel_quantizer"]["class_name"], "FakeKQ")
        self.assertEqual(params["bias_quantizer"]["class_name"], "FakeBQ")
        self.assertEqual(params["output_quantizer"]["class_name"], "FakeOQ")
        self.assertEqual(params["input_kif"]["bits"], 4)
        self.assertEqual(params["kernel_kif"]["bits"], 2)
        self.assertEqual(params["bias_kif"]["bits"], 3)
        self.assertEqual(params["output_kif"]["bits"], 4)
        self.assertEqual(params["in_bw"], 4.0)
        self.assertEqual(params["kq_bw"], 2.0)
        self.assertEqual(params["sparsity"], 0.5)
        self.assertEqual(dense_p["input_kifs"][0]["bits"], 4)
        self.assertEqual(dense_p["output_kifs"][0]["bits"], 4)
        self.assertEqual(dense_p["precision_source"], "hgq")
        self.assertEqual(sched_edge["bitwidth"], 4.0)
        self.assertEqual(sched_edge["tensor_width_bits"], 16.0)
        self.assertEqual(sched_edge["volume_bits"], 16.0)
        self.assertEqual(sched_edge["qint"]["step"], 0.5)
        self.assertEqual(sched_edge["kif"]["bits"], 4)

    def test_elementwise_inputs_are_refreshed_from_incoming_edges(self):
        nn_g = FakeHGraph()
        nn_g.pmap["name"] = "fake_nn"

        a = nn_g.add_vx()
        b = nn_g.add_vx()
        add = nn_g.add_vx()
        nn_g.vertices = [a, b, add]

        common_dense = {
            "op_kind": "dense",
            "in_shapes": [(None, 4)],
            "out_shapes": [(None, 4)],
            "kernel_shape": (4, 4),
            "qkernel_values": np.eye(4, dtype=np.float32),
            "kernel_values": np.eye(4, dtype=np.float32),
            "weights": np.eye(4, dtype=np.float32),
            "iq_bw_avg": 4.0,
            "iq_bw": 4.0,
            "kq_bw_avg": 2.0,
            "kq_bw": 2.0,
            "kernel_sparsity": 0.0,
            "sparsity": 0.0,
        }
        nn_g.pmap[a] = {"layer_idx": 0, "layer_name": "dense_a", "src_marker": "a", **common_dense}
        nn_g.pmap[b] = {"layer_idx": 1, "layer_name": "dense_b", "src_marker": "b", **common_dense}
        nn_g.pmap[add] = {
            "layer_idx": 2,
            "layer_name": "q_add",
            "op_kind": "qadd",
            "in_shapes": [(None, 4), (None, 4)],
            "out_shapes": [(None, 4)],
            "iq_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "iq_kif": {"bits": 4},
            "iq_bw_avg": 4.0,
            "iq_bw": 4.0,
        }

        e0 = (a, add)
        e1 = (b, add)
        nn_g.edges = [e0, e1]
        nn_g.pmap[e0] = {
            "tensor_shape": (None, 4),
            "src_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "src_kif": {"bits": 4},
            "src_bitwidth_bits": 4.0,
            "tensor_width_bits": 16.0,
            "volume_bits_exact": 16.0,
            "volume_bits": 16.0,
            "bitwidth_src": 4.0,
        }
        nn_g.pmap[e1] = {
            "tensor_shape": (None, 4),
            "src_qint": {"min": -2.0, "max": 1.0, "step": 1.0},
            "src_kif": {"bits": 3},
            "src_bitwidth_bits": 3.0,
            "tensor_width_bits": 12.0,
            "volume_bits_exact": 12.0,
            "volume_bits": 12.0,
            "bitwidth_src": 3.0,
        }

        sg = decomposer.decompose_nn_to_sched(nn_g, name="fake_sched")
        add_vx = next(vx for vx in sg.vertices if sg.pmap[vx]["nn_layer_name"] == "q_add")
        add_p = sg.pmap[add_vx]
        params = add_p["op_params"]

        self.assertEqual([k["bits"] for k in add_p["input_kifs"]], [4, 3])
        self.assertEqual([q["step"] for q in add_p["input_qints"]], [0.5, 1.0])
        self.assertEqual(add_p["input_tensor_width_bits"], [16.0, 12.0])
        self.assertEqual([k["bits"] for k in params["input_kifs"]], [4, 3])
        self.assertEqual(params["n_inputs"], 2)
        self.assertEqual(params["op"], "add")
        self.assertEqual(params["elementwise_op"], "add")


if __name__ == "__main__":
    unittest.main()
