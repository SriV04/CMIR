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


fake_heterograph = ModuleType("heterograph")
fake_heterograph.HGraph = FakeHGraph
sys.modules.setdefault("heterograph", fake_heterograph)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


schema = _load("sched_schema_precision_test", HERE / "schema.py")
precision = _load("sched_precision_test", HERE / "precision.py")


class PrecisionPropagationTests(unittest.TestCase):
    def _graph(self) -> FakeHGraph:
        return FakeHGraph(vinit=schema.vinit_sched, einit=schema.einit_sched, ginit=schema.ginit_sched)

    def test_bound_output_precision_is_copied_to_edges_and_consumer_inputs(self):
        g = self._graph()
        src = g.add_vx()
        dst = g.add_vx()
        g.add_edge(src, dst)

        src_p = g.pmap[src]
        src_p["op"] = "dense"
        src_p["nn_layer_name"] = "dense_src"
        src_p["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}
        src_p["output_qints"] = [
            {"min": -1.0, "max": 1.0, "step": 0.25},
            {"min": -2.0, "max": 2.0, "step": 0.5},
        ]
        src_p["output_kifs"] = [
            {"k": True, "i": 2, "f": 0, "bits": 3},
            {"k": True, "i": 3, "f": 0, "bits": 4},
        ]
        src_p["output_tensor_width_bits"] = 7
        src_p["precision_source"] = "da4ml"

        dst_p = g.pmap[dst]
        dst_p["op"] = "dense"
        dst_p["nn_layer_name"] = "dense_dst"
        dst_p["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}
        dst_p["output_kifs"] = [{"bits": 5}]
        dst_p["op_params"] = {}

        ep = g.pmap[(src, dst)]
        ep["tensor_shape"] = (None, 2)
        ep["dst_qint"] = {"min": -1.0, "max": 1.0, "step": 0.125}
        ep["dst_kif"] = {"k": True, "i": 1, "f": 2, "bits": 4}

        before_vertices = list(g.vertices)
        before_edges = list(g.edges)
        precision.propagate_precision(g)

        self.assertEqual(g.vertices, before_vertices)
        self.assertEqual(g.edges, before_edges)
        self.assertEqual(ep["src_qint"], src_p["output_qints"])
        self.assertEqual(ep["src_kif"], src_p["output_kifs"])
        self.assertEqual(ep["qint"], src_p["output_qints"])
        self.assertEqual(ep["kif"], src_p["output_kifs"])
        self.assertEqual(ep["src_bitwidth_bits"], 4)
        self.assertEqual(ep["element_bitwidth_bits"], 4)
        self.assertEqual(ep["bitwidth"], 4)
        self.assertEqual(ep["tensor_width_bits"], 7)
        self.assertEqual(ep["volume_bits_exact"], 7)
        self.assertEqual(ep["volume_bits"], 7)
        self.assertTrue(ep["needs_cast"])
        self.assertTrue(ep["has_quantization_boundary"])
        self.assertEqual(ep["cast_mode"], "producer_to_consumer_quantizer")
        self.assertEqual(dst_p["input_qints"], [ep["dst_qint"]])
        self.assertEqual(dst_p["input_kifs"], [ep["dst_kif"]])
        self.assertEqual(dst_p["op_params"]["input_kif"], ep["dst_kif"])
        self.assertTrue(g.pmap["precision_propagated"])

    def test_falls_back_to_op_params_output_precision_and_validates_elementwise_arity(self):
        g = self._graph()
        a = g.add_vx()
        b = g.add_vx()
        add = g.add_vx()
        g.add_edge(a, add)
        g.add_edge(b, add)

        for vx, bits in ((a, 4), (b, 3)):
            p = g.pmap[vx]
            p["op"] = "dense"
            p["nn_layer_name"] = f"dense_{bits}"
            p["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}
            p["op_params"] = {
                "output_qint": {"min": -1.0, "max": 1.0, "step": 2 ** -bits},
                "output_kif": {"bits": bits},
            }

        add_p = g.pmap[add]
        add_p["op"] = "elementwise"
        add_p["nn_layer_name"] = "add"
        add_p["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}
        add_p["op_params"] = {"op": "add", "output_kif": {"bits": 5}}

        precision.propagate_precision(g)

        self.assertEqual([items[0]["bits"] for items in add_p["input_kifs"]], [4, 3])
        self.assertEqual([items[0]["bits"] for items in add_p["op_params"]["input_kifs"]], [4, 3])
        self.assertEqual(g.pmap[(a, add)]["tensor_width_bits"], 4)
        self.assertEqual(g.pmap[(b, add)]["tensor_width_bits"], 3)

    def test_legacy_out_bw_fallback_marks_incomplete_kif(self):
        g = self._graph()
        src = g.add_vx()
        dst = g.add_vx()
        g.add_edge(src, dst)

        src_p = g.pmap[src]
        src_p["op"] = "dense"
        src_p["nn_layer_name"] = "legacy_dense"
        src_p["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}
        src_p["op_params"] = {"out_bw": 6}
        g.pmap[(src, dst)]["tensor_shape"] = (None, 1)

        precision.propagate_precision(g)

        fallback = g.pmap[(src, dst)]["src_kif"][0]
        self.assertEqual(
            fallback,
            {"k": None, "i": None, "f": None, "bits": 6, "source": "legacy_out_bw"},
        )

    def test_broadcast_equivalent_kifs_do_not_mark_cast(self):
        g = self._graph()
        src = g.add_vx()
        dst = g.add_vx()
        g.add_edge(src, dst)

        kif = {"k": True, "i": 2, "f": 1, "bits": 4}
        src_p = g.pmap[src]
        src_p["op"] = "dense"
        src_p["nn_layer_name"] = "producer"
        src_p["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}
        src_p["output_kifs"] = kif
        src_p["output_tensor_width_bits"] = 8

        ep = g.pmap[(src, dst)]
        ep["tensor_shape"] = (None, 2)
        ep["dst_kif"] = [kif, kif]

        precision.propagate_precision(g)

        self.assertFalse(ep["needs_cast"])
        self.assertFalse(ep["has_quantization_boundary"])

    def test_validate_warns_when_kif_list_does_not_match_edge_shape_volume(self):
        g = self._graph()
        src = g.add_vx()
        dst = g.add_vx()
        g.add_edge(src, dst)

        src_p = g.pmap[src]
        src_p["op"] = "dense"
        src_p["nn_layer_name"] = "producer"
        src_p["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}
        src_p["output_kifs"] = [
            {"k": True, "i": 1, "f": 0, "bits": 2},
            {"k": True, "i": 1, "f": 0, "bits": 2},
            {"k": True, "i": 1, "f": 0, "bits": 2},
        ]
        src_p["output_tensor_width_bits"] = 6
        g.pmap[(src, dst)]["tensor_shape"] = (None, 2)

        precision.propagate_precision(g)

        warnings = g.pmap["precision_warnings"] or []
        self.assertTrue(any("3 src_kif entries for tensor volume 2" in msg for msg in warnings))


if __name__ == "__main__":
    unittest.main()
