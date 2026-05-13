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

    def add_edge(self, src, dst):
        edge = (src, dst)
        if edge in self.edges:
            return []
        self.edges.append(edge)
        if self._einit is not None:
            self._einit(self, edge)
        return [edge]

    def rm_edge(self, edge):
        self.edges.remove(edge)
        self.pmap.pop(edge, None)


fake_heterograph = ModuleType("heterograph")
fake_heterograph.HGraph = FakeHGraph
sys.modules.setdefault("heterograph", fake_heterograph)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


schema = _load("sched_schema_infra_precision_test", HERE / "schema.py")
infra = _load("sched_infra_precision_test", HERE / "infrastructure.py")


class InfrastructurePrecisionTests(unittest.TestCase):
    def _resource_yaml(self) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write("fpga:\n  device: VU13P\n")
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def _graph(self) -> FakeHGraph:
        g = FakeHGraph(vinit=schema.vinit_sched, einit=schema.einit_sched, ginit=schema.ginit_sched)
        g.pmap["resource_yaml"] = str(self._resource_yaml())
        return g

    def test_edge_width_prefers_exact_tensor_width(self):
        self.assertEqual(
            infra._edge_width_bits(
                {
                    "tensor_shape": (None, 8),
                    "bitwidth": 4,
                    "tensor_width_bits": 7,
                    "volume_bits_exact": 7,
                }
            ),
            7,
        )

    def test_buffer_insertion_preserves_precision_payload_on_both_edges(self):
        g = self._graph()
        src = g.add_vx()
        dst = g.add_vx()
        g.add_edge(src, dst)

        g.pmap[src]["op"] = "dense"
        g.pmap[src]["nn_layer_name"] = "src"
        g.pmap[src]["t_end"] = 10
        g.pmap[src]["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}
        g.pmap[dst]["op"] = "dense"
        g.pmap[dst]["nn_layer_name"] = "dst"
        g.pmap[dst]["cost"] = {"lut": 1, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1, "ii": 1}

        ep = g.pmap[(src, dst)]
        ep.update(
            {
                "tensor_shape": (None, 2),
                "qint": [{"step": 0.25}],
                "kif": [{"bits": 3}, {"bits": 4}],
                "src_qint": [{"step": 0.25}],
                "src_kif": [{"bits": 3}, {"bits": 4}],
                "src_bitwidth_bits": 4,
                "dst_qint": {"step": 0.5},
                "dst_kif": {"bits": 4},
                "element_qint": [{"step": 0.25}],
                "element_kif": [{"bits": 3}, {"bits": 4}],
                "element_bitwidth_bits": 4,
                "tensor_width_bits": 7,
                "volume_bits_exact": 7,
                "volume_bits": 7,
                "bitwidth": 4,
                "has_quantization_boundary": True,
                "needs_cast": True,
                "cast_mode": "producer_to_consumer_quantizer",
                "t_produce": 10,
                "t_consume": 15,
                "lifetime": 5,
            }
        )

        infra.insert_buffers(g)
        buf = next(vx for vx in g.vertices if g.pmap[vx].get("op") == "buffer")

        for edge in ((src, buf), (buf, dst)):
            new_ep = g.pmap[edge]
            self.assertEqual(new_ep["tensor_width_bits"], 7)
            self.assertEqual(new_ep["volume_bits_exact"], 7)
            self.assertEqual(new_ep["src_kif"], ep["src_kif"])
            self.assertEqual(new_ep["dst_kif"], ep["dst_kif"])
            self.assertEqual(new_ep["element_bitwidth_bits"], 4)
            self.assertTrue(new_ep["needs_cast"])
            self.assertTrue(new_ep["has_quantization_boundary"])
            self.assertEqual(new_ep["cast_mode"], "producer_to_consumer_quantizer")

        self.assertEqual(g.pmap[buf]["op_params"]["width_bits"], 7)


if __name__ == "__main__":
    unittest.main()
