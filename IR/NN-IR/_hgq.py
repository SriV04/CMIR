from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_nn_ir_hgq_extractor",
        here / "hgq2" / "hgq_extractor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_impl = _load()

safe_array = _impl.safe_array
safe_get_config = _impl.safe_get_config
extract_quantizer_variables = _impl.extract_quantizer_variables
extract_quantizer_modes = _impl.extract_quantizer_modes
extract_kif = _impl.extract_kif
kif_to_qint = _impl.kif_to_qint
quantizer_summary = _impl.quantizer_summary
extract_all_quantizers = _impl.extract_all_quantizers
find_output_quantizer = _impl.find_output_quantizer
find_activation_quantizer = _impl.find_activation_quantizer
extract_layer_values = _impl.extract_layer_values
weight_stats = _impl.weight_stats
bitwidth_from_kif = _impl.bitwidth_from_kif
bw_array = _impl.bw_array
avg_bw = _impl.avg_bw
max_bw = _impl.max_bw
min_bw = _impl.min_bw
sparsity = lambda kernel, tol=1e-12: _impl.weight_stats(kernel, include_histogram=False).get("sparsity")
