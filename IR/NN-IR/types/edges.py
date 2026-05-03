from __future__ import annotations


def default_edge_properties() -> dict:
    return {
        "tensor_shape": None,

        # --- Producer / consumer precision ---
        "src_qint": None,
        "src_kif": None,
        "src_bitwidth_bits": None,
        "dst_qint": None,
        "dst_kif": None,
        "dst_bitwidth_bits": None,

        # --- Tensor-wide width ---
        "element_bitwidth_bits": None,
        "element_kif": None,
        "element_qint": None,
        "tensor_width_bits": None,
        "volume_bits_exact": None,

        # --- Quantization / cast semantics ---
        "has_quantization_boundary": False,
        "producer_quantizer": None,
        "consumer_quantizer": None,
        "needs_cast": False,
        "cast_mode": None,

        # --- Legacy aliases ---
        "bitwidth_src": None,
        "bitwidth_dst": None,
        "volume_bits": None,
    }
