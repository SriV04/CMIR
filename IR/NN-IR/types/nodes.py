from __future__ import annotations


def default_node_properties() -> dict:
    return {
        # --- Identity ---
        "layer_name": None,
        "layer_class": None,
        "layer_idx": None,

        # --- Operation semantics ---
        "op_kind": None,
        "equation": None,
        "activation": None,
        "kernel_shape": None,
        "has_bn": False,
        "bn_folded_into_qkernel": False,

        # --- Tensor geometry ---
        "in_shapes": None,
        "out_shapes": None,

        # --- Raw parameter values ---
        "kernel_values": None,
        "kernel_float_values": None,
        "bias_values": None,
        "batchnorm_values": None,

        # --- Quantized parameter values ---
        "qkernel_values": None,
        "qbias_values": None,
        "uses_qkernel": False,

        # --- Quantizer summaries ---
        "iq": None,
        "kq": None,
        "bq": None,
        "oq": None,
        "aq": None,

        # --- KIF / QInterval source of truth ---
        "iq_kif": None,
        "kq_kif": None,
        "bq_kif": None,
        "oq_kif": None,

        "iq_qint": None,
        "kq_qint": None,
        "bq_qint": None,
        "oq_qint": None,

        # --- Bitwidth summaries ---
        "iq_bw_avg": None,
        "iq_bw_max": None,
        "iq_bw_min": None,
        "iq_bw_shape": None,

        "kq_bw_avg": None,
        "kq_bw_max": None,
        "kq_bw_min": None,
        "kq_bw_shape": None,

        "bq_bw_avg": None,
        "bq_bw_max": None,
        "bq_bw_min": None,
        "bq_bw_shape": None,

        "oq_bw_avg": None,
        "oq_bw_max": None,
        "oq_bw_min": None,
        "oq_bw_shape": None,

        # --- Quantization behaviour ---
        "iq_overflow_mode": None,
        "iq_round_mode": None,
        "kq_overflow_mode": None,
        "kq_round_mode": None,
        "bq_overflow_mode": None,
        "bq_round_mode": None,
        "oq_overflow_mode": None,
        "oq_round_mode": None,

        # --- Structural quantization metadata ---
        "quantizer_granularity": None,
        "quantizer_place": None,
        "quantizer_source": None,

        # --- Weight statistics ---
        "kernel_sparsity": None,
        "kernel_nonzero_count": None,
        "kernel_zero_count": None,
        "kernel_unique_values": None,
        "kernel_unique_count": None,
        "kernel_value_histogram": None,
        "kernel_min": None,
        "kernel_max": None,
        "kernel_dtype": None,

        # --- Parameter count ---
        "num_params": None,

        # --- Legacy aliases ---
        "iq_bw": None,
        "kq_bw": None,
        "bq_bw": None,
        "iq_bw_per_param": None,
        "kq_bw_per_param": None,
        "sparsity": None,
        "weights": None,
        "biases": None,
    }
