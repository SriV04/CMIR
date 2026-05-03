from __future__ import annotations


OP_PRIMITIVES = (
    "dense",
    "reduce",
    "elementwise",
    "activation",
    "buffer",
    "mux",
)

REDUCE_OPS = ("sum", "mean", "max", "min")
REDUCE_MODES = REDUCE_OPS  # legacy alias
REDUCE_IMPL_MODES = ("spatial", "temporal_accumulate", "hybrid")

ELEMENTWISE_OPS = (
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs",
    "max",
    "min",
)

ACTIVATION_FUNCS = (
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "linear",
)

BUFFER_KINDS = (
    "register",
    "fifo",
    "bram",
    "uram",
)

MUX_KINDS = (
    "combinational",
    "registered",
)

EDGE_KINDS = (
    "data",
    "control",
    "reuse_ordering",
    "cast",
    "buffered",
    "mux_input",
)

PRECISION_SOURCES = (
    "hgq",
    "da4ml",
    "derived",
    "inherited",
    "promoted",
    "fallback",
    "unknown",
)

INSERTED_BY = (
    "decomposer",
    "scheduler",
)

