from heterograph import HGraph


# Sched-IR answers *how does the model run on the hardware* — it lowers NN-IR
# into primitive operator blocks that a compiler (da4ml / hls4ml / ...) can
# accept, and then assigns each block to a kernel instance and a time slot.
#
# The IR is built in two stages:
#
#   1. Decomposition   NN-IR → unscheduled Sched-IR (compiler-independent).
#      Every NN-IR vertex is expanded into one or more primitive vertices.
#      Nothing is bound to a kernel, folded, or scheduled yet.
#
#   2. Scheduling      Unscheduled Sched-IR + Resource YAML → scheduled IR.
#      Runs BIND → FOLD → SCHEDULE → INSERT INFRASTRUCTURE → COST ROLL-UP as
#      described in `sched-IR Plan.md`.
#
# The six primitives currently supported (see `sched-IR Plan.md`):
#
#   dense        — constant MVM / MMM (+ optional fused BN and activation)
#   reduce       — reduction over one or more axes (sum / mean / max)
#   elementwise  — unary or binary elementwise (add / mul / sub / ...)
#   activation   — standalone nonlinearity (relu / sigmoid / tanh / softmax)
#   buffer       — (scheduler-inserted) storage for lifetime management
#   mux          — (scheduler-inserted) input selection for folded kernels
#
# Buffer and mux are *not* model operations: they are inserted by the scheduler
# during the INSERT INFRASTRUCTURE phase when a folded schedule requires them.


# --------------------------------------------------------------------------- #
# Allowed enums (documented here, enforced by convention in the builder)
# --------------------------------------------------------------------------- #

OP_PRIMITIVES = (
    "dense",         # constant matrix-vector/matrix-matrix multiply (+BN, +act)
    "reduce",        # reduction along one or more axes
    "elementwise",   # unary/binary elementwise op
    "activation",    # standalone nonlinearity
    "buffer",        # inserted by scheduler for lifetime management
    "mux",           # inserted by scheduler for folded input selection
)

REDUCE_MODES = ("sum", "mean", "max")

ELEMENTWISE_OPS = ("add", "sub", "mul", "div", "neg", "abs")

ACTIVATION_FUNCS = ("relu", "sigmoid", "tanh", "softmax", "linear")

BUFFER_KINDS = ("register", "fifo", "bram")

EDGE_KINDS = (
    "data",            # normal dataflow dependency
    "control",         # synchronisation (e.g. accumulator reset, barrier)
    "reuse_ordering",  # serialisation between fold iterations on the same instance
)


def vinit_sched(g, vx):
    g.pmap[vx] = {
        # ---------------------------------------------------------------- #
        # Provenance — link back to the NN-IR vertex that produced this one
        # ---------------------------------------------------------------- #
        "nn_layer_idx":       None,   # model.layers index of the source NN-IR vertex
        "nn_layer_name":      None,   # keras layer name, for readability
        "nn_op_kind":         None,   # NN-IR op_kind, e.g. 'einsum_dense_bn', 'qsum', 'qadd'
        "decomp_index":       None,   # position within this layer's decomposition
        "inserted_by":        None,   # 'decomposer' | 'scheduler' (buffer/mux come from the scheduler)

        # ---------------------------------------------------------------- #
        # Primitive operation
        # ---------------------------------------------------------------- #
        "op":                 None,   # one of OP_PRIMITIVES
        "op_params":          None,   # dict of op-specific params (see below)
        #
        # Recommended shape of op_params per primitive:
        #
        # dense:
        #   {
        #     'kernel_shape':  (In, Out),   # or (M, K, N) for batched einsum
        #     'equation':      'bnc,cC->bnC' | None,
        #     'in_bw':         float,       # input quantizer avg bitwidth
        #     'kq_bw':         float,       # kernel quantizer avg bitwidth
        #     'out_bw':        float,       # estimated output bitwidth (after accumulation)
        #     'sparsity':      float,       # fraction of zero weights
        #     'activation':    'relu' | None,
        #     'has_bn':        bool,
        #     'has_bias':      bool,
        #   }
        #
        # reduce:
        #   {
        #     'mode':          'sum' | 'mean' | 'max',
        #     'axes':          [int, ...],
        #     'in_shape':      tuple,
        #     'in_bw':         float,
        #     'out_bw':        float,   # typically in_bw + ceil(log2(reduction_width))
        #     'reduction_width': int,   # product of reduced dims
        #     'keepdims':      bool,
        #     'scale':         float | None,   # e.g. 1/N for mean
        #   }
        #
        # elementwise:
        #   {
        #     'op':            'add' | 'sub' | 'mul' | ...,
        #     'in_shapes':     [tuple, tuple],   # per-port (may differ under broadcast)
        #     'in_bws':        [float, float],
        #     'out_bw':        float,            # add/sub: max(in_bws)+1, etc.
        #     'broadcast':     {port_idx: [axes]} | None,
        #   }
        #
        # activation:
        #   {
        #     'func':          'relu' | 'sigmoid' | ...,
        #     'in_shape':      tuple,
        #     'in_bw':         float,
        #     'out_bw':        float,
        #     'lut_entries':   int | None,       # for LUT-based activations
        #   }
        #
        # buffer (scheduler-inserted):
        #   {
        #     'width_bits':    int,
        #     'depth':         int,
        #     'total_bits':    int,   # width_bits * depth
        #   }
        #
        # mux (scheduler-inserted):
        #   {
        #     'n_inputs':      int,
        #     'width_bits':    int,
        #     'select_bits':   int,   # ceil(log2(n_inputs))
        #   }

        # ---------------------------------------------------------------- #
        # Kernel binding (Phase 1 — BIND)
        # ---------------------------------------------------------------- #
        "kernel_type":        None,   # name from resource YAML, e.g. 'da4ml_dense'
        "kernel_instance":    None,   # physical instance id (0, 1, 2, ...); two vertices
                                      # sharing the same (kernel_type, kernel_instance)
                                      # are reusing the same hardware at different times.
        "cost":               None,   # dict filled by kernel.cost_query(op_params):
                                      #   {
                                      #     'lut':            int,
                                      #     'ff':             int,
                                      #     'dsp':            int,
                                      #     'bram':           int,
                                      #     'latency_cycles': int,   # L (kernel pipeline depth)
                                      #     'ii':             int,   # = T (mirrors node.ii)
                                      #   }

        # ---------------------------------------------------------------- #
        # N–P–T timing model (populated by FOLD; read by SCHEDULE)
        # ---------------------------------------------------------------- #
        # Every compute vertex obeys the invariants
        #     T  == ceil(N / P)
        #     II == T
        #     latency_total == L + (T - 1)
        # so II and latency_total are never assigned independently of (N, P, L).
        "parallelism_N":      None,   # logical parallelism — size of the fold axis.
                                      # For vertices outside any fold group: 1.
        "lanes_P":            None,   # hardware parallel lanes (work per cycle).
                                      # For dense: number of physical kernel copies.
                                      # For reduce: spatial width of the reduction
                                      #             (P_reduce — elements consumed per cycle).
        "temporal_steps_T":   None,   # = ceil(N / P). Cycles to finish one batch of work.
        "pipeline_latency_L": None,   # intrinsic kernel pipeline depth (mirrors cost.latency_cycles).
        "elements_per_cycle": None,   # = P. Explicit for transport / bandwidth analyses.
        "ii":                 None,   # initiation interval = T.
        "latency_total":      None,   # total cycles from accepting inputs to final output
                                      # = L + (T - 1).

        # ---------------------------------------------------------------- #
        # Folding (Phase 2 — FOLD)
        # ---------------------------------------------------------------- #
        "fold_axes":          None,   # list of *tensor dimension indices* this op
                                      # can be folded over (batch dim included).
                                      # e.g. [1] for a JEDI-linear (B,N,C) dense.
                                      # Reductions that *consume* an axis list it
                                      # here too, and then switch to temporal
                                      # accumulation during FOLD.
        "fold_factor":        None,   # Legacy alias for ``temporal_steps_T`` — kept for
                                      # back-compat with tooling that hasn't migrated yet.
                                      # Semantically identical to T (1 = fully spatial,
                                      # N = fully temporal).
        "fold_iteration":     None,   # this vertex represents iteration t of the fold
                                      # (0 <= t < temporal_steps_T). None until expansion.
        "fold_group":         None,   # group id shared by vertices that *must* fold
                                      # together (symmetric-fold constraint, e.g. the
                                      # self-path dense and the QAdd that consumes it).
        "reuse_group":        None,   # group id shared by vertices mapped to the same
                                      # kernel instance across time (≈ kernel_instance,
                                      # but explicit for visualisation / analysis).
        "physical_instances": None,   # how many physical copies of this kernel exist
                                      # in hardware after folding:
                                      #   - vertices outside any fold group: 1
                                      #   - vertices inside a group: ceil(N / T) = P
                                      # Phase 5 ROLL-UP multiplies per-instance cost by
                                      # this value to get the total area contribution.
                                      # For reductions this is a legacy alias of lanes_P
                                      # (P_reduce), which is the spatial reduce-tree width,
                                      # not the number of independent trees.

        # Reduction-specific mode (set only when op == 'reduce'):
        "reduce_mode":        None,   # 'spatial'             — full tree of adders, unfolded
                                      # | 'temporal_accumulate' — pure accumulator (K = N)
                                      # | 'hybrid'             — spatial tree of width N/K
                                      #                          plus K-step accumulator

        # ---------------------------------------------------------------- #
        # Schedule (Phase 3 — SCHEDULE)
        # ---------------------------------------------------------------- #
        "t_ready":            None,   # cycle at which all inputs are available
        "t_start":            None,   # cycle this op begins
        "t_end":              None,   # cycle this op completes
                                      # t_end == t_start + latency_total
                                      #       == t_start + L + (T - 1)
        "critical_path":      False,  # set by the scheduler if this vertex lies on
                                      # the critical path of the final schedule.
    }


def einit_sched(g, e):
    g.pmap[e] = {
        # ---------------------------------------------------------------- #
        # Data description
        # ---------------------------------------------------------------- #
        "tensor_shape":       None,   # shape of the data flowing on this edge
        "bitwidth":           None,   # wire bitwidth
        "volume_bits":        None,   # prod(non-batch dims) * bitwidth
        "edge_kind":          "data", # one of EDGE_KINDS

        # ---------------------------------------------------------------- #
        # Timing (Phase 3 — SCHEDULE)
        # ---------------------------------------------------------------- #
        "t_produce":          None,   # cycle at which source emits this value
        "t_consume":          None,   # cycle at which sink reads this value
        "t_producer":         None,   # alias of t_produce for clearer edge naming
        "t_consumer":         None,   # alias of t_consume for clearer edge naming
        "lifetime":           None,   # t_consume - t_produce
                                      # lifetime > 1 ⇒ storage required between cycles
        "fold_iteration":     None,   # when source/sink are folded, which iteration of
                                      # the fold this edge instance carries

        # ---------------------------------------------------------------- #
        # Buffering (Phase 4 — INSERT INFRASTRUCTURE) - ARCH IR ONLY
        # ---------------------------------------------------------------- #
        # Populated on edges that the scheduler decides need explicit storage.
        # The scheduler may replace such an edge with an inserted `buffer`
        # vertex; these fields still describe the *intent* for analysis passes.
        "needs_buffer":       False,
        "buffer_kind":        None,   # 'register' | 'fifo' | 'bram'
        "buffer_depth":       None,   # entries
        "buffer_width_bits":  None,   # bits per entry  (= bitwidth * prod(non-batch dims))
        "buffer_total_bits":  None,   # depth * width_bits

        # ---------------------------------------------------------------- #
        # Mux (Phase 4) — only set if this edge feeds an inserted mux vertex  - ARCH IR ONLY
        # ---------------------------------------------------------------- #
        "mux_select":         None,   # which select line this edge corresponds to
    }


def ginit_sched(g):
    # ---------------------------------------------------------------- #
    # Identity / provenance
    # ---------------------------------------------------------------- #
    g.pmap["name"] = None
    g.pmap["source_nn_ir"] = None          # handle / path to the NN-IR this was lowered from
    g.pmap["resource_yaml"] = None         # path to the kernel resource description

    # ---------------------------------------------------------------- #
    # Target hardware
    # ---------------------------------------------------------------- #
    g.pmap["target_device"] = None         # e.g. 'VU13P'
    g.pmap["target_fmax"] = None           # Hz, e.g. 300e6
    g.pmap["target_slrs"] = None           # super-logic regions, e.g. 3

    # ---------------------------------------------------------------- #
    # Scheduling objective (drives Phase 2 / 3)
    # ---------------------------------------------------------------- #
    g.pmap["objective"] = None             # 'min_latency' | 'min_area' | 'pareto'
    g.pmap["area_budget"] = None           # dict {'lut': int, 'ff': int, 'bram': int, 'dsp': int}
    g.pmap["latency_budget"] = None        # cycles

    # ---------------------------------------------------------------- #
    # Fold plan (populated during Phase 2 — FOLD)
    # ---------------------------------------------------------------- #
    # A fold plan is a list of entries, each describing how a set of vertices
    # sharing a fold axis was folded:
    #   [{ 'axis': 'particle',
    #      'factor': 4,
    #      'members': [vx, vx, ...],
    #      'reductions_temporalised': [vx, ...] },
    #    ...]
    g.pmap["fold_plan"] = None

    # ---------------------------------------------------------------- #
    # Schedule summary (filled after Phase 3 / Phase 5 — COST ROLL-UP)
    # ---------------------------------------------------------------- #
    g.pmap["makespan"] = None              # total_cycles from the first to the last op
    g.pmap["initiation_interval"] = None   # II of the pipelined schedule
    g.pmap["pipeline_depth"] = None        # register stages on the critical path

    g.pmap["total_luts"] = None            # sum of kernel instance LUT costs
    g.pmap["total_ffs"] = None             # kernel FFs + inserted buffer FFs
    g.pmap["total_dsps"] = None
    g.pmap["total_brams"] = None

    g.pmap["kernel_utilization"] = None    # dict kernel_type → fraction of cycles active
    g.pmap["critical_path"] = None         # ordered list of vertex ids on the critical path
