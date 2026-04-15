from heterograph import HGraph
from IPython.display import SVG, display

# ============================================================================
# IR SCHEMA DEFINITION
# ============================================================================
# These init hooks are called automatically by Heterograph whenever a vertex,
# edge, or graph is created. They ensure every element has a consistent
# property schema that analysis passes can rely on.

def vinit(g, vx):
    """
    Default properties for every new vertex in the IR.
    
    Called automatically by HGraph when add_vx() is invoked.
    Properties are organized by IR node type:
    - Common: present on all nodes
    - CMVM-specific: only meaningful for compute nodes
    - Transport-specific: only meaningful for data movement nodes
    - Control-specific: only meaningful for sync nodes
    - Analysis: filled by later analysis passes (Phase 5)
    """
    g.pmap[vx] = {
        # --- Common properties (all node types) ---
        'ntype':       None,   # 'cmvm' | 'compute' | 'transport' | 'control' | 'composite'
        'compute_cost':       None,   # adder count or EBOPs
        'transport_cost':     None,   # bits × replication (sum over incident edges)
        'dominance':          None,   # which cost dominates: 'compute' | 'transport' | 'balanced'
        'layer_name':  None,   # Original Keras layer header
        'op_kind':     None,   # Subtype: 'einsum_dense_bn', 'reduction', 'elementwise_add',
                               #          'broadcast', 'gather', 'merge', 'split', ...
        'layer_idx':   None,   # Index in model.layers
        'in_shapes':   None,   # List of input tensor shapes (one entry per input port)
        'out_shapes':  None,   # List of output tensor shapes

        # --- Decomposition / composite-node support ---
        # Some Keras-level ops (QAdd, QSum, QSum→Dense→QAdd) decompose into a
        # TRANSPORT phase + COMPUTE phase. Rather than splitting the vertex
        # eagerly, the IR records both phases on a single composite vertex so
        # later passes can choose to lower, fuse, or schedule them.
        'phases':          None,   # Ordered list of phase dicts, e.g.
                                   #   [{'kind':'transport','op':'broadcast', ...},
                                   #    {'kind':'compute',  'op':'elementwise_add', ...},
                                   #    {'kind':'transport','op':'forward', ...}]
                                   # Each phase carries its own shape, bitwidth,
                                   # cost, and schedule fields. None for atomic nodes.
        'is_composite':    False,  # True if this vertex bundles multiple phases.
        'lowered_from':    None,   # Original Keras op that was lowered ('QAdd', 'QSum', ...)

        # --- Macro-fusion grouping ---
        # Allows passes to mark a chain like QSum→EinsumDense3→QAdd as a single
        # transport macro-operation without removing the underlying vertices.
        'macro_id':        None,   # ID of the fusion group this node belongs to
        'macro_role':      None,   # 'head' | 'body' | 'tail' | None
        'macro_kind':      None,   # 'transport_macro' | 'compute_macro' | None

        # --- CMVM / compute-specific (ntype in {'cmvm', 'compute'}) ---
        'equation':       None,   # Einsum notation string, e.g. 'bnc,cC->bnC'
        'activation':     None,   # Activation function name, e.g. 'relu'
        'kernel_shape':   None,   # Weight matrix shape, e.g. (64, 64)
        'iq_bw':          None,   # Average input quantizer bitwidth (from HGQ)
        'kq_bw':          None,   # Average kernel quantizer bitwidth
        'bq_bw':          None,   # Average bias quantizer bitwidth
        'num_params':     None,   # Total parameter count
        'adder_count':    None,   # Physical adders required (e.g. 8 for QAdd if folded,
                                  # 512 if fully unrolled, 448 for 8→1 sum tree)
        'mult_count':     None,   # Physical multipliers required (CMVM)
        'da4ml_lut_count': None,  # FPGA LUT count (filled in Phase 4)
        'da4ml_latency':   None,  # Critical path depth (filled in Phase 4)

        # --- Bitwidth growth tracking (compute phase) ---
        # The two inputs of an elementwise op (e.g. QAdd) may come from
        # different quantizers, so per-input bitwidths must be tracked
        # separately. Output bitwidth grows: add → max(bw_a,bw_b)+1,
        # k-input sum tree → max(bw_i) + ceil(log2(k)).
        'bw_inputs':       None,  # List of per-input bitwidths (one per input port)
        'bw_output':       None,  # Bitwidth at the compute output
        'bw_growth_rule':  None,  # 'add' | 'sum_tree' | 'mac' | 'identity'
        'bw_growth':       None,  # int: bw_output - max(bw_inputs)

        # --- Schedule / register-folding choice ---
        # An asymmetric op (e.g. QAdd with one 8×64 and one 64 input) can be
        # implemented as 512 parallel adders, or as 8 adders folded over the
        # 64-channel axis with a register. This is an exploration knob the IR
        # exposes rather than commits to.
        'schedule_kind':   None,  # 'parallel' | 'serial' | 'register_folded' | 'tiled'
        'unroll_factor':   None,  # How many compute units in space (rest is in time)
        'fold_axis':       None,  # Axis being scheduled in time (e.g. channel=64)
        'register_count':  None,  # Pipeline registers needed for the chosen schedule
        'register_placement': None,  # 'pre_op' | 'post_op' | 'per_stage' | None

        # --- Transport-specific (ntype='transport' or transport phase) ---
        'transport_kind': None,   # 'reduction' | 'broadcast' | 'gather' | 'concat'
                                  # | 'forward' | 'activation'
        'reduction_axes': None,   # Axes reduced over (for QSum)
        'reduction_scale': None,  # Scale factor (for QSum, e.g. 0.125 = 1/N)
        'keepdims':       None,   # Whether reduction keeps dimensions
        'fanout_map':     None,   # Dict mapping each output dimension to its replication count
        'broadcast_axes': None,   # Per-input axes that get broadcast (e.g. {0: [particle_dim]})
        'broadcast_factor': None, # Replication count per broadcast axis (e.g. {particle_dim: 8})
        'wire_endpoints': None,   # Total physical wire endpoints after broadcast/gather
                                  # (e.g. 64 wires × 8 broadcast = 512 endpoints)

        # --- Reduction-tree topology (for QSum and similar fan-in ops) ---
        # Lets passes explore tree reshaping (binary vs k-ary, balanced vs
        # skewed) without rewriting the vertex.
        'tree_radix':       None, # Fan-in per stage (2 for binary, 4 for radix-4, ...)
        'tree_stages':      None, # Number of pipeline stages (e.g. log2(8)=3 for 8→1 binary)
        'tree_shape':       None, # 'binary_balanced' | 'kary_balanced' | 'linear' | 'skewed'
        'reduction_width':  None, # Fan-in count being reduced (e.g. 8 for particle reduction)

        # --- Control-specific (filled only for ntype='control') ---
        'control_kind':    None,  # 'split' | 'merge' | 'barrier'
        'num_inputs':      None,  # Number of inputs (for merge nodes)

        # --- Analysis annotations (filled by Phase 5 passes) ---
        'hotspot_score':   0.0,
        'partition_id':    -1,
        'sparsity':           None,   # fraction of zero weights
        'kq_bw_max':          None,   # max kernel bitwidth (determines worst-case wire width)
    }


def einit(g, e):
    """
    Default properties for every new edge in the IR.
    
    Edges represent data flow between nodes. The key properties are:
    - bitwidth: how many bits wide is the data on this connection
    - volume_bits: total bits transferred = product(shape) × bitwidth
    
    These are used by the dominance classifier to estimate transport cost.
    """
    g.pmap[e] = {
        # --- Bitwidth on the wire ---
        'bitwidth_input':  None,  # Bitwidth at the source-side of the edge
        'bitwidth_output': None,  # Bitwidth at the sink-side (after any requantization)
        'bw_growth':       None,  # int: bitwidth_output - bitwidth_input (>0 for adders/MACs)

        # --- Geometry ---
        'tensor_shape':    None,  # Logical shape of data flowing on this edge
        'volume_bits':     None,  # Total bits transferred per inference (shape × bitwidth)
        'wire_count':      None,  # Physical wire count = prod(shape) × bitwidth × replication

        # --- Transport semantics ---
        # An edge can be a 1:1 forward, a 1:N broadcast (one source fans out to
        # N sinks), or an N:1 gather (N sources converge to one sink).
        'edge_kind':          None,  # 'forward' | 'broadcast' | 'gather' | 'split'
        'replication_factor': None,  # 1:N — how many times the source value is replicated
        'convergence_factor': None,  # N:1 — how many sources converge here (merge/reduce)
        'broadcast_axis':     None,  # Which logical axis is broadcast over (e.g. particle_dim)
        'gather_axis':        None,  # Which logical axis is being gathered/reduced

        # --- Asymmetry marker for elementwise ops with mismatched input shapes ---
        # E.g. QAdd takes X[8,64] (512 wires) and d[64] (64 wires fanning out
        # to 8 endpoints). The IR records which input port is the "small" one.
        'is_asymmetric_input': False,
        'asymmetry_ratio':     None,  # ratio of larger:smaller wire count (e.g. 8 for QAdd)

        # --- Macro-fusion / scheduling metadata ---
        'macro_id':       None,   # Macro this edge belongs to (matches vertex macro_id)
        'is_internal':    False,  # True if both endpoints are inside the same macro
                                  # (such edges may be elided as physical wires)
        'register_stage': None,   # Pipeline stage index assigned by scheduler
    }


def ginit(g):
    """
    Graph-level properties — metadata about the model and IR.
    """
    g.pmap['name'] = None
    g.pmap['model_source'] = None      # 'keras_hgq' | 'hls4ml' | 'qonnx'
    g.pmap['n_particles'] = None       # Number of particles (N) - should go into a data field 
    g.pmap['n_features'] = None        # Number of input features
    g.pmap['n_classes'] = None         # Number of output classes

    g.pmap['pipeline_depth'] = None       # Estimated critical path depth
    g.pmap['target_fmax'] = None        # Total parameter count
    g.pmap['target_device'] = None      # Target FPGA device (e.g. 'Xilinx Alveo U280')

