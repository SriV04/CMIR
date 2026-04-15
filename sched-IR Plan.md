Sched-IR receives NN-IR as an input and utilises a scheduler engine to populate the schema elements. This IR operates on primitives that hls4ml/da4ml accepts as inputs. They are operator-level blocks translating layers into subtly lower level blocks. 

Currently Focusing on JEDI Linear we build 6 primitives 

```yaml
primitives:

  dense:
    description: "Constant matrix-vector or matrix-matrix multiply with optional fused BN and activation"
    what_compiler_receives:
      - kernel weights (or shape + bitwidths for estimation)
      - input shape and bitwidth
      - activation type
    what_compiler_produces:
      - a single synthesised block with known latency, LUT, DSP, FF cost
    examples:
      - QEinsumDenseBatchnorm in JEDI-linear
      - Dense layer in any MLP
      - Conv2D (im2col form) 
      - Q/K/V projections in attention

  reduce:
    description: "Reduction over one or more axes (sum, mean, max)"
    what_compiler_receives:
      - input shape, bitwidth
      - reduction axis
      - reduction mode (sum/mean/max)
    what_compiler_produces:
      - a tree or accumulator block, pipelined internally
    examples:
      - QSum (global average pooling over particles)
      - Attention softmax denominator
      - Global average pooling in CNNs

  elementwise:
    description: "Element-wise binary or unary operation"
    what_compiler_receives:
      - input shapes, bitwidths (possibly asymmetric due to broadcast)
      - operation (add/multiply/subtract)
      - broadcast semantics
    what_compiler_produces:
      - array of parallel arithmetic units
    examples:
      - QAdd (residual/broadcast addition)
      - Attention score scaling
      - Element-wise multiply in gating

  activation:
    description: "Nonlinear activation function"
    what_compiler_receives:
      - input shape, bitwidth
      - function type (relu/sigmoid/tanh/softmax)
    what_compiler_produces:
      - comparator array (relu) or LUT array (sigmoid/tanh)
    examples:
      - ReLU after dense
      - Softmax in classification head

  buffer:
    description: "Explicit storage inserted by scheduler for lifetime management"
    not_a_model_operation: true
    inserted_by: scheduler
    what_it_does: holds data between producer and consumer when folding
                  creates a temporal gap

  mux:
    description: "Input selection for folded kernels"
    not_a_model_operation: true
    inserted_by: scheduler
    what_it_does: selects which fold iteration's data enters a reused block
```

Note: there is no **CONV** and explicit **CMVM.** 

We also will provide a **Resource YAML** which defines the compilation engine we will be proceeding with - this will provide insights into the evaluation of different design spaces in this IR - latency costs, resource consumptions and unit requirements can be calculated by arithmetic on the translation of the primitives to the compilation engine (da4ml/hls4ml)  ****

```yaml
framework: da4ml

kernels:
  da4ml_dense:
    supported_ops: [dense]
    constraints:
      weight_source: constant    # da4ml only handles constant weights
    instances: unlimited
    cost_model: query            # means: call da4ml estimator with params
    estimator: "da4ml.estimate_cost(kernel_shape, in_bw, sparsity)"

  da4ml_reduce:
    supported_ops: [reduce]
    instances: unlimited
    cost_model: query
    estimator: "da4ml.estimate_reduce_cost(in_shape, in_bw, axis, mode)"

  da4ml_elementwise:
    supported_ops: [elementwise]
    instances: unlimited
    cost_model: query

  hls4ml_dense:
    supported_ops: [dense]
    constraints: {}    # hls4ml handles both constant and dynamic weights
    instances: unlimited
    cost_model: query
    estimator: "hls4ml.estimate_cost(...)"

  register_buffer:
    supported_ops: [buffer]
    instances: unlimited
    cost_model:
      ff: "width * depth"
      lut: 0

  bram_buffer:
    supported_ops: [buffer]
    constraints:
      min_depth: 64    # below this, use registers instead
    instances: limited
    max_instances: 2688
    cost_model:
      bram: "ceil(width * depth / 36864)"

  lut_mux:
    supported_ops: [mux]
    instances: unlimited
    cost_model:
      lut: "ceil(log2(n_inputs)) * bw"
```

The Reuse and Scheduling opportunities will then present themselves as 

1. **Instance Reuse across the Fold Axis (Bakhtiar Scheduling)** 
When multiple Sched-IR operations are identical in type and parameters, they can share a hardware block and utilise buffers and mux to schedule.
Eg. 8,64 → EinsumDense is 8 separate EinsumDense kernels with identical activation, quantisation so we can fold 
’any axis of the computation that replicates the same operation across independent data elements — particles in a GNN, spatial positions in a CNN, sequence positions in a Transformer, batch elements — is a fold axis where instance reuse applies.’
2. **Producer Consumer Scheduling** 
When vertex A produces data that vertex B consumes - scheduling to decide how much temporal overlap exists
Key: Reductions create synchronisation barriers, operations taht are element wise can pipeline 
3. **Asymmetric Folding across Layers** 
When folding 2 paths - the interactions path in JEDI Linear 
4. **Cross layer fusion (LAST)** 

### Flow of creating Sched-IR

### The Flow

**Step 1: Decomposition (NN-IR → Initial Sched-IR)**

The decomposition rules are not architecture-independent. They're operation-independent. The distinction matters. A decomposition rule says "a dense layer becomes a `dense` Sched-IR node" — that's true regardless of whether you're compiling for da4ml or hls4ml. The rule doesn't know or care what compiler will handle it. It only knows the mathematical operation and how to express it as primitives.

The initial Sched-IR is not fully unrolled. It's not folded either. It's unscheduled — it has no time slots, no instance assignments, no fold factors. It's a one-to-one translation of the NN-IR where each layer becomes one or more Sched-IR nodes with their `op` and `params` set but everything else blank.

Here's exactly what happens:

```
NN-IR (JEDI-linear, 8 particles, 16 features):

  Layer 0: Input                    shape=(1,8,3)
  Layer 1: QEinsumDenseBatchnorm    kernel=(3,64),   act=relu
  Layer 2: QSum                     axis=particle,   mode=mean
  Layer 3: QEinsumDenseBatchnorm    kernel=(64,64),  act=relu
  Layer 4: QEinsumDenseBatchnorm    kernel=(64,64),  act=relu
  Layer 5: QAdd                     broadcast over particle_dim
  Layer 6: QEinsumDenseBatchnorm    kernel=(64,64),  act=relu
  Layer 7: QSum                     axis=particle,   mode=mean
  Layer 8: QEinsumDenseBatchnorm    kernel=(64,64),  act=relu
  Layer 9: QEinsumDenseBatchnorm    kernel=(64,32),  act=relu
  Layer 10: QEinsumDenseBatchnorm   kernel=(32,16),  act=relu
  Layer 11: QEinsumDenseBatchnorm   kernel=(16,5),   act=softmax
```

The decomposition rules fire:

```yaml
rules:
  einsum_dense_bn:
    match: {op_kind: einsum_dense_bn}
    emit:
      - op: dense
        params_from: [kernel_shape, iq_bw, kq_bw, bq_bw, activation, sparsity]
        fold_axes: "{particle_axes_if_present}"

  qsum:
    match: {op_kind: qsum}
    emit:
      - op: reduce
        params_from: [in_shape, iq_bw, reduction_axes, mode, scale]
        fold_axes: null    # reduction consumes the fold axis

  qadd:
    match: {op_kind: qadd}
    emit:
      - op: elementwise
        params_from: [in_shapes, in_bws, broadcast_axes, op_type]
        fold_axes: "{broadcast_axis}"
```

The resulting initial Sched-IR:

```
v0:  op=dense,       params={kernel=(3,64), in_bw=4, ...},    fold_axes=[particle]
v1:  op=reduce,      params={axis=particle, mode=mean, ...},  fold_axes=null
v2:  op=dense,       params={kernel=(64,64), ...},            fold_axes=[particle]
v3:  op=dense,       params={kernel=(64,64), ...},            fold_axes=null
v4:  op=elementwise, params={op=add, broadcast=particle},     fold_axes=[particle]
v5:  op=dense,       params={kernel=(64,64), ...},            fold_axes=[particle]
v6:  op=reduce,      params={axis=particle, mode=mean, ...},  fold_axes=null
v7:  op=dense,       params={kernel=(64,64), ...},            fold_axes=null
v8:  op=dense,       params={kernel=(64,32), ...},            fold_axes=null
v9:  op=dense,       params={kernel=(32,16), ...},            fold_axes=null
v10: op=dense,       params={kernel=(16,5), activation=softmax}, fold_axes=null

Edges: v0→v1, v0→v4 (skip), v1→v2, v2→v3, v3→v4, v4→v5, v5→v6, v6→v7, ...
```

Notice what this graph captures: each node knows which axes it could be folded over (`fold_axes`), but no folding decision has been made. The `fold`, `fold_iter`, `instance`, and `t` fields are all still null. The graph is a dependency graph with metadata — nothing more.

Also notice: layers 7–10 (the MLP classification head) have `fold_axes=null` because they operate on the single global vector after the second QSum. They can't be folded over the particle axis because that axis has been reduced away. This is automatically determined from the shapes — if the particle dimension isn't present in a layer's input, it has no fold axis.

---

**Step 2: The Scheduler Receives the Initial Sched-IR + Resource YAML**

The scheduler doesn't modify the decomposition. It doesn't add or remove model-derived nodes. It makes three decisions per node: fold factor, instance assignment, and time slot. And it inserts infrastructure nodes (buffers and muxes) where needed.

The resource YAML tells the scheduler what's available:

```yaml
# Resource YAML: da4ml on VU13P
framework: da4ml

kernels:
  da4ml_dense:
    supported_ops: [dense]
    constraints: {weight_source: constant}
    cost_query: "da4ml.estimate(params)"
    
  da4ml_reduce:
    supported_ops: [reduce]
    cost_query: "da4ml.estimate_reduce(params)"
    
  da4ml_elementwise:
    supported_ops: [elementwise]
    cost_query: "da4ml.estimate_elementwise(params)"

device:
  family: VU13P
  lut_budget: 1728000
  ff_budget: 3456000
  dsp_budget: 12288
  bram_budget: 2688
  slr_count: 3
  target_fmax: 300e6

fusion_rules:
  - pattern: [dense, activation]
    condition: "{activation in ['relu', 'linear']}"
```

A different resource YAML for hls4ml:

```yaml
framework: hls4ml

kernels:
  hls4ml_dense:
    supported_ops: [dense]
    constraints: {}    # handles both constant and dynamic weights
    cost_query: "hls4ml.estimate(params)"
    
  hls4ml_reduce:
    supported_ops: [reduce]
    cost_query: "hls4ml.estimate_reduce(params)"

device:
  family: VU13P
  # ... same device, different compiler
```

The same initial Sched-IR graph works with either YAML. The scheduler just gets different cost estimates back from the compiler queries.

---

**Step 3: The Scheduler's Algorithm**

```
INPUT:  Sched-IR graph G (unscheduled)
        Resource YAML R
        Objective: minimize latency subject to area ≤ budget
                   OR minimize area subject to latency ≤ budget
                   OR enumerate Pareto frontier

PHASE 1 — BIND:
  For each vertex v in G:
    Find kernels in R where v.op in kernel.supported_ops
    AND v.params satisfies kernel.constraints
    Assign v.kernel = best matching kernel
    Query compiler: v.cost = kernel.cost_query(v.params)
    v.cost now contains {lut, ff, dsp, bram, latency_cycles, ii}

PHASE 2 — FOLD:
  Identify fold groups: sets of vertices sharing the same fold_axes
  For JEDI-linear: {v0, v4, v5} share fold_axes=[particle]
                   {v1, v6} are reductions over particle (consume the axis)
                   {v2, v3, v7, v8, v9, v10} have fold_axes=null

  For each fold group, choose fold factor K:
    K=1: no folding, fully spatial
    K=N: fully folded, one instance reused N times
    1<K<N: partially folded
    
  Apply fold:
    For each vertex v with fold_axes and fold_factor K:
      v.fold = K
      total instances of v = ceil(original_parallelism / K)
      
  Enforce symmetric constraints:
    If v0 (EinsumDense1) is folded by K over particles,
    then v4 (QAdd), v5 (EinsumDense after QAdd) must also fold by K
    and v1 (QSum) must switch from spatial reduce to temporal accumulate

PHASE 3 — SCHEDULE:
  Topological sort of G
  For each vertex v in order:
    v.t = max(predecessor.t + predecessor.cost.latency_cycles
              for each predecessor of v)
    
    IF v.fold > 1:
      # This vertex runs K times on the same hardware
      # It occupies time slots [v.t, v.t + v.fold * v.cost.ii)
      v.t_end = v.t + v.fold * v.cost.ii

    IF v.op == 'reduce' AND v.fold_axis is being folded:
      # Reduction becomes temporal accumulation
      # It can start as soon as the first input arrives
      # but doesn't produce output until all inputs accumulated
      v.t_end = v.t + fold_factor * predecessor.cost.ii

  For each data edge (u, v):
    e.t_produce = u.t + u.cost.latency_cycles  (+ fold iteration offset if folded)
    e.t_consume = v.t
    e.lifetime = e.t_consume - e.t_produce

PHASE 4 — INSERT INFRASTRUCTURE:
  For each data edge where lifetime > 1:
    depth = lifetime
    width = prod(e.shape) * e.bw
    
    IF width * depth < threshold:
      Insert buffer node: op=buffer, params={width, depth}
      Bind to register_buffer kernel
    ELSE:
      Insert buffer node: op=buffer, params={width, depth}
      Bind to bram_buffer kernel

  For each folded vertex v where inputs come from different data sources
  across fold iterations:
    Insert mux node: op=mux, params={n_inputs=v.fold, bw=input_width}
    Bind to lut_mux kernel

PHASE 5 — COST ROLL-UP:
  total_area.lut = sum(v.cost.lut for all v, counting shared instances once)
  total_area.ff  = sum(v.cost.ff ...) + sum(buffer.cost.ff)
  total_area.dsp = sum(v.cost.dsp ...)
  total_area.bram = sum(buffer.cost.bram)
  makespan = max(v.t_end for all v)
  ii = critical_cycle_path through the folded graph
```

---

**Step 4: The Pareto Exploration Loop**

```
FOR K in [1, 2, 4, 8]:                          # fold factors
  apply_fold(sched_ir, K)
  
  FOR fusion_set in enumerate_fusions(sched_ir): # fusion combos
    fused = apply_fusions(sched_ir, fusion_set)
    bind(fused, resource_yaml)
    schedule(fused)
    insert_infrastructure(fused)
    cost = rollup(fused)
    
    IF cost.area <= device.budget:
      record(K, fusion_set, cost.makespan, cost.area)

pareto = extract_pareto_frontier(all_recorded_points)
```

For JEDI-linear the fold factor is a single parameter (particle dimension), so this is 4 iterations. For a CNN with spatial folding, it might be fold_h × fold_w × fold_c. For a Transformer, fold_seq × fold_heads. The decomposition rules expose which axes exist by populating `fold_axes` — the scheduler doesn't need to know what "particles" or "sequence positions" mean, just that there's an axis of size 8 that can be folded.

---

**The separation of concerns:**

```
Decomposition rules:  operation-specific, compiler-independent
                      "a QSum is a reduce primitive"
                      Same rules for da4ml, hls4ml, or any future compiler
                      Know about neural network operations
                      Don't know about hardware

Resource YAML:        compiler-specific, operation-independent
                      "da4ml can implement dense ops, here's how to query cost"
                      Same YAML for JEDI-linear, ResNet, or any future architecture
                      Know about hardware
                      Don't know about neural networks

Scheduler:            both operation-independent and compiler-independent
                      Takes a typed DAG, fold constraints, and cost functions
                      Assigns folds, times, instances, inserts buffers
                      Knows about neither neural networks nor hardware
                      Only knows about dependencies, time, and resource budgets

NN-IR:                architecture-specific, everything-else-independent
                      Pure mathematical description of the model
                      Populated by a frontend (Keras parser, ONNX parser, etc.)
```

Adding a new architecture means: write a parser to populate NN-IR, possibly add decomposition rules if it uses layer types not yet covered. Adding a new compiler means: write a new resource YAML and implement the cost query interface. Adding a new FPGA device means: change the device section of the resource YAML. The scheduler never changes.