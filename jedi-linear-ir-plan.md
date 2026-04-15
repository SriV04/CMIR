# Plan: JEDI-linear → High-Level IR with Heterograph

## Understanding the JEDI-linear architecture

Before we can convert anything, we need a precise understanding of what JEDI-linear actually computes. The architecture is a linearised variant of the JEDI-net interaction network. The key insight from the paper is that by constraining the edge interaction function to be affine, the O(N²) pairwise edge computations collapse into O(N) shared transformations plus a single global aggregation.

The dataflow, layer by layer, looks like this:

```
Input: particle features  [N_particles × N_features]
                │
    ┌───────────┤
    │     Node Transform (φ_N)
    │     EinsumDense: shared across particles
    │     [N × N_features] → [N × D]
    │           │
    │     ┌─────┴─────┐
    │     │            │
    │     │    Global Aggregation
    │     │    mean/sum over particle dim
    │     │    [N × D] → [D]
    │     │            │
    │     │      Broadcast
    │     │      [D] → [N × D]
    │     │            │
    │     └─────┬──────┘
    │           │
    │     Concatenate
    │     [N × D] ++ [N × D] → [N × 2D]
    │           │
    │     Node Update MLP (φ_O)
    │     EinsumDense layers (shared across particles)
    │     [N × 2D] → [N × D']
    │           │
    │  (repeat interaction block if >1 layer)
    │           │
    └───────────┘
                │
        Global Pooling
        mean over particle dim
        [N × D'] → [D']
                │
        Classification MLP (ϕ_C)
        Dense → Dense → Dense
        [D'] → [N_classes]
                │
        Softmax → output logits
```

Each of the named blocks above has a distinct role: some are **compute** (matrix-vector multiplies with learned weights), some are **transport** (data movement — aggregation, broadcast, concatenation, reshaping), and some are **control** (synchronisation points where multiple paths merge or fork). The job of the IR is to make these roles explicit.

---

## The node type taxonomy for this model

Given JEDI-linear's structure, here is how the three IR node types map:

**Compute nodes** (`cmvm`): Any operation with learned weight matrices. In JEDI-linear these are:

- `φ_N`: the shared node transformation (EinsumDense, equation `"bnf,fh->bnh"` or similar)
- Each layer of `φ_O`: the node update MLP (EinsumDense layers)
- Each layer of `ϕ_C`: the classification head (Dense layers)

Each of these is a constant matrix-vector multiply (CMVM) when the model is frozen — the weights are fixed at inference time, and the only variable input is the activation vector. This is exactly what da4ml targets.

**Transport nodes** (`transport`): Any operation that moves, reshapes, or reduces data without learned parameters:

- `global_aggregation`: reduction (sum/mean) over the particle dimension — this is the key transport operation that replaces JEDI-net's O(N²) message passing
- `broadcast`: copying the global context vector back to each of the N particles
- `concat`: merging the per-particle features with the broadcast global context
- `global_pooling`: final reduction from [N × D'] to [D'] before classification
- Any activation functions (ReLU etc.) applied element-wise between dense layers — these are technically element-wise compute but cost essentially nothing on an FPGA compared to CMVM, so we classify them as transport

**Control nodes** (`control`):

- `input_split`: the entry point where the input feature tensor fans out to the node transform
- `merge`: the synchronisation point where the per-node path and the aggregation path rejoin before concatenation
- `barrier`: if the model has multiple interaction blocks (layers), the point between blocks where one must complete before the next begins

---

## The plan

### Phase 1 — Get the trained model loaded and inspectable (days 1–2)

**Goal:** Have a running JEDI-linear Keras model in memory, able to iterate over its layers and print their shapes, types, and connectivity.

**Steps:**

1. Clone `https://github.com/calad0i/JEDI-linear` and set up the conda environment from their `environment.yml`. This pulls in HGQ2, da4ml, Keras 3 with JAX backend.

2. Extract the official pretrained models from `official_models.tar.gz`.

3. Load one of the smaller configs (e.g., an 8-particle, 3-feature variant) and load its pretrained weights. The entry point is `jet_classifier.py` — read it to understand how the model is built. The model-building function is likely in `src/` and will use HGQ2 quantised layers wrapping `EinsumDense` and `Dense`.

4. Print the model summary. For each layer, record: layer class name, input shape(s), output shape(s), number of parameters, and connectivity (which layers feed it). This is your ground truth for the IR.

5. If the model uses HGQ2 layers, understand the wrapping: an HGQ2 `EinsumDense` wraps a Keras `EinsumDense` with per-parameter quantisation metadata. The underlying einsum equation and weight shapes are what you need for the IR.

**Deliverable:** A notebook cell that prints a table like:

```
Layer 0: Input            shape=(None, 8, 3)       → feeds [1]
Layer 1: HGQ_EinsumDense  shape=(None, 8, 16)      → feeds [2, 3]   eq="bnf,fh->bnh"
Layer 2: GlobalMean        shape=(None, 16)          → feeds [4]
Layer 3: Broadcast         shape=(None, 8, 16)       → feeds [4]      (identity/tile)
Layer 4: Concatenate       shape=(None, 8, 32)       → feeds [5]
...
```

### Phase 2 — Define the IR schema in Heterograph (days 2–3)

**Goal:** Define the `vinit`, `einit`, `ginit` functions and the `add_cmvm`, `add_transport`, `add_control` helpers. Get a hand-built toy IR rendering correctly.

**Steps:**

1. Install Heterograph (`conda env` or `pip install` from GitHub, plus `graph-tool` via conda).

2. Define the property schema. Every vertex gets an `ntype` ('cmvm', 'transport', 'control') plus type-specific properties. Use the schema from the revised plan we produced earlier:

   ```python
   def vinit(g, vx):
       g.pmap[vx] = {
           'ntype': None,
           'layer_name': None,
           'op_kind': None,       # e.g. 'einsum_dense', 'reduction', 'broadcast', etc.
           'input_shape': None,
           'output_shape': None,
           # cmvm-specific (filled only for cmvm nodes)
           'einsum_eq': None,
           'weight_shape': None,
           'in_bitwidth': None,
           'weight_bitwidth': None,
           'da4ml_lut_count': None,
           'da4ml_latency': None,
           # transport-specific
           'transport_kind': None, # 'reduction', 'broadcast', 'concat', 'activation', 'reshape'
           'fanout': None,
           'volume_bits': None,
           # control-specific
           'control_kind': None,   # 'split', 'merge', 'barrier'
           # analysis (filled later)
           'hotspot_score': 0.0,
           'partition_id': -1,
       }
   ```

3. Write helper functions: `add_cmvm(g, ...)`, `add_transport(g, ...)`, `add_control(g, ...)` that create a vertex, fill its pmap, and return the vertex ID.

4. Hand-build a tiny 3-node test graph (one cmvm, one transport, one control) and render it with `g.view()` to verify styling works. Set up `vstyle` with colours by type (purple/coral/gray).

**Deliverable:** An `ir/schema.py` and `ir/graph.py` module, plus a test that builds and renders a toy graph.

### Phase 3 — Write the Keras-to-IR converter (days 3–5)

**Goal:** Automatically walk the JEDI-linear Keras model and produce an `HGraph` IR.

This is the core engineering work. The converter must:

1. **Iterate the Keras model's layer graph.** In Keras 3, `model.layers` gives the list, and each layer's `_inbound_nodes` (or the functional API's connectivity info) tells you which layers feed it. Alternatively, use `model.get_config()` to get the JSON connectivity.

2. **Classify each layer** into one of the three IR node types:

   | Keras layer class           | IR node type  | `op_kind`        |
   |-----------------------------|---------------|------------------|
   | `EinsumDense` / `Dense`     | `cmvm`        | `einsum_dense` / `dense` |
   | `HGQ_EinsumDense`           | `cmvm`        | `einsum_dense`   |
   | `GlobalAveragePooling1D`    | `transport`   | `reduction`      |
   | custom mean/sum layer       | `transport`   | `reduction`      |
   | `Concatenate`               | `transport`   | `concat`         |
   | `Activation` / `ReLU`       | `transport`   | `activation`     |
   | `Reshape` / `RepeatVector`  | `transport`   | `broadcast`      |
   | `InputLayer`                | `control`     | `split`          |
   | `Add` (residual)            | `control`     | `merge`          |

   **Important:** The exact layer names depend on how the model is built. Read the `src/` directory in the JEDI-linear repo to see the actual layer classes used. HGQ2 wraps standard Keras layers, so you may see `HGQ.layers.EinsumDense` instead of `keras.layers.EinsumDense`.

3. **Extract attributes** for each node:
   - For `cmvm` nodes: the einsum equation string, weight shape, bitwidths (from HGQ2 quantisation metadata)
   - For `transport` nodes: input/output shapes, the reduction axis, estimated volume in bits
   - For `control` nodes: just the kind

4. **Add edges** based on the Keras layer connectivity. In the functional API, `layer._inbound_nodes[0].input_tensors` tells you which layers feed this one.

5. **Handle the broadcast pattern explicitly.** JEDI-linear's global aggregation → broadcast may not appear as a single named Keras layer. It might be a `Lambda` or custom layer that does `tf.reduce_mean(x, axis=1, keepdims=True)` followed by `tf.tile`. The converter needs to recognise this pattern and emit two IR nodes: a `reduction` transport node and a `broadcast` transport node.

**Deliverable:** An `ingestion/keras_to_ir.py` module with a `convert(model) -> HGraph` function. Test it on the loaded JEDI-linear model and render the output.

### Phase 4 — Query da4ml for CMVM costs (days 5–6)

**Goal:** Enrich each `cmvm` node in the IR with da4ml's LUT count and latency estimates.

**Steps:**

1. Read the da4ml source at `github.com/calad0i/da4ml` to understand its API. Based on the search results, da4ml converts models to a DAIS (distributed arithmetic instruction set) IR. The integration point with hls4ml uses `strategy='distributed_arithmetic'` on Dense/EinsumDense layers.

2. For each `cmvm` node in the IR, extract the frozen weight matrix and quantisation metadata (bitwidths) from the Keras model.

3. Call da4ml's optimiser on each weight matrix to get `lut_count` and `critical_path_depth`. Cache results keyed on `(weight_hash, bitwidths, delay_constraint)`.

4. Write these into the pmap of each cmvm node:
   ```python
   g.pmap[vx]['da4ml_lut_count'] = result.lut_count
   g.pmap[vx]['da4ml_latency'] = result.critical_path_depth
   ```

5. Render the graph with node sizes proportional to LUT count. This immediately shows which layers dominate hardware cost.

**Deliverable:** Updated cmvm nodes with da4ml cost annotations. A rendered graph showing cost distribution.

### Phase 5 — Analysis passes (days 6–8)

**Goal:** Run the dominance classifier and other analysis passes on the populated IR.

With the IR built and costed, you can now answer the key question: **is JEDI-linear compute-dominated or transport-dominated?**

1. **Dominance classifier**: Sum LUT costs of all cmvm nodes. Estimate transport cost as the sum of `volume_bits × fanout` over all transport nodes. Compare via the normalised ratio.

2. **Hotspot analysis**: Which single node consumes the most LUTs? In JEDI-linear, this is likely the first EinsumDense (φ_N) or the largest layer of the node update MLP (φ_O), because these operate on every particle.

3. **Pattern detection using AQL**: Use Heterograph's query system to find patterns:
   - Find all `cmvm → transport(reduction) → transport(broadcast) → cmvm` chains (the aggregation pattern)
   - Find all sequences of consecutive cmvm nodes (the MLP sublayers)

4. **Reuse analysis**: Check if any cmvm nodes have identical weight shapes and bitwidths. In JEDI-linear, if the model uses multiple interaction blocks with shared weights, these are shareable.

**Deliverable:** A metrics table and annotated graph rendering. An answer to "is this model compute- or transport-dominated?"

### Phase 6 — Visualisation and write-up (days 8–10)

**Goal:** Produce publication-quality IR graphs and a summary document.

1. Use Heterograph's `vstyle` to colour nodes by type, size by cost, and label with layer names.

2. Use `g.view()` for interactive exploration in notebook, and `g.render()` for SVG/PDF export.

3. Annotate the graph with the dominance classification result and per-node hotspot scores.

4. Write a summary comparing the IR structure against the original Keras model: how many nodes of each type, what fraction of LUTs are in compute vs transport, and what the critical path is.

---

## The critical unknowns (resolve these first)

Before writing code, you need to resolve three things by reading the actual JEDI-linear source:

1. **How is the global aggregation implemented?** Is it a named Keras layer, a `Lambda`, or inline in a custom `call()` method? This determines whether the converter can find it automatically or needs a pattern-matching heuristic.

2. **How does HGQ2 expose quantisation metadata?** You need per-layer bitwidth information. HGQ2 stores this as layer attributes — check whether it's `layer.kernel_quantizer.bits` or similar.

3. **What is da4ml's actual Python API?** The `da4ml.cmvm` module may not have a top-level `optimize_cmvm()` function. Read `da4ml`'s source and the hls4ml integration in `hls4ml/backends/vivado/passes/` to find the entry point.

Resolve these by reading code, not by guessing. Each one, if wrong, can waste a full day.

---

## Directory structure

```
jedi-linear-ir/
├── ir/
│   ├── schema.py         # vinit, einit, ginit, helper functions
│   └── graph.py          # IRGraph wrapper (thin, over HGraph)
├── ingestion/
│   ├── keras_to_ir.py    # Keras model → HGraph converter
│   └── da4ml_cost.py     # da4ml cost query + cache
├── analyses/
│   ├── dominance.py      # compute vs transport classification
│   ├── hotspot.py        # per-node cost ranking
│   └── patterns.py       # AQL-based pattern detection
├── notebooks/
│   ├── 01_load_model.ipynb
│   ├── 02_build_ir.ipynb
│   └── 03_analyse.ipynb
├── tests/
│   ├── test_schema.py
│   ├── test_ingestion.py
│   └── test_analyses.py
└── environment.yml
```

## Timeline

| Days | Milestone |
|------|-----------|
| 1–2  | JEDI-linear model loaded, layers inspected, shapes recorded |
| 2–3  | HGraph IR schema defined, toy graph rendering verified |
| 3–5  | Keras→IR converter working, full JEDI-linear model converted |
| 5–6  | da4ml cost annotations on all cmvm nodes |
| 6–8  | Analysis passes: dominance, hotspot, patterns |
| 8–10 | Visualisation, write-up, clean-up |
