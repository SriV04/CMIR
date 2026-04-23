from heterograph import HGraph
from IPython.display import SVG, display


# NN-IR answers "what does the model compute and with what shapes/bitwidths?" It's a faithful structural translation of the Keras model 
# — one vertex per layer, edges following the Keras connectivity, properties copied directly from the model and HGQ metadata. No analysis,
# no decomposition, no scheduling decisions.

def vinit_nn(g, vx):
    g.pmap[vx] = {
        # --- Identity ---
        'layer_name':   None,    # e.g. 'q_einsum_dense_batchnorm'
        'layer_class':  None,    # e.g. 'QEinsumDenseBatchnorm'
        'layer_idx':    None,    # position in model.layers

        # --- Operation semantics ---
        'op_kind':      None,    # 'input' | 'einsum_dense_bn' | 'einsum_dense' | 'dense'
                                 # | 'qsum' | 'qadd' | 'activation'
        'equation':     None,    # einsum string if applicable
        'activation':   None,    # 'relu' | 'softmax' | None
        'kernel_shape': None,    # weight matrix shape

        # --- Tensor geometry ---
        'in_shapes':    None,    # list of input shapes
        'out_shapes':   None,    # list of output shapes

        # --- Quantization (read directly from HGQ) ---
        'iq_bw':        None,    # input quantizer avg bitwidth
        'kq_bw':        None,    # kernel quantizer avg bitwidth
        'bq_bw':        None,    # bias quantizer avg bitwidth
        'iq_bw_per_param': None, # full bitwidth array from HGQ (not just mean)
        'kq_bw_per_param': None, # full bitwidth array
        'sparsity':     None,    # fraction of zero weights (countable from kq)

        # --- Parameter count ---
        'num_params':   None,

        # --- Data fields ---
        'weights':      None,    # weight matrix
        'biases':       None,    # bias vector
    }

def einit_nn(g, e):
    g.pmap[e] = {
        'tensor_shape': None,    # shape of data on this edge
        'bitwidth_src': None,    # source-side output bitwidth (producer's view)
        'bitwidth_dst': None,    # consumer-side input bitwidth (downstream iq)
        'volume_bits':  None,    # prod(non-batch shape) × bitwidth_dst
    }

def ginit_nn(g):
    g.pmap['name'] = None
    g.pmap['model_source'] = None     # 'keras_hgq'
    g.pmap['n_features'] = None
    g.pmap['n_classes'] = None

    