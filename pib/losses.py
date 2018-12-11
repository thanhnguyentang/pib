import keras.backend as K
import numpy as np
import pib

_PIB_GAMMAS = []

def _update_var(pib_gammas):
    _PIB_GAMMAS.extend(pib_gammas)

def weighted_vcr(y_true, y_pred):
    """Computes weighted VCR.
    
    This computation does not use `softmax` but instead uses `logsumexp` which is more computationally stable.
    
    # Arguments
        y_true: A 2-D `Tensor`.
        y_red: A n-D `Tensor` whose dimensions represent the samples of the corresponding layers in a DNN. 
        *args: A optional list of vcr weights for each layer (0 <=l <= L). By default, args = [], and all weights are 1. 
        
    # Returns
        A `Scalar`. 
    """
    input_shape = K.int_shape(y_pred)
    rank = len(input_shape)  
    loss = 0. 
    for i in range(rank-1):
        layer_input_shape = (-1, int(np.prod(input_shape[1:i+1])), int(np.prod(input_shape[i+1:-1])),input_shape[-1])
        reshaped_inputs = K.reshape(y_pred, layer_input_shape)
        alp = reshaped_inputs - K.logsumexp(reshaped_inputs, axis=-1, keepdims=True) - K.log(1. * layer_input_shape[-2])
        log_dist_given_zl = K.logsumexp(alp, axis=-2)
        loss += _PIB_GAMMAS[i] * K.mean(-K.sum(y_true * K.mean(log_dist_given_zl, 1), -1))
    return loss 

pib.CUSTOM_OBJECTS['weighted_vcr'] = weighted_vcr