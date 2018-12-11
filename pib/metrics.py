import keras.backend as K
from keras.metrics import categorical_accuracy 
import numpy as np
import pib
import pib.common 

def stochastic_categorical_error(y_true, y_pred):
    input_shape = K.int_shape(y_pred)
    n_samples = int(np.prod(input_shape[1:-1]))
    y_pred_prob = K.softmax(y_pred, -1)
    y_pred_avg_prob = K.mean(K.reshape(y_pred_prob, (-1, n_samples, input_shape[-1])), 1)
    return (1. - categorical_accuracy(y_true, y_pred_avg_prob))*100. * pib.common._PRECISION # Add 100 precision so that Keras Model checkpoints take care of up to 2 + 1 = 3 digits before the decimal point.  

pib.CUSTOM_OBJECTS['stochastic_categorical_error'] = stochastic_categorical_error
