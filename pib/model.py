from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers, initializers
import pib 
from pib.layers import BernoulliStochasticLayer, GaussianStochasticLayer, ExtendConv2D
from pib.metrics import stochastic_categorical_error
from pib.losses import weighted_vcr, _update_var
from pib.optimizers import _OPTIMIZER_FACTORY
from pib.generic_utils import get_model_size
from pib.saving import load_model

def get_berdense_model(config):
    _update_var(config.pib_gammas)
    # weighted_betas = [b*w for b, w in zip(config.pib_betas, config.pib_gammas[1:])]
    weighted_betas = config.pib_betas
 
    kernel_regularizer = None 
    if config.l1_kernel_reg > 0 and config.l2_kernel_reg > 0:
        kernel_regularizer = regularizers.l1_l2(l1=config.l1_kernel_reg, l2=config.l2_kernel_reg)
    if config.l1_kernel_reg > 0 and config.l2_kernel_reg == 0:
        kernel_regularizer = regularizers.l1(config.l1_kernel_reg)
    if config.l1_kernel_reg == 0 and config.l2_kernel_reg > 0:
        kernel_regularizer = regularizers.l2(config.l2_kernel_reg)

    prior_regularizer = None 
    if config.l1_prior_reg > 0 and config.l2_prior_reg > 0:
        kernel_regularizer = regularizers.l1_l2(l1=config.l1_prior_reg, l2=config.l2_prior_reg)
    if config.l1_prior_reg > 0 and config.l2_prior_reg == 0:
        kernel_regularizer = regularizers.l1(config.l1_prior_reg)
    if config.l1_prior_reg == 0 and config.l2_prior_reg > 0:
        kernel_regularizer = regularizers.l2(config.l2_prior_reg)
    # Model 
    inputs = Input(shape=(config.layer_sizes[0],), dtype='float32', name='inputs')
    z = inputs
    for i in range(len(config.n_particles)):
        h = Dense(config.layer_sizes[i+1],kernel_regularizer=kernel_regularizer, kernel_initializer=initializers.glorot_uniform(seed=2019))(z)
        z = BernoulliStochasticLayer(config.n_particles[i], 1, weighted_betas[i])(h)
    outputs = Dense(config.layer_sizes[-1], name='main_output', kernel_initializer=initializers.glorot_uniform(seed=2019))(z)
    model = Model(inputs=inputs, outputs= outputs, name=config.model_name)

    if config.verbose:
        get_model_size(model)

    y_true = Input(shape=(config.layer_sizes[-1],), dtype='float32', name='targets')

    model.compile(loss= [weighted_vcr],
                target_tensors=[y_true],
                metrics = [stochastic_categorical_error],
                optimizer=_OPTIMIZER_FACTORY[config.optimizer](lr=config.learning_rate, clipnorm=config.clipnorm, clipvalue=config.clipvalue))
    return model

def get_gaussdense_model(config):
    _update_var(config.pib_gammas)
    weighted_betas = [b*w for b, w in zip(config.pib_betas, config.pib_gammas[1:])]
 
    kernel_regularizer = None 
    if config.l1_kernel_reg > 0 and config.l2_kernel_reg > 0:
        kernel_regularizer = regularizers.l1_l2(l1=config.l1_kernel_reg, l2=config.l2_kernel_reg)
    if config.l1_kernel_reg > 0 and config.l2_kernel_reg == 0:
        kernel_regularizer = regularizers.l1(config.l1_kernel_reg)
    if config.l1_kernel_reg == 0 and config.l2_kernel_reg > 0:
        kernel_regularizer = regularizers.l2(config.l2_kernel_reg)
    # Model 
    inputs = Input(shape=(config.layer_sizes[0],), dtype='float32', name='inputs')
    z = inputs
    for i in range(len(config.n_particles)):
        h = Dense(config.layer_sizes[i+1],kernel_regularizer=kernel_regularizer, activation='relu')(z)
        z = GaussianStochasticLayer(config.n_particles[i], 1, weighted_betas[i])(h)
    outputs = Dense(config.layer_sizes[-1], name='main_output')(z)
    model = Model(inputs=inputs, outputs= outputs, name=config.model_name)

    if config.verbose:
        get_model_size(model)

    y_true = Input(shape=(config.layer_sizes[-1],), dtype='float32', name='targets')

    model.compile(loss= [weighted_vcr],
                target_tensors=[y_true],
                metrics = [stochastic_categorical_error],
                optimizer=_OPTIMIZER_FACTORY[config.optimizer](lr=config.learning_rate, clipnorm=config.clipnorm, clipvalue=config.clipvalue))
    return model

def load_sto2det(modelpath):
    """Creates a deterministic dense model of the same topology as `model`
    """
    return load_model(modelpath, custom_objects=None, compile=True, config_modifier=config_sto2det, loss='categorical_crossentropy', metrics=['accuracy'])

def load_sto2sto(modelpath, config):
    _update_var(config.pib_gammas)
    return load_model(modelpath, custom_objects=pib.CUSTOM_OBJECTS, compile=True, \
        target_tensors=[Input(shape=(config.layer_sizes[-1],), dtype='float32', name='targets')], config_modifier=config_sto2sto(config.test_n_particles))

def get_sconv_model(n_particles, config):
    return 0. 


# Config modifiers
def config_sto2det(sto_config):
    import copy 
    det_config = copy.deepcopy(sto_config)
    det_config['config']['layers'] = []
    latest_inbound_node_buffer = None
    for layer in sto_config['config']['layers']:
        det_layer = copy.deepcopy(layer)
        if det_layer['class_name'] in pib.CUSTOM_OBJECTS:
            if latest_inbound_node_buffer is None:
                latest_inbound_node_buffer = det_layer['inbound_nodes']
            continue
        # If berdense is detected, the det model equivalent has sigmoid activation. 
        if det_layer['class_name'] == 'Dense' and det_layer['config']['activation'] == 'linear':
            det_layer['config']['activation'] = 'sigmoid'
        if len(det_layer['inbound_nodes']) > 0 and latest_inbound_node_buffer is not None:
            det_layer['inbound_nodes'] = latest_inbound_node_buffer
        # Reset the buffer. 
        latest_inbound_node_buffer = None
        det_config['config']['layers'].append(det_layer)
    return det_config

def config_sto2sto(n_particles):
    def _config_sto2sto(sto_config):
        index = 0 
        for layer in sto_config['config']['layers']:
            if 'n_samples' in layer['config']:
                layer['config']['n_samples'] = n_particles[index]
                index += 1 
        return sto_config
    return _config_sto2sto