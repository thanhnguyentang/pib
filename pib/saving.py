"""Model saving customized utilities.

(thanhnt) 
    Allows `target_tensors` when compiling a restored model in `load_model`.
    Adds `consider_weight_name_match` in `load_weights` which allows weight loading 
        even when there is mismatch in the number of weights within two corresponding layer. 
        Note that `load_weights` here is a static method, not a model method. 
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import json
import warnings
from six.moves import zip

import keras.backend as K
from keras.utils.io_utils import H5Dict 
from keras.engine.saving import model_from_config, preprocess_weights_for_loading, load_weights_from_hdf5_group, load_attributes_from_hdf5_group
# from keras import optimizers
from pib import optimizers

try:
    import h5py
    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    h5py = None


def _deserialize_model(h5dict, custom_objects=None, compile=True, **kwargs):
    """De-serializes a model serialized via _serialize_model

    # Arguments
        h5dict: `keras.utils.hdf5_utils.HFDict` instance.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.

    # Returns
        A Keras model instance. If an optimizer was found
        as part of the saved model, the model is already
        compiled. Otherwise, the model is uncompiled and
        a warning will be displayed. When `compile` is set
        to False, the compilation is omitted without any
        warning.
    """
    if not custom_objects:
        custom_objects = {}

    def convert_custom_objects(obj):
        """Handles custom object lookup.

        # Arguments
            obj: object, dict, or list.

        # Returns
            The same structure, where occurrences
                of a custom object name have been replaced
                with the custom object.
        """
        if isinstance(obj, list):
            deserialized = []
            for value in obj:
                deserialized.append(convert_custom_objects(value))
            return deserialized
        if isinstance(obj, dict):
            deserialized = {}
            for key, value in obj.items():
                deserialized[key] = convert_custom_objects(value)
            return deserialized
        if obj in custom_objects:
            return custom_objects[obj]
        return obj

    model_config = h5dict['model_config']
    if model_config is None:
        raise ValueError('No model found in config.')
    model_config = json.loads(model_config.decode('utf-8'))
    model_weights_group = h5dict['model_weights']
    if 'config_modifier' in kwargs:
        model_config = kwargs['config_modifier'](model_config)
        layer_names = []
        for layer in model_config['config']['layers']:
            layer_names.append(layer['name'])
    else:
        layer_names = model_weights_group['layer_names']

    model = model_from_config(model_config,  custom_objects=custom_objects)
    if 'keras_version' in model_weights_group:
        original_keras_version = model_weights_group['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in model_weights_group:
        original_backend = model_weights_group['backend'].decode('utf8')
    else:
        original_backend = None


    layers = model.layers

    filtered_layers = []
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)

    filtered_layer_names = []
    for name in layer_names:
        layer_weights = model_weights_group[name]
        weight_names = layer_weights['weight_names']
        if weight_names:
            filtered_layer_names.append(name)

    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError('You are trying to load a weight file'
                         ' containing {} layers into a model with {} layers'
                         .format(len(layer_names), len(filtered_layers))
                         )

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        layer_weights = model_weights_group[name]
        weight_names = layer_weights['weight_names']
        weight_values = [layer_weights[weight_name] for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = layer.weights
        weight_values = preprocess_weights_for_loading(layer,
                                                       weight_values,
                                                       original_keras_version,
                                                       original_backend,
                                                       reshape=False)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer #' + str(k) +
                             ' (named "' + layer.name +
                             '" in the current model) was found to '
                             'correspond to layer ' + name +
                             ' in the save file. '
                             'However the new layer ' + layer.name +
                             ' expects ' + str(len(symbolic_weights)) +
                             ' weights, but the saved weights have ' +
                             str(len(weight_values)) +
                             ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)

    if compile:
        training_config = h5dict.get('training_config')
        if training_config is None:
            warnings.warn('No training configuration found in save file: '
                          'the model was *not* compiled. '
                          'Compile it manually.')
            return model
        training_config = json.loads(training_config.decode('utf-8'))
        optimizer_config = training_config['optimizer_config']
        optimizer = optimizers.deserialize(optimizer_config,
                                           custom_objects=custom_objects)

        # Recover loss functions and metrics.
        try:
            loss = kwargs['loss']
        except:
            loss = convert_custom_objects(training_config['loss'])
        try: 
            metrics = kwargs['metrics']
        except:
            metrics = convert_custom_objects(training_config['metrics'])
        try:
            sample_weight_mode = kwargs['sample_weight_mode']
        except:
            sample_weight_mode = training_config['sample_weight_mode']
        try: 
            loss_weights = kwargs['loss_weights']
        except:
            loss_weights = training_config['loss_weights']
        try:
            target_tensors = kwargs['target_tensors']
        except:
            target_tensors = None

        # Compile model.
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics,
                      loss_weights=loss_weights,
                      target_tensors=target_tensors, #(thanhnt): Add `target_tensors` when compiling a restored model. 
                      sample_weight_mode=sample_weight_mode)

        # Set optimizer weights.
        if 'optimizer_weights' in h5dict:
            # Build train function (to get weight updates).
            model._make_train_function()
            optimizer_weights_group = h5dict['optimizer_weights']
            optimizer_weight_names = [
                n.decode('utf8') for n in
                optimizer_weights_group['weight_names']]
            optimizer_weight_values = [optimizer_weights_group[n] for n in
                                       optimizer_weight_names]
            try:
                model.optimizer.set_weights(optimizer_weight_values)
            except ValueError:
                warnings.warn('Error in loading the saved optimizer '
                              'state. As a result, your model is '
                              'starting with a freshly initialized '
                              'optimizer.')
    return model

def load_model(filepath, custom_objects=None, compile=True, **kwargs):
    """Loads a model saved via `save_model`.

    This method can be used to resume a training as the optimization states are also restored. 

    Example: Restore a PIB model for training resume. Note that for PIB, always explicitly specify 
        `target_tensors` in `load_model` as the output of PIB is not the same shape as its input. 

        from keras.layers import Input
        from pib.generic_utils import parse_best_keras_checkpoint
        from pib.saving import load_model
        from pib.layers import BinaryStochasticDense
        from pib.losses import weighted_vcr, _update_var
        from pib.metrics import stochastic_categorical_error
    
        pib_model_path = parse_best_keras_checkpoint(checkpoint_path, mode='min', ckpt_type='model')
        custom_objects = {
            'BinaryStochasticDense':BinaryStochasticDense, 
            'stochastic_categorical_error':stochastic_categorical_error, 
            'weighted_vcr': weighted_vcr}
        y_true = Input(shape=(config.layer_sizes[-1],), dtype='float32', name='targets')
        restored_model = load_model(pib_model_path, custom_objects=custom_objects, compile=True, target_tensors=[y_true])

    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File or h5py.Group object from which to load the model
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.
        kwargs: Custom arguments.  __allowed_kwargs = ['target_tensors', 'config_modifier', 'loss', 'metrics', 'sample_weight_mode', 'loss_weights' ]
            target_tensors: Tensor. It is critical to explicitly provide `target_tensors` for a stochastic model. 
            config_modifier: A function instance that modifies the model config.
            loss: 
            metrics:
            sample_weight_mode:
            loss_weights:


    # Returns
        A Keras model instance. If an optimizer was found
        as part of the saved model, the model is already
        compiled. Otherwise, the model is uncompiled and
        a warning will be displayed. When `compile` is set
        to False, the compilation is omitted without any
        warning.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')
    model = None
    opened_new_file = not isinstance(filepath, h5py.Group)
    h5dict = H5Dict(filepath, 'r')
    try:
        model = _deserialize_model(h5dict, custom_objects, compile, **kwargs)
    finally:
        if opened_new_file:
            h5dict.close()
    return model

def load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=False,
                                         reshape=False, consider_weight_name_match=False):
    """Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: A list of target layers.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.
        consider_weight_name_match: Boolean, whether to consider loading of layers
            even when there is a mismatch in the number of weights,
            in this case loading any weights that have name and shape match,
            only applicable when `skip_mismatch` = False

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file and skip_mismatch=False.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend,
                reshape=reshape)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    warnings.warn('Skipping loading of weights for '
                                  'layer {}'.format(layer.name) + ' due to mismatch '
                                  'in number of weights ({} vs {}).'.format(
                                      len(symbolic_weights), len(weight_values)))
                    continue
                else: #(thanhnt): Allows loading if variable name match (conditioned on variable shape match)
                    if not consider_weight_name_match:
                        raise ValueError('Layer #' + str(k) +
                                        ' (named "' + layer.name +
                                        '") expects ' +
                                        str(len(symbolic_weights)) +
                                        ' weight(s), but the saved weights' +
                                        ' have ' + str(len(weight_values)) +
                                        ' element(s).' +
                                        'Consider set `consider_weight_name_match`' + 
                                        ' to `True` to load weights by name match.')
                    else: 
                        warnings.warn('Mismatch in '
                                      'the number of weights ({} vs {}).'.format(
                                      len(symbolic_weights), len(weight_values)) + 
                                      ' Loading still continues for whichever model variable whose name matches that of the stored variables '
                                      '(conditioned on variable shape match).')
                        warning_weights = []
                        for i in range(len(symbolic_weights)):
                            symbolic_shape = K.int_shape(symbolic_weights[i])
                            symbolic_name = symbolic_weights[i].name.split('/')[-1].split(':')[0]
                            # Look up for any weight name match
                            _check = [  weight_value_tuples.append((symbolic_weights[i], weight_value))    \
                                for weight_name, weight_value in zip(weight_names, weight_values) \
                                    if weight_name.split('/')[-1].split(':')[0] == symbolic_name and \
                                        weight_value.shape == symbolic_shape ]
                            if len(_check) == 0:
                                warning_weights.append(symbolic_weights[i].name)
                        if len(warning_weights) > 0:
                            warnings.warn('Skipping loading of weights of some variables for '
                                        'layer {}'.format(layer.name) + ' due to mismatch '
                                        'in variable names or variable shapes. ' 
                                        'The variables are {}.'.format(warning_weights) + 
                                        'The stored variables are {}.'.format(weight_names))
            else:
                # Set values.
                for i in range(len(weight_values)):
                    symbolic_shape = K.int_shape(symbolic_weights[i])
                    if symbolic_shape != weight_values[i].shape:
                        if skip_mismatch:
                            warnings.warn('Skipping loading of weights for '
                                        'layer {}'.format(layer.name) + ' due to '
                                        'mismatch in shape ({} vs {}).'.format(
                                            symbolic_weights[i].shape,
                                            weight_values[i].shape))
                            continue
                        else:
                            raise ValueError('Layer #' + str(k) +
                                            ' (named "' + layer.name +
                                            '"), weight ' +
                                            str(symbolic_weights[i]) +
                                            ' has shape {}'.format(symbolic_shape) +
                                            ', but the saved weight has shape ' +
                                            str(weight_values[i].shape) + '.')
                    else:
                        weight_value_tuples.append((symbolic_weights[i],
                                                    weight_values[i]))

    K.batch_set_value(weight_value_tuples)

def load_weights(model, filepath, by_name=False,
                     skip_mismatch=False, reshape=False, consider_weight_name_match=False):
        """Loads all layer weights from a HDF5 save file.

        This method should only be used for testing a model, NOT for resuming a training as the optimization update is not restored. 

        If `by_name` is False (default) weights are loaded
        based on the network's topology, meaning the architecture
        should be the same as when the weights were saved.
        Note that layers that don't have weights are not taken
        into account in the topological ordering, so adding or
        removing layers is fine as long as they don't have weights.

        If `by_name` is True, weights are loaded into layers
        only if they share the same name. This is useful
        for fine-tuning or transfer-learning models where
        some of the layers have changed.

        # Arguments
            filepath: String, path to the weights file to load.
            by_name: Boolean, whether to load weights by layer name
                or by topological order.
            skip_mismatch: Boolean, whether to skip loading of layers
                where there is a mismatch in the number of weights,
                or a mismatch in the shape of the weight
                (only valid when `by_name`=True).
            reshape: Reshape weights to fit the layer when the correct number
                of weight arrays is present but their shape does not match.
            consider_weight_name_match: Boolean, whether to consider loading of layers
                even when there is a mismatch in the number of weights,
                in this case loading any weights that have name and shape match,
                only applicable when `skip_mismatch` = False and `by_name` = True

        # Raises
            ImportError: If h5py is not available.
        """
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights'] # if `filepath` obtained by `save_model`
            if by_name:
                load_weights_from_hdf5_group_by_name(
                    f, model.layers, skip_mismatch=skip_mismatch,
                    reshape=reshape,consider_weight_name_match=consider_weight_name_match)
            else:
                load_weights_from_hdf5_group(
                    f, model.layers, reshape=reshape)