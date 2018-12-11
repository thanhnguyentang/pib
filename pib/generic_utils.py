import os 
import glob 
import json, jsbeautifier
import keras.backend as K

def mkdir(*path):
    """Make a concatenated path specified by the argument. 
    """
    concat_path = ""
    for p in path:
        concat_path = os.path.join(concat_path, p)
    if not os.path.exists(concat_path):
        os.makedirs(concat_path)
    return concat_path

def parse_best_keras_checkpoint(ckpt_dir, mode='min', restore_prefix='val'):
    """Parse the best keras checkpoint. 

    If two checkpoints have the same metric values, pick the one with greater epoch.

    # Arguments
        ckpt_dir: str
        mode: str, min/max 
        restore_prefix: str, model/best

    """
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.hdf5'))
    assert len(ckpts) > 0, 'Not checkpoint found!'
    ckpt_full_names = []
    ckpt_metrics = []
    epochs = []
    for ckpt in ckpts:
        cpkt_name = ckpt.split('/')[-1].split('-')
        _ckpt_type = cpkt_name[0].split('.')[0] # 'model' or 'best
        epoch = int(cpkt_name[0].split('.')[1])
        if _ckpt_type != restore_prefix:
            continue
        ckpt_metric = float(cpkt_name[1][:-5]) 
        ckpt_metrics.append(ckpt_metric)
        ckpt_full_names.append(ckpt)
        epochs.append(epoch)

    # Sort descending order by `epochs`
    zipped = zip(epochs, ckpt_metrics, ckpt_full_names )
    zipped = sorted(zipped, reverse=True)
    epochs, ckpt_metrics, ckpt_full_names = zip(*zipped)
    if mode =='min':
        index = ckpt_metrics.index(min(ckpt_metrics))
    elif mode =='max':
        index = ckpt_metrics.index(max(ckpt_metrics))
    else:
        raise ValueError('`mode` has value {}. '.format(mode) + 
            'Only `min` and `max` are allowed.')
    return ckpt_full_names[index], epochs[index], ckpt_metrics[index]

def beauty_js(js_path, **kwargs):
    with open(js_path, 'w') as f:
        f.write(jsbeautifier.beautify(json.dumps(kwargs)))

def get_model_size(model):
    """Count the total number of trainable parameters in a Keras Model instance.
    """
    model_size = 0
    for variable in model.trainable_weights:
        shape = K.int_shape(variable) 
        var_size = 1
        for dim in shape:
            var_size *= dim
        model_size += var_size
    print("Number of trainable params in {}: {}".format(model.name, model_size))
