import os 

from pib.generic_utils import beauty_js 
from pib.generic_utils import mkdir

def infer_and_assert_cfg(config):
    all_optimizer_alias = [
        'sgd',
        'rmsprop',
        'adagrad',
        'adadelta',
        'adam',
        'adamax',
        'nadam',
        'tfoptimizer']
    
    n_hidden_layers = len(config.layer_sizes) - 2 
    if n_hidden_layers > 1:
        if len(config.pib_gammas) == 1:
            config.pib_gammas = [config.pib_gammas[0]]*(n_hidden_layers + 1)
        if len(config.pib_betas) == 1:
            config.pib_betas = [config.pib_betas[0]]*n_hidden_layers
        if len(config.n_particles) == 1:
            config.n_particles = [config.n_particles[0]]*n_hidden_layers
        if len(config.test_n_particles) == 0:
            config.test_n_particles = config.n_particles 
        if len(config.test_n_particles) == 1:
            config.test_n_particles = [config.test_n_particles[0]]*n_hidden_layers

    assert len(config.pib_gammas) == n_hidden_layers + 1 
    assert len(config.n_particles) == n_hidden_layers
    assert len(config.test_n_particles) == n_hidden_layers
    assert len(config.pib_betas) == n_hidden_layers
    assert config.optimizer in all_optimizer_alias
    config = sanitize_config(config)
    config.model_path = mkdir(config.run_dir, 'train', config.model_name)
    config.checkpoint_path = mkdir(config.model_path, 'checkpoints')
    config.log_path = mkdir(config.model_path, 'log-train.{}'.format(config.train_mode))
    def _extract_list_as_string(xs):
        return ''.join(str(xs)[1:-1].split())
    config.config_path = os.path.join(config.model_path, 'cfg-mode.{}.{}.json'.format(config.mode, config.train_mode if config.mode=='train' else config.test_mode))
    config.report_path = os.path.join(config.model_path, 'report-trained.{}-tested.{}-particles.{}.json'.format(config.restore_prefix, config.test_mode, _extract_list_as_string(config.test_n_particles)))
    return config

def _convert_string_config(xs):
    if isinstance(xs[0], int) or isinstance(xs[0], float):
        return xs 
    elif isinstance(xs[0], str):
        if len(xs[0].split('.')) == 1:
            return [int(x) for x in xs]
        else:
            return [float(x) for x in xs]
    else: 
        raise ValueError('Unvalid input type {}'.format(type(xs[0])))

def sanitize_config(config):
    """Converts any attribute in config that is a list of strings into a list of corresponding numeric values. 
    """
    update_dict = {}
    for k in config:
        if isinstance(config[k].value, list):
            update_dict[k] = _convert_string_config(config[k].value)
    return update_cfg(config, **update_dict)

def update_cfg(config, **kwargs):
    """Update `config` tf.flags 
    """
    for k,v in kwargs.items():
        config[k].parse(v)
    return config 

def save_cfg(config):
    # Overwrite the latest one
    # if not os.path.exists(config_name):
    _hidden_config = ['help', 'h', 'helpfull', 'helpshort']
    sanitized_config_dict = {}
    for k,v in config.flag_values_dict().items():
        if k in _hidden_config:
            continue 
        sanitized_config_dict[k] = v
    beauty_js(config.config_path, **sanitized_config_dict)