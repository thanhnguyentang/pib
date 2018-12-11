from keras.layers import Input, Dense
from keras.models import Model 

from pib.generic_utils import parse_best_keras_checkpoint, beauty_js
from pib.model import get_berdense_model, get_gaussdense_model
from pib.training_utils import get_pib_callbacks, _train_with_validation, _train_until_last_epoch, _test_and_report
from pib.data_utils import StandardSplitMNIST

def train(config):
    model = get_berdense_model(config)
    # model = get_gaussdense_model(config)
    data = StandardSplitMNIST(feature_rank=1)

    if config.train_mode == 'validate':
        _train_with_validation(model, data.x_train, data.y_train, config)
    elif config.train_mode == 'last_epoch':
        _, best_epoch, _ = parse_best_keras_checkpoint(config.checkpoint_path, mode='min', restore_prefix='val')
        if not (isinstance(best_epoch,int) and best_epoch > 0):
            raise TypeError('No checkpointed model is found at {}. '.format(config.checkpoint_path) + 
                'Remind: Training on the entire data (i.e., `train_mode` = {}) requires a checkpointed **validated** model.'.format(config.train_mode))
        _train_until_last_epoch(model, data.x_train, data.y_train, config, best_epoch=best_epoch)
    else:
        raise KeyError('Unrecognized train mode: {}.'.format(config.train_mode))
def test(config):
    data = StandardSplitMNIST(feature_rank=1)
    if config.test_mode=='test':
        _test_and_report(data.x_test, data.y_test, config)
    elif config.test_mode=='val':
        data.split_val(config.validation_split)
        _test_and_report(data.x_val, data.y_val, config)
    else:
        raise KeyError('Unrecognized test mode: {}.'.format(config.test_mode))
def debug(config):
    print('Hello World!')