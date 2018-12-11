"""Various training routines for PIB.
"""
import numpy as np 
import os 
import keras 
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import save_model
import pib.common
from pib.generic_utils import parse_best_keras_checkpoint, beauty_js
from pib.losses import _update_var
from pib.config_utils import save_cfg, sanitize_config
from pib.model import load_sto2det, load_sto2sto

def _train_with_validation(model, x_train, y_train, config):
    config = sanitize_config(config)
    save_cfg(config)

    callbacks_list = get_pib_callbacks(config)
    model.fit(x_train, y_train,
            batch_size=config.batch_size,
            epochs=config.n_epochs,
            verbose=config.verbose, 
            callbacks=callbacks_list, 
            validation_split=config.validation_split, 
            shuffle=False) 
    # _, best_epoch, _= parse_best_keras_checkpoint(config.checkpoint_path, mode='min', restore_prefix='val')
    # return best_epoch 

def _train_until_last_epoch(model, x_train, y_train, config, best_epoch=None):
    config.checkpoint = False 
    config.early_stopping = False 
    config.histogram_freq = 0 #No val data
    config.n_epochs = best_epoch 
    save_cfg(config)

    callbacks_list = get_pib_callbacks(config)
    model.fit(x_train, y_train,
            batch_size=config.batch_size,
            epochs=config.n_epochs,
            verbose=config.verbose, 
            callbacks=callbacks_list, 
            validation_split=0., 
            shuffle=False) 
    ckpt_path = config.checkpoint_path + "/full.{:02d}-{:.5f}.hdf5".format(best_epoch, 0)
    save_model(model, ckpt_path, overwrite=True, include_optimizer=True)

def _test_and_report(x_test, y_test, config):
    save_cfg(config)

    report={}
    pib_model_path, _, stochastic_metric = parse_best_keras_checkpoint(config.checkpoint_path, mode='min', restore_prefix=config.restore_prefix)
    report['stochastic_metric'] = stochastic_metric
    if config.verbose:
        print('Loading from {}'.format(pib_model_path))
    # PIB 
    pib_model = load_sto2sto(pib_model_path, config)

    pib_test_errs = []
    for _ in range(config.n_runs):
        _, pib_test_err = pib_model.evaluate(x_test, y_test, verbose=config.verbose)
        pib_test_errs.append(pib_test_err / pib.common._PRECISION)
    report['pib_test_err_mean'] = np.mean(pib_test_errs)
    report['pib_test_err_var'] =  np.var(pib_test_errs)
    # DET
    if config.run_det:
        det_model = load_sto2det(pib_model_path)
        det_test_loss, det_test_acc = det_model.evaluate(x_test, y_test, verbose=config.verbose)
        report['det_test_err'] = (1. - det_test_acc)*100

    beauty_js(config.report_path, **report)
    if config.verbose:
        for k,v in report.items():
            print('{} : {}'.format(k,v))

def get_pib_callbacks(config):
    callbacks_list = []
    if config.checkpoint:
        model_filepath = config.checkpoint_path + "/val.{epoch:02d}-{val_stochastic_categorical_error:.5f}.hdf5"
        model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_stochastic_categorical_error', verbose=config.verbose, \
                                    save_best_only=True, mode='min', save_weights_only=False)

        callbacks_list.append(model_checkpoint)
    # Early Stopping
    if config.early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_stochastic_categorical_error',
                            min_delta=0.001,
                            patience=config.patience, 
                            verbose=config.verbose, 
                            mode='min', 
                            baseline=None, 
                            restore_best_weights=True)
        callbacks_list.append(early_stopping)

    if config.tensorboard:
        tb = keras.callbacks.TensorBoard(log_dir=config.log_path, 
                            histogram_freq=config.histogram_freq, 
                            batch_size=config.batch_size, 
                            write_graph=False, 
                            write_grads=True, 
                            write_images=False, 
                            embeddings_freq=0, 
                            embeddings_layer_names=None, 
                            embeddings_metadata=None, 
                            embeddings_data=None, 
                            update_freq='epoch')
        callbacks_list.append(tb)
    return callbacks_list