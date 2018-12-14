"""Configuration for PIB experiments
"""
# An effort for reproducibility. 
import numpy as np 
import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))
import random 
import os 

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(2019)
random.seed(2019)

from keras import backend as K 
tf.set_random_seed(2019)
# sess_config = tf.ConfigProto(intra_op_parallelism_threads=1,
#                               inter_op_parallelism_threads=1)
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(graph=tf.get_default_graph(), config=sess_config)
K.set_session(sess)


from pib.main import train, test, debug
from pib.generic_utils import mkdir
from pib.config_utils import infer_and_assert_cfg

import pib 
print("PIB version: {}".format(pib.__version__))

run_dir = mkdir('../pib_run')

flags = tf.app.flags # It does not interfere with the keras backend as it only uses the flags of Tensorflow. 
flags.DEFINE_string("mode", "train", "Which mode to run [prepro/train/test/]. ")
flags.DEFINE_string('train_mode', 'validate', 'Whether `validate` or `last_epoch`. During training (i.e., `mode`=train), whether to validate the model (`train_mode`=validate) or train until the last epoch without validation inbetween (`train_mode`=last_epoch)')
flags.DEFINE_string('test_mode', 'test','Either `test` or `val` depending on whether to report the result on the test set or validation set.')
flags.DEFINE_string('run_dir', run_dir, 'Where to store training progress. Recommend not to interfere the train returns with the code dir. This would make life much easier when you later push the code base to Git or any cloud service.')
flags.DEFINE_string('model_name', 'PIBv1_val_512x3_16x3', 'Model name.')

# To be inferred
flags.DEFINE_string('model_path', '', 'Model path.')
flags.DEFINE_string("checkpoint_path", '', "Checkpoint path.")
flags.DEFINE_string('log_path', '', 'Log path. ')
flags.DEFINE_string('report_path', '', 'Where to write prediction report.')
flags.DEFINE_string('config_path', '', 'File path to save config.')

flags.DEFINE_bool('checkpoint', True, 'Whether to checkpoint the model duing training. ')
flags.DEFINE_bool('early_stopping', True, 'Whether to use early stopping. ')
flags.DEFINE_integer('patience', 20, 'Patience (in terms of epochs) for early stopping. ')
flags.DEFINE_bool('tensorboard', True, 'Whether to use tensorboard. ')
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate. ")
flags.DEFINE_list("layer_sizes", [784,512,10], "Layer sizes of the input, hidden and output layers.") #int for default but will be string if passed in Terminal. 
flags.DEFINE_list("pib_gammas", [1., 0.0], "Weights of the PIB objective of the super layer and all hidden layers. If one value is specified, that value is used for all layers.")
flags.DEFINE_list("pib_betas", [0.0], "Compression weights for each hidden layer. If one value is specified, that value is used for all layers.")
flags.DEFINE_list("n_particles", [16], "Number of particles to generate at each hidden layers given the previous layer. If one value is specified, that value is used for all layers.")
flags.DEFINE_list('test_n_particles', [32], 'Number of particles to generate at each hidden layers given the previous layer at test. If one value is specified, that value is used for all layers. If empty list, set to `n_particles`.')
flags.DEFINE_integer('batch_size', 16, 'Batch size. ')
flags.DEFINE_integer('n_epochs', 1000, 'Number of training epochs.')
flags.DEFINE_string("optimizer", "rmsprop", "Optimizer. ")
flags.DEFINE_bool('verbose', True, 'Verbose.')
flags.DEFINE_string('restore_prefix', 'val', 'Either `val` or `full`. Prefix of hdf5 model files to be parsed at test. `val` for the saved validated model and `full` for the saved full-train model.')
flags.DEFINE_float('validation_split', 1/ 60., 'Validation split. ')
flags.DEFINE_integer('histogram_freq', 1, 'Histogram freq in terms of epoch to show in Tensorboard.')
flags.DEFINE_float('clipnorm', 5., 'Clip gradients if the L2 norm exceeds this value. 0 indicates not clipping. ')
flags.DEFINE_float('clipvalue', 0., 'Clip gradients if the absolute value exceeds this value. 0 indicates no clipping. ')
flags.DEFINE_float('l1_kernel_reg', 0, 'L1 kernel regularization.')
flags.DEFINE_float('l2_kernel_reg', 0, 'L1 kernel regularization.')
flags.DEFINE_float('l1_prior_reg', 0, 'L1 prior regularization.')
flags.DEFINE_float('l2_prior_reg', 0, 'L1 pripr regularization.')
flags.DEFINE_bool('run_det', True, 'Whether to evaluate the deterministic equivalent model at test.')
flags.DEFINE_integer('n_runs', 100, 'Number of times to run the stochastice model on a test set.')
flags.DEFINE_string('discrete_grad_relax', 'raiko', 'Relaxation of discrete variables for estimating gradients.')
flags.DEFINE_float('temperature', 1.0, 'Temperature in Gumbel Trick.')



def main(_):
    config = infer_and_assert_cfg(flags.FLAGS) 
    if config.mode == 'train':
        train(config)
    elif config.mode == 'test':
        test(config)
    elif config.mode == 'debug':
        run_mlp(config)
if __name__ == "__main__":
    tf.app.run()