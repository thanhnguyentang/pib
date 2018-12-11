import numpy as np 
import keras
import keras.backend as K
from keras.datasets import mnist

# Data 

class Data(object):
    """Abstract data class. 
    """
    def __init__(self, feature_rank=1, width = 28, height =28, channels = 1, name='MNIST'):
        self.name = name
        self.feature_rank = feature_rank
        self.width = width
        self.height = height 
        self.channels = channels 
        if feature_rank == 1:
            self.feature_shape = (width * height * channels,)
        elif feature_rank == 3:
            if K.image_data_format() == 'channels_first':
                self.feature_shape = (channels, width, height)
            else:
                self.feature_shape = (width, height, channels)
        else:
            raise ValueError('Unrecognized feature rank: {}.'.format(feature_rank))

class StandardSplitMNIST(Data):
    """Split mnist into trainval:test = 60k:1k, 
    then train the model on the entire 60k with the best configuration on val. 

    This training practice follows from the VIB paper. 

    """
    def __init__(self, **kwargs):
        super(StandardSplitMNIST, self).__init__(**kwargs)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.reshape(x_train, (60000,) + self.feature_shape)
        x_test = np.reshape(x_test, (10000,) + self.feature_shape)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        self.x_train = x_train 
        self.y_train = y_train 
        self.x_test = x_test 
        self.y_test = y_test
    def split_val(self, validation_split):
        val_num = int(validation_split * self.x_train.shape[0])
        self.x_val = self.x_train[-val_num:,:]
        self.y_val = self.y_train[-val_num:,:]