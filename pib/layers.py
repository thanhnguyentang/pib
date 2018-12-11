import keras.backend as K 
from keras.utils import conv_utils
from keras.layers import Layer, Conv2D, Dense
from keras import initializers
from keras.engine.base_layer import InputSpec
from pib.backend_utils import bernoulli_compression, gaussian_compression, variational_compression
import pib

class StochasticLayer(Layer):
    """Abstract Stochastic Layer. 
    
    This layer draws `n_samples` from a nD tensor of rank k. 
    
    # Arguments
        n_samples: Integer. Number of samples to draw. 
        feature_rank: Integer. The last `feature_rank` dimensions of the input tensor is the features and the other dimensions are samples. 
        beta: float. The weight of the layer's compression. 
        
    # Returns: 
        A `Tensor` of rank (k+1) where dimension -(k+1) has `n_samples` elements. 
    
    """
    def __init__(self, n_samples, feature_rank, beta,  **kwargs):
        super(StochasticLayer, self).__init__(**kwargs)
        self.n_samples = n_samples
        self.feature_rank = feature_rank
        self.beta = beta
    
    def call(self, inputs):
        assert not isinstance(inputs, list)        
        inp_shape = K.shape(inputs)
        out_shape = (inp_shape[:-self.feature_rank], K.constant(self.n_samples, shape=(1,), dtype='int32'), inp_shape[-self.feature_rank:])
        out_shape = K.concatenate(out_shape, axis=0)
        return out_shape
    
    def compute_output_shape(self, input_shape):
        feature_shape = tuple(input_shape[-self.feature_rank:])
        return tuple(input_shape[:-self.feature_rank]) + (self.n_samples, ) + feature_shape
    
    def get_config(self):
        config = {'n_samples': self.n_samples, 'feature_rank': self.feature_rank, 'beta': self.beta }
        base_config = super(StochasticLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class BernoulliStochasticLayer(StochasticLayer):
    def build(self, input_shape):
        if self.beta > 0:
            self.prior = K.sigmoid(self.add_weight(shape=tuple(input_shape[-self.feature_rank:]),
                                  initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.5, seed=2019),
                                  regularizer=None,
                                  trainable=True,
                                  name='pior'))
        super(BernoulliStochasticLayer, self).build(input_shape)
    def call(self, inputs):
        out_shape = super(BernoulliStochasticLayer, self).call(inputs)
        eps = K.random_uniform(out_shape, seed=2019)
        inp = K.sigmoid(inputs)
        p = K.expand_dims(inp,  -(self.feature_rank+1))
        y = K.cast( (p - eps) > 0, dtype='float32' )
        z = p + K.stop_gradient(y - p)
        if self.beta > 0: 
            prior = self.prior
            mi = bernoulli_compression(inp, prior)
            self.add_loss(self.beta * mi, inputs=inputs)
        return z

class GaussianStochasticLayer(StochasticLayer):
    def build(self, input_shape):
        feature_shape = tuple(input_shape[-self.feature_rank:])
        self.mu = self.add_weight(shape=feature_shape,
                              initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.5),
                              regularizer=None,
                              trainable=True,
                              name='mu')
        self.log_sigma = self.add_weight(shape=feature_shape,
                              initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.5),
                              regularizer=None,
                              trainable=True,
                              name='sigma')
        super(GaussianStochasticLayer, self).build(input_shape)
        
    def call(self, inputs):
        out_shape = super(GaussianStochasticLayer, self).call(inputs)
        eps = K.random_normal(shape=out_shape, mean=0.0, stddev=1.0)

        # mean = inputs 
        # stddev = K.ones_like(mean)
        # if self.beta > 0:
        #     mi = gaussian_compression(mean, stddev**2, feature_rank=self.feature_rank)
        #     self.add_loss(self.beta * mi, inputs=inputs)  
        # return K.expand_dims(mean, -(self.feature_rank+1)) + eps

        mu = self.mu 
        sigma = K.exp(self.log_sigma)
        rank = K.int_shape(out_shape)[0]
        for _ in range( rank - self.feature_rank -1):
            mu = K.expand_dims(mu, 0)
            sigma = K.expand_dims(sigma,0)
        mean = mu * inputs 
        stddev = sigma * inputs
        if self.beta > 0:
            mi = gaussian_compression(mean, stddev**2, feature_rank=self.feature_rank)
            self.add_loss(self.beta * mi, inputs=inputs)  
        return K.expand_dims(mean, -(self.feature_rank+1)) + eps * K.expand_dims(stddev, -(self.feature_rank+1))

class ExtendConv2D(Conv2D):
    """Allows input Tensor of rank > 4. 
    
    Note: the convolution still performs on the last three dimension of the input tensor. 
    
    # Input shape
        (batch, n_samples, ..., n_samples, w,h,d) if `data_formats`=`"channels_last"`
        (batch. n_samples, ..., n_samples, d, w,h) if `data_formats`=`"channels_first"`
    """
    def __init__(self, **kwargs):
        super(ExtendConv2D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=None)
    def call(self, inputs):
        # Reshape input to a 4D tensor.
        inp_shape = K.int_shape(inputs)
        inputs_rsp = K.reshape(inputs, (-1,) +  tuple(inp_shape[-3:]))
        output = super(ExtendConv2D, self).call(inputs_rsp)
        # Reshape to match the original input rank. 
        out_shape = (-1,) + tuple(inp_shape[1:-3]) + tuple(K.int_shape(output)[1:])
        return K.reshape(output, out_shape)
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = -3
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=None, axes={channel_axis: input_dim})
        self.built = True
        
    def compute_output_shape(self, input_shape):
        sample_space = (-1,) + tuple(input_shape[1:-3])
        if self.data_format == 'channels_last':
            space = input_shape[-3:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return sample_space + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[-2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return sample_space + (self.filters, ) + tuple(new_space)

# Update CUSTOM_OBJECTS for new customized Keras layers. 
pib.CUSTOM_OBJECTS.update(
    {
        'BernoulliStochasticLayer': BernoulliStochasticLayer,
        'GaussianStochasticLayer': GaussianStochasticLayer
    }
)