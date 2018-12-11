import numpy as np
import keras.backend as K

def kl_bernoulli(p, q):
    """Computes KL divergence between Bernoulli distributions. 
    
    # Arguments
        p: A tensor with the same shape as `q` (broadcasting is possble), encoding Bernoulli distributions.  
        q: A tensor, encoding Bernoulli distributions. 
    
    # Returns
        A tensor with the same shape as `q`
    """
    return -K.binary_crossentropy(p, p) + K.binary_crossentropy(p, q)

def variational_compression(sampled_cond_dist, prior_dist):
    """Computes the variational compression in our paper. 
        
    # Arguments
        sampled_cond_dist: A Tensor of the same shape as `prior_dist` (broadcasting is possible)
        prior_dist: A Tensor
        
    # Returns
        A Tensor
    """
    dim = sampled_cond_dist.get_shape().as_list()[-1]
    sampled_cond_dist = K.reshape(sampled_cond_dist, (-1, dim))
    prior_dist = K.expand_dims(prior_dist, 0)
    return K.mean(K.sum(kl_bernoulli(sampled_cond_dist, prior_dist), -1))

def gaussian_compression(mu1, sigma1, mu2=None, sigma2=None, feature_rank=1):
    """Computes the Gaussian variational compression. 

    # Arguments
        mu1: A `Tensor`.
        sigma1: A `Tensor` of the same shape as `mu1`.
        mu2: A `Tensor` of the same rank as `mu1` where the last `feature_rank` dimensions are the same as those of `mu1` 
            while the other dimensions (so-called sample dimensions) can be either the same as those of `mu1` or 1s. 
        sigma2: A `Tensor` of the same shape as `mu2`. 
        feature_rank: A Integer. 

    # Returns
        A scalar. 
    """
    if mu2 is None or sigma2 is None:
        mu2 = K.zeros_like(mu1)
        sigma2 = K.ones_like(sigma1)
        # epsilon = 0.
    # else:
    epsilon = K.common.epsilon()

    output = 0.5 * (((mu1 - mu2)**2 + sigma1**2) / (sigma2**2 + epsilon) - 2 * K.log(sigma1 + epsilon) + 2*K.log(sigma2 + epsilon) - 1 )
    for _ in range(feature_rank):
        output = K.sum(output, axis=-1)
    return K.mean(output) 

def bernoulli_compression(p, q, feature_rank=1):
    """Computes the Bernoulli variational compression. 
        
    # Arguments
        p: A `Tensor` that has the same last `feature_rank` dimensions as `q`.
        q: A `Tensor` of rank `feature_rank`.
        
    # Returns
        A scalar. 
        
    """
    inp_shape = K.int_shape(p)
    feature_dim = int(np.prod(inp_shape[-feature_rank:]))
    p = K.reshape(p, (-1, feature_dim))
    q = K.reshape(q, (-1, feature_dim))
    kl =  -K.binary_crossentropy(p, p) + K.binary_crossentropy(p, q)
    return K.mean(K.sum(kl, -1))