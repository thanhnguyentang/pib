# PIB: Parametric Information Bottleneck  
[PIB (Parametric Information Bottleneck)](https://arxiv.org/abs/1712.01272) is a framework that fully incorporates at layer level the Information Bottleneck principle into the training of a stochastic neural network.

## Setup   
`pip install -r requirement.txt`    
`python setup.py install`  

## Train and Test  
`python config.py --mode=train`   
`python config.py --mode=test`

## Changelog   
### Version 1.2 (2018/10)
* Migated to Keras with improved readability, added Bernoulli and Gaussian PIB layers, and  modified Keras training routines (which is helpful for training stochastic neural networks in general).  
### Version 1.1 (2018/05)
* Migated to native Tensorflow. 
### Version 1.0 (2017/10)  
* Initial development based on Theano. 

*A placeholder for PIB code release.*       
Contact: nguyent2792 AT gmail DOT com  
