# batch normalisation

# forward pass of batch normalisation
# input : Values of x over a mini-batch B = {x1..xm}
#         parameters to be learned gamma, beta
# output: {y1 = BNgamma, beta(xi)}
#refer to https://towardsdatascience.com/implementing-batch-normalization-in-python-a044b0369567


import numpy as np

def batchnormforward(x, gamma, beta, eps=1e-5):
    """
       Forward pass for batch normalization.
       During training the sample mean and (uncorrected) sample variance are
       computed from minibatch statistics and used to normalize the incoming data.
       During training we also keep an exponentially decaying running mean of the
       mean and variance of each feature, and these averages are used to normalize
       data at test-time.
       At each timestep we update the running averages for mean and variance using
       an exponential decay based on the momentum parameter:
       running_mean = momentum * running_mean + (1 - momentum) * sample_mean
       running_var = momentum * running_var + (1 - momentum) * sample_var

       Input:
       - x: Data of shape (N, D)
       - gamma: Scale parameter of shape (D,)
       - beta: Shift paremeter of shape (D,)
       - bn_param: Dictionary with the following keys:
         - mode: 'train' or 'test'; required
         - eps: Constant for numeric stability
         - momentum: Constant for running mean / variance.
         - running_mean: Array of shape (D,) giving running mean of features
         - running_var Array of shape (D,) giving running variance of features
       Returns a tuple of:
       - out: of shape (N, D)
       - cache: A tuple of values needed in the backward pass
       """



    N,D = x.shape

    sample_mean = x.mean(axis = 0)
    sample_var = x.var(axis=0)

    std = np.sqrt(sample_var + eps)
    x_centered = x- sample_mean
    out = gamma * x_norm + beta

    cache = (x_norm, x_centered, std, gamma)

    return out, cache


