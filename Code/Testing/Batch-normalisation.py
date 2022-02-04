import theano
import theano.tensor as T
import numpy as np
import math
import time


class BatchNormalization(object):
    def __init__(self, input_shape, mode=0, momentum=0.9):
        '''
        # params :
        input_shape :
            when mode is 0, we assume 2D input. (mini_batch_size, # features)
            when mode is 1, we assume 4D input. (mini_batch_size, # of channel, # row, # column)
        mode :
            0 : feature-wise mode (normal BN)
            1 : window-wise mode (CNN mode BN)
        momentum : momentum for exponential average
        '''
        self.input_shape = input_shape
        self.mode = mode
        self.momentum = momentum
        self.run_mode = 0  # run_mode : 0 means training, 1 means inference

        self.insize = input_shape[1]

        # random setting of gamma and beta, setting initial mean and std
        rng = np.random.RandomState(int(time.time()))
        self.gamma = theano.shared(np.asarray(
            rng.uniform(low=-1.0 / math.sqrt(self.insize), high=1.0 / math.sqrt(self.insize), size=(input_shape[1])),
            dtype=theano.config.floatX), name='gamma', borrow=True)
        self.beta = theano.shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name='beta', borrow=True)
        self.mean = theano.shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name='mean', borrow=True)
        self.var = theano.shared(np.ones((input_shape[1]), dtype=theano.config.floatX), name='var', borrow=True)

        # parameter save for update
        self.params = [self.gamma, self.beta]

    def set_runmode(self, run_mode):
        self.run_mode = run_mode

    def get_result(self, input):
        # returns BN result for given input.
        epsilon = 1e-06

        if self.mode == 0:
            if self.run_mode == 0:
                now_mean = T.mean(input, axis=0)
                now_var = T.var(input, axis=0)
                now_normalize = (input - now_mean) / T.sqrt(now_var + epsilon)  # should be broadcastable..
                output = self.gamma * now_normalize + self.beta
                # mean, var update
                self.mean = self.momentum * self.mean + (1.0 - self.momentum) * now_mean
                self.var = self.momentum * self.var + (1.0 - self.momentum) * (
                            self.input_shape[0] / (self.input_shape[0] - 1) * now_var)
            else:
                output = self.gamma * (input - self.mean) / T.sqrt(self.var + epsilon) + self.beta

        else:
            # in CNN mode, gamma and beta exists for every single channel separately.
            # for each channel, calculate mean and std for (mini_batch_size * row * column) elements.
            # then, each channel has own scalar gamma/beta parameters.
            if self.run_mode == 0:
                now_mean = T.mean(input, axis=(0, 2, 3))
                now_var = T.var(input, axis=(0, 2, 3))
                # mean, var update
                self.mean = self.momentum * self.mean + (1.0 - self.momentum) * now_mean
                self.var = self.momentum * self.var + (1.0 - self.momentum) * (
                            self.input_shape[0] / (self.input_shape[0] - 1) * now_var)
            else:
                now_mean = self.mean
                now_var = self.var
            # change shape to fit input shape
            now_mean = self.change_shape(now_mean)
            now_var = self.change_shape(now_var)
            now_gamma = self.change_shape(self.gamma)
            now_beta = self.change_shape(self.beta)

            output = now_gamma * (input - now_mean) / T.sqrt(now_var + epsilon) + now_beta

        return output

    # changing shape for CNN mode
    def change_shape(self, vec):
        return T.repeat(vec, self.input_shape[2] * self.input_shape[3]).reshape(
            (self.input_shape[1], self.input_shape[2], self.input_shape[3]))
