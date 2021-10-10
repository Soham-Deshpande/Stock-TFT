# softmax function
import numpy as np


class Softmax():
    def __int__(self):
        super().__init__()

    def softmax(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]  # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] # dito
        return e_x / div

#doesnt work, you have to do a step 2 question if you dont complete by november