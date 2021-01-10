# All Activation Functions here
import numpy as np


class Sigmoid():
    def __int__(self):
        super().__init__()

    def sigmoid_func(self, x):
        return 1 / (1 + np.exp(-x))  # Calculates the sigmoid function

    def sigmoid_derivative(self, x):
        func_x = self.sigmoid_func(x)
        return func_x * (1 - func_x)

    def sigmoid_second_derivative(self, x):
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)
