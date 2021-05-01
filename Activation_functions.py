# All Activation Functions here
import numpy as np

class Sigmoid():
    def __int__(self):
        super().__init__()

    def sigmoid_func(self, x):
        #print('The sigmoid derivative not working anymore also ~Soham')
        return 1 / (1 + np.exp(-x))  # Calculates the sigmoid r


    def sigmoid_derivative(self, x):
        func_x = self.sigmoid_func(x)
        return func_x * (1 - func_x)

    def sigmoid_second_derivative(self, x):
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)

