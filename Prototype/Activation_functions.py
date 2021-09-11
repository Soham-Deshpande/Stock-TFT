# All Activation Functions here
import numpy as np


# sigmoid function
# refer to write up for graph vs computing values
# refer to write up for maths and derivation

class Sigmoid:
    def __int__(self):
        super().__init__()

    def sigmoid_func(self, x):
        # print('The sigmoid derivative not working anymore also ~Soham'), thank you michau
        return 1 / (1 + np.exp(-x))  # Calculates the sigmoid r

    def sigmoid_derivative(self, x):
        func_x = self.sigmoid_func(x)
        func_x = self.sigmoid_func(x)
        return func_x * (1 - func_x)

    def sigmoid_second_derivative(self, x):
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class Softmax:
    def __init__(self):
        super().__init__()

    # explain function here
    def softmax(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]  # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]  # dito
        return e_x / div


class ELU:
    def __init__(self):
        super().__init__()

    def elu(self, z, alpha):
        return z if z >= 0 else alpha * (np.exp(z) - 1)

    def elu_deriv(self, z, alpha):
        return 1 if z > 0 else alpha * np.exp(z)


class Sinh:
    def __init__(self):
        super().__init__()

    def sinh(self,x):
        #doublesinh = np.exp(x) – np.exp(-x)
        comp1 = np.exp(x)
        comp2 = np.exp(-x)
        comp3 = 0.5 * (comp1-comp2)
        return comp3

    def sinh_deriv(self,x):
        # doublesinh = np.exp(x) – np.exp(-x)
        comp1 = np.exp(x)
        comp2 = np.exp(-x)
        final = 0.5 * (comp1 + comp2)
        return final




x = Sinh()
print(x.sinh_deriv(1))
const = 1
tanh = (x.sinh(const))/(x.sinh_deriv(const))
print(tanh)
#dont need cosh function as deriv of sinh is cosh
# tanh can be worked out by doing sinh / d/dx{sinh}

