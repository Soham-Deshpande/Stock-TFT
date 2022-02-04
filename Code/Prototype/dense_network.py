# ---------------------------------------------------#
#
#   File       : Dense_Network.py
#   Author     : Soham Deshpande
#   Date       : November 2021
#   Description: Dense block
#
#
#
#
# ----------------------------------------------------#



import numpy as np

class Dense_Network:
    def __init__(self, input_shape, output_shape, activation, weights, biases):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.weights = weights
        self.biases = biases
        self.output = None
        self.input = None

    def forward_pass(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        self.output = self.activation(self.output)
        return self.output

    def backward_pass(self, error):
        self.error = error
        self.error = self.error * self.activation(self.output, derivative=True)
        self.weights_error = np.dot(self.input.T, self.error)
        self.biases_error = np.sum(self.error, axis=0)
        return self.error

    def update_weights(self, learning_rate):
        self.weights = self.weights - learning_rate * self.weights_error
        self.biases = self.biases - learning_rate * self.biases_error

    def get_weights(self):
        return self.weights



x = np.array([[1,2,3],[4,5,6]])


nn = Dense_Network(x, (2, 2), np.tanh, np.random.randn(2, 3), np.random.randn(2, 1))

for i in range(90):
    nn.forward_pass(x)
    nn.backward_pass(np.array([[1, 1], [1, 1]]))
    nn.update_weights(0.01)
    print(nn.output)
    print(nn.weights)

