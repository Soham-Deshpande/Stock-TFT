#Feed Forward here

import math
import numpy as np


from Activation_functions import Sigmoid

class Model:
    def __init__(self, input, hidden, output):
        self.input = input + 1 # one extra for the bias node
        self.output = output
        self.layers = hidden

    def feed_forward(self):
        #feed the neurons and data forward to the next layer
        pass


    def backpropagation(self):
        pass
    #finish by 25th see threats
