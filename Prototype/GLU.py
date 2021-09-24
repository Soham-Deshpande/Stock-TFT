# ---------------------------------------------------#
#
#   File       : GLU.py
#   Author     : Soham Deshpande
#   Date       : July 2021
#   Description: Gated Linear Unit
#
#
#
# ----------------------------------------------------#
from Activation_functions import Sigmoid
from torch import nn

class GLU():
    def __init__(self, input_size):
        super().__init__()

        self.matrixa = nn.Linear(input_size,input_size) #setting up matrix A
        self.matrixb = nn.Linear(input_size,input_size) #setting up matrix B
        self.Sigmoid = Sigmoid()

    def forward(self):
        pass

        


