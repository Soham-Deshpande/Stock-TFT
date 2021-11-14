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
    def __init__(self, input_size,output_size):
        super().__init__()

        self.matrixa = nn.Linear(input_size,input_size) #setting up matrix A
        self.matrixb = nn.Linear(input_size,input_size) #setting up matrix B
        self.Sigmoid = Sigmoid()

    def forward(self,x):
        a = self.matrixa(x)
        b = self.Sigmoid(self.matrixb(x))

        return a*b






#
# class GLU:
#     """
#      Gated Linear Unit
#
#     h(x) = {(x*w + b)}*[tensor product]*{sigmoid[(x*w + c)]}
#
#     Args:
#         size :
#
#     """
#     def __init__(self, input_size, hidden_layer_size, dropout_rate):
#         super().__init__()
#         self.inp_size = input_size
#         self.hidden_layer_size = hidden_layer_size
#         self.dropout_rate = dropout_rate
#         self.linear = nn.linear(input_size,input_size*2)
#         self.sigmoid = Sigmoid()
#
#     def forward(self,x):
#         out = self.linear(x)
#         return out[:, :self.inp_size] * self.sigmoid.sigmoid_func(out[:, self.inp_size:])
#
# class GRN:

        


