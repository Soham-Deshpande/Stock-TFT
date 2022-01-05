# ---------------------------------------------------#
#
#   File       : GLU.py
#   Author     : Soham Deshpande
#   Date       : December 2021
#   Description: Gated Residual Network
#
#
#
# ----------------------------------------------------#

from Activation_functions import ELU, Sigmoid
from torch import nn



class GRN(nn.Module):

    """
    Gated Residual Network
    
    GRN(x) = LayerNorm(a + GLU(Linear(a)))
         
    Args:
       int   input_size  : Size of the input tensor
       int   hidden_size : Size of the hidden layer
       int   output_size : Size of the output layer
       float   dropout   : Fraction between 0 and 1 showing the dropout rate
       int   context_size: Size of the context vector
       bool  is_temporal : Decides if the Temporal Layer has to be used or not

    """

    def __init__(self, input_size,hidden_size, output_size, dropout, context_size=None, is_temporal=True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout     = dropout
        self.is_temporal = is_temporal


