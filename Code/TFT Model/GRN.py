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
from Imports import nn
from Temporal_Layer import *
from GLU import *

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


    This unit controls how much of the original input is used. It can skip over
    layers where the GLU output might be close to 0.
    When there is no context vector present, the GRN will treat the input as 0.
    """

    def __init__(self, input_size,hidden_size, output_size, dropout,
            context_size=None, is_temporal=True):
        super().__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.dropout      = dropout
        self.is_temporal  = is_temporal
        self.context_size = context_size

        if self.input_size != self.output_size:
                self.skip_layer = TemporalLayer(nn.Linear(self.input_size,
                    self.output_size))

            # Context vector c
        if self.context_size != None:
            self.c = TemporalLayer(nn.Linear(self.context_size,
                self.hidden_size, bias=False))

        # Dense & ELU
        self.dense1 = TemporalLayer(nn.Linear(self.input_size,
            self.hidden_size))
        self.elu = nn.ELU()

        # Dense & Dropout
        self.dense2 = TemporalLayer(nn.Linear(self.hidden_size,
            self.output_size))
        self.dropout = nn.Dropout(self.dropout)

        # Gate, Add & Norm
        self.gate = TemporalLayer(GLU(self.output_size))
        self.layer_norm = TemporalLayer(nn.BatchNorm1d(self.output_size))

    def forward(self, x):
        a = F.elu(self.c(x))
        a = self.dropout(self.fc2(a))

        a = self.gate(a)

        if(self.skip != None):
            return self.norm(self.skip(x) + a)
        return self.norm(x + a)
