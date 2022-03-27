#----------------------------------------------------#
#
#   File       : TemporalLayer.py
#   Author     : Soham Deshpande
#   Date       : January 2022
#   Description: Temporal Layer
#
#
#
# ----------------------------------------------------#

from Imports import nn


class TemporalLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        An implitation of the TimeDistributed layer used in Tensorflow.
        Applied at every temporal slice of an input

        """
        self.module = module


    def forward(self, x):
        """
        Args:
            x (torch.tensor): Tensor with time steps to pass through the same layer.
        """
        t, n = x.size(0), x.size(1)
        x = x.reshape(t * n, -1)
        x = self.module(x)
        x = x.reshape(t, n, x.size(-1))

        return x


