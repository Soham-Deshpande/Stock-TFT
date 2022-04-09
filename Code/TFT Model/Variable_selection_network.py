#----------------------------------------------------#
#
#   File       : VariableSelectionNetwork.py
#   Author     : Soham Deshpande
#   Date       : January 2022
#   Description: VariableSelection Network
#
#
#
# ----------------------------------------------------#

from Imports import *
#from GRN import *
import torch
import torch.nn as nn
class VariableSelectionNetwork(nn.Module):

    """
    VariableSelectionNetwork

    VRN(x) = GRN(x) x GRN(x) x Softmax(GRN(x))

    Args:
        int input_size : Size of input tensor
        int hidden_size: Size of the hidden layer
        int output_size: Size of the output layer
        float dropout  : Fraction between 0 and 1 showing the dropout rate

    """

    def __init__(self, input_size, output_size, hidden_size,dropout):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout    = dropout
        self.flattened_inputs = nn.GRN(self.output_size*self.input_size,
                                                     self.hidden_size, self.output_size,
                                                     self.dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.transformed_inputs = nn.ModuleList([nn.GRN(
                self.input_size, self.hidden_size, self.hidden_size,
                self.dropout) for i in range(self.output_size)])

    def forward(self, embedding, context=None):
        """
        Args:
          embedding (torch.tensor): Entity embeddings for categorical variables and linear
                     transformations for continuous variables.
          context (torch.tensor): The context is obtained from a static covariate encoder and
                   is naturally omitted for static variables as they already
                   have access to this
        """

        # Generation of variable selection weights
        sparse_weights = self.flattened_inputs(embedding, context)
        if self.is_temporal:
            sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        else:
            sparse_weights = self.softmax(sparse_weights).unsqueeze(1)

        # Additional non-linear processing for each feature vector
        transformed_embeddings = torch.stack(
            [self.transformed_inputs[i](embedding[
                Ellipsis, i*self.input_size:(i+1)*self.input_size]) for i in range(self.output_size)], axis=-1)

        # Processed features are weighted by their corresponding weights and combined
        combined = transformed_embeddings*sparse_weights
        combined = combined.sum(axis=-1)

        return combined, sparse_weight


#vsn = VariableSelectionNetwork(2,2,4,0.1)
#x = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
#y = torch.tensor(np.array([[1, 2, 3]]))

#print(vsn.forward(1))
