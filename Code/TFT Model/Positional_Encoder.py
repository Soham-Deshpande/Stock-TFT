#----------------------------------------------------#
#
#   File       : Positional_Encoder.py
#   Author     : Soham Deshpande
#   Date       : February 2022
#   Description: Positional Encoder
#
#
#
# ----------------------------------------------------#

import torch
import math
import torch.nn as nn

class PositionalEncoder():
    """
    PositionalEncoder

    Positional Encode to assist the time indexes when training the model
    Refer to write up for further explanation

    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    """

    def __init__(self, d_model, height, width, max_len=5000):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        self.d_model = d_model
        self.height = height
        self.width = width
        self.max_len = max_len
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.pe = pe

        #print(pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

    def get_embedding(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

    def get_embedding_layer(self):
        return nn.Embedding(self.max_len, self.d_model)







