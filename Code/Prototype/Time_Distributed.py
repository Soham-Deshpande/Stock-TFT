# ---------------------------------------------------#
#
#   File       : Time_Distributed.py
#   Author     : Soham Deshpande
#   Date       : July 2021
#   Description: A wrapper that applies to a layer to
#                every temporal slice of an input
#
#

# ----------------------------------------------------#


import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y




#credit to https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4