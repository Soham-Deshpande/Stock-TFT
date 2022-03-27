# ---------------------------------------------------#
#
#   File       : Loss_Functions.py
#   Author     : Soham Deshpande
#   Date       : December 2021
#   Description: Quantile Loss, Normalised Quantile Loss
#                and Poisson Loss
#
#
# ----------------------------------------------------#

import Activation_functions
from Imports import *
import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    """
   Quantile Loss

   Used to predict intervals rather than just points. This gives the user the
   uncertainty levels. This loss function aims to give different penalties to
   overestimation and underestimation based on the value of the chosen quantile


   QuantileLoss(pred, outcome) = max{q(pred-outcome),(q-1)(pred-outcome)}

    Args:
       int   pred    : Tensor with predictions
       int   outcome : Tensor with outcomes
       int   quantile: Quantile percentage
       float loss    : Output float with loss value
    """

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def loss(self, pred,outcome, quantile):
        assert quantile > 0.0 and quantile < 1.0
        delta = outcome - pred
        loss = quantile * F.relu(delta) + (1.0 - quantile) * F.relu(-delta)
        return loss.unsqueeze(1)

    def forward(self, pred, gt):
        loss = []
        for i, q in enumerate(self.quantiles):
            loss.append(
                self.loss(pred[:, :, i], outcome[:, :, i], q)
            )
        loss = torch.mean(torch.sum(torch.cat(loss, axis=1), axis=1))
        return loss



class NormalisedQuantileLoss(nn.Module):

    def forward(self, pred,outcome, quantile):
        assert quantile > 0.0 and quantile < 1.0
        delta = outcome - pred
        weighted_errors = quantile * F.relu(delta) + (1.0 - quantile) * F.relu(-delta)
        quantile_loss = weighted_errors.mean()
        normaliser = outcome.abs().mean() + 1e-9
        return 2 * quantile_loss / normaliser




class PoissonLoss(nn.Module):

    """
   Poisson Loss

   Output describes the mean of the Poisson distribution and target is a sample
   from the distribution

   PoissonLoss() = \frac{1}{N\times\sum{outcome - pred\times\ln{outcome}}}

   Args:
       int   pred    : Tensor with predictions
       int   outcome : Tensor with outcomes
       float loss    : Output float with loss value

    """
    def __init__():
        self.neuron = neuron
        self.avg = avg
        self.bias = bias

    def forward(self, pred, outcome):
        target = target.detach()
        loss = outcome - pred * torch.log(outcome + self.bias)
        if not self.neuron:
            return loss.mean() if self.avg else loss.sum()
        else:
            loss = loss.view(-1, loss.shape[-1])
            return loss.mean(dim=0) if self.avg else loss.sum(dim=0)




