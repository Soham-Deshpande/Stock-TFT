# ---------------------------------------------------#
#
#   File       : GLU.py
#   Author     : Soham Deshpande
#   Date       : July 2021
#   Description: Gated Residual Network
#
#
#
# ----------------------------------------------------#

from Activation_functions import ELU, Sigmoid
from torch import nn

def GLU():
    return nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256),
        nn.PReLU(),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.PReLU(),
        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256),
        nn.PReLU(),
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.PReLU()
    )