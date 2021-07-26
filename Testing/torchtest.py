import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler

class GLU(nn.Module):
    def __init__(self):

