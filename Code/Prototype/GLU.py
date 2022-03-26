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

class GLU_prototype():
    """
    GLU Prototype

    GLU built from scratch to test funcionality and gain a deeper understanding
    of the system


    """
    def __init__(self, input_size,output_size):
        super().__init__()

        self.matrixa = nn.Linear(input_size,input_size) #setting up matrix A
        self.matrixb = nn.Linear(input_size,input_size) #setting up matrix B
        self.Sigmoid = Sigmoid()

    def forward(self,x):
        a = self.matrixa(x)
        b = self.Sigmoid(self.matrixb(x))

        return a*b



class GLU(nn.Module):
    """
    Gated Linear Unit

    GLU(x,y) = multiply(x, sigmoid(y))

    Args:
        int input size: Defines the size of the input matrix and output size of
        the gate

    """

    def __init__(self,input_size):
        super().__init__()
        #input
        self.x = nn.Linear(input_size,input_size) # construct matrix 1

        #Gate
        self.y = nn.Linear(input_size, input_size) # construct matrix 2
        self.sigmoid = nn.Sigmoid() # construct sigmoid function


    def forward(self,a):
        """
        Args:
            float(tensor) a: Tensor that passes through the gate

        """
        gate = self.sigmoid(self.y(x))
        x = self.x(a)

        return torch.mul(gate, x) #multiply both tensors together




