# ---------------------------------------------------#
#
#   File       : Layer_Normalisation.py
#   Author     : Soham Deshpande
#   Date       : December 2021
#   Description: Layer Normalisation
#
#
#
# ----------------------------------------------------#
from Imports import nn

class LayerNormalisation(nn.Module):
    """
    Layer Normalisation
    y = \frac{x - E(x)}{\sqrt{Var(x) + \eta}}\times\gamma+\beta

    The mean and standard deviation are calculated over the last
    'D' dimensions where D is the dimension of the input shape
    Layer normalisation is applied to each element unlike batch or
    instance normalisation

    """
    def __init__(self, shape, eps, elementwise_affine= True):
        super(LayerNormalisation,self).__init__()
        if isinstance(shape):
            self.normalized_shape = tuple(shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if self.elementwise_affine:
                self.weight = Parameter(torch.empty(self.shape))
            self.bias = Parameter(torch.empty(self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()


    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

