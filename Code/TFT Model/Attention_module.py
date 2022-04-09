#----------------------------------------------------#
#
#   File       : Attention Module.py
#   Author     : Soham Deshpande
#   Date       : January 2022
#   Description: Interpretable MultiHead Attention
#
#
#
# ----------------------------------------------------#

#from d2l import torch as d2l
import torch.nn as nn
import torch
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention

    Implement Multihead attention
    Accept tensors as inputs
    Output an attention score

    1. Linear -> Tanh
    2. Unsqueze
    3. Softmax
    4. Unsqueeze
    5. Repeat
    6. Transpose
    7. Output Tensor

    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        #self.attention = torch.dot(dropout)
        self.attention = dropout
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)#define query vector
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)#define key vector
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)#define value vector
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            #repeats elements in the tensor
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


    def transpose_qkv(X, num_heads):
        """
        Transposition allows for parallel computation of multiple
        attention heads.

        """
        # Shape of input `X`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
        # Shape of output `X`:
        # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
        # `num_hiddens` / `num_heads`)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

        # Shape of output `X`:
        # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        X = X.permute(0, 2, 1, 3)

        # Shape of `output`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        return X.reshape(-1, X.shape[2], X.shape[3])


    def transpose_output(X, num_heads):
        """
        Reverse the operation of `transpose_qkv`
        """
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


#import numpy as np
#v = torch.Tensor([[1, 2, 3], [4, 5, 6]])
#print(v)

#x = MultiHeadAttention(2,2,2,2,1,v)
#print(x.forward)
#print(x.transpose_qkv)
#print(x.transpose_output)
