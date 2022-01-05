import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m, -1)


def attention(Q, K, V):
    # Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K)  # (batch_size, dim_attn, seq_length)
    return torch.matmul(a, V)  # (batch_size, seq_length, seq_length)


class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)

    def forward(self, x, kv=None):
        if (kv is None):
            # Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))

        # Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))


class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))

        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        a = []
        for h in self.heads:
            a.append(h(x, kv=kv))

        a = torch.stack(a, dim=-1)  # combine heads
        a = a.flatten(start_dim=2)  # flatten all head outputs

        x = self.fc(a)

        return x


class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_val, bias=False).cuda()

    def forward(self, x):
        return self.fc1(x)


class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False).cuda()

    def forward(self, x):
        return self.fc1(x)


class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False).cuda()

    def forward(self, x):
        return self.fc1(x)


def QuantileLoss(net_out, Y, q):
    return (q * F.relu(net_out - Y)) + ((1 - q) * F.relu(Y - net_out))


from torch_test_data import one_hot


def forward_pass(model, data_gen, batch_size, quantiles, gpu=True):
    model.reset(batch_size, gpu=gpu)

    # Get input and target data, one-hot encode discrete variables, continuous variables have already been normalized
    in_seq_continuous, in_seq_discrete, future_in_seq_discrete, target_seq = next(data_gen)
    in_seq_discrete = one_hot(in_seq_discrete, [24, 31, 12])
    future_in_seq_discrete = one_hot(future_in_seq_discrete, [24, 31, 12])

    # forward pass
    net_out, vs_weights = model(in_seq_continuous, in_seq_discrete, None, future_in_seq_discrete)
    loss = torch.mean(QuantileLoss(net_out, target_seq, quantiles))

    return loss, net_out, vs_weights, (in_seq_continuous, in_seq_discrete, future_in_seq_discrete, target_seq)
