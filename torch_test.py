import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_test_utils import *


class GLU(torch.nn.Module):
    def __init__(self, dim_input):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_input)
        self.fc2 = nn.Linear(dim_input, dim_input)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x)) * self.fc2(x)


class GRN(torch.nn.Module):
    def __init__(self, dim_input, dim_out=None, n_hidden=10, dropout_r=0.1):
        super(GRN, self).__init__()

        if (dim_out != None):
            self.skip = nn.Linear(dim_input, dim_out)
        else:
            self.skip = None
            dim_out = dim_input

        self.fc1 = nn.Linear(dim_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, dim_out)
        self.dropout = nn.Dropout(dropout_r)

        self.gate = GLU(dim_out)

        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        a = F.elu(self.fc1(x))
        a = self.dropout(self.fc2(a))

        a = self.gate(a)

        if (self.skip != None):
            return self.norm(self.skip(x) + a)
        return self.norm(x + a)


class VSN(torch.nn.Module):
    def __init__(self, n_var_cont, n_var_disc, dim_model, dropout_r=0.1):
        super(VSN, self).__init__()
        n_var_total = n_var_cont + len(n_var_disc)

        # Linear transformation of inputs into dmodel vector
        self.linearise = []
        for i in range(n_var_cont):
            self.linearise.append(nn.Linear(1, dim_model, bias=False).cuda())
        self.fc = nn.Linear(1, dim_model, bias=False).cuda()

        # entity embedings for discrete inputs
        self.entity_embed = []
        for i in n_var_disc:
            self.entity_embed.append(nn.Linear(i, dim_model, bias=False).cuda())

        self.input_grn = GRN(dim_model, dropout_r=dropout_r)
        self.vs_grn = GRN(n_var_total * dim_model, dim_out=n_var_total, dropout_r=dropout_r)

    # takes input (batch_size, seq_len, n_variables, input_size)
    def forward(self, x_cont, x_disc):
        # linearise continuous inputs
        linearised = []
        for idx, fc in enumerate(self.linearise):
            linearised.append(fc(x_cont[:, :, idx]))

        # entity embeddings for discrete inputs
        embedded = []
        for x, fc in zip(x_disc, self.entity_embed):
            embedded.append(fc(x))

        if (len(self.linearise) != 0 and len(self.entity_embed) != 0):
            linearised = torch.stack(linearised, dim=-2)
            embedded = torch.stack(embedded, dim=-2)
            vectorised_vars = torch.cat((linearised, embedded),
                                        dim=-2)  # (batch_size, seq_len, dim_model, n_vars_total)
        elif (len(self.linearise) != 0 and len(self.entity_embed) == 0):
            vectorised_vars = torch.stack(linearised, dim=-2)  # (batch_size, seq_len, n_var_cont, dim_model)
        elif (len(self.entity_embed) != 0 and len(self.linearise) == 0):
            vectorised_vars = torch.stack(embedded, dim=-2)  # (batch_size, seq_len, n_var_disc, dim_model)

        # flatten everything except across batch for variable selection weights
        vs_weights = self.vs_grn(vectorised_vars.flatten(start_dim=2))  # (batch_size, seq_len, n_variables)
        vs_weights = torch.softmax(vs_weights, dim=-1).unsqueeze(-1)  # (batch_size, seq_len, n_variables, 1)

        # input_grn applied to every input separatly
        input_weights = self.input_grn(vectorised_vars)  # (batch_size, seq_len, n_variables, dim_model)

        x = torch.sum((vs_weights * input_weights), dim=2)
        return x, vs_weights  # returns(batch_size, seq_len, dim_model)


class LSTMLayer(torch.nn.Module):
    def __init__(self, dim_model, n_layers=1, dropout_r=0.1):
        super(LSTMLayer, self).__init__()
        self.n_layers = n_layers
        self.dim_model = dim_model

        self.lstm = nn.LSTM(dim_model, dim_model, num_layers=n_layers, batch_first=True)
        self.hidden = None

        self.dropout = nn.Dropout(dropout_r)

    # takes input (batch_size, seq_len, dim_model)
    def forward(self, x):
        if (self.hidden == None):
            raise Exception("Call reset() to initialise LSTM Layer")

        x, self.hidden = self.lstm(x, self.hidden)
        x = self.dropout(x)

        return x, self.hidden  # returns (batch_size, seq_len, dim_model), hidden

    def reset(self, batch_size, gpu=True):
        if (not gpu):
            dtype = torch.FloatTensor
        else:
            dtype = torch.cuda.FloatTensor
        self.hidden = (torch.zeros([self.n_layers, batch_size, self.dim_model]).type(dtype),
                       torch.zeros([self.n_layers, batch_size, self.dim_model]).type(dtype))


class TFN(torch.nn.Module):
    def __init__(self, n_var_past_cont, n_var_future_cont, n_var_past_disc, n_var_future_disc
                 , dim_model, n_quantiles=3, dropout_r=0.1, n_lstm_layers=1, n_attention_layers=1, n_heads=4):
        super(TFN, self).__init__()
        self.vs_past = VSN(n_var_past_cont, n_var_past_disc, dim_model, dropout_r=dropout_r)
        self.vs_future = VSN(n_var_future_cont, n_var_future_disc, dim_model, dropout_r=dropout_r)

        self.enc = LSTMLayer(dim_model, dropout_r=dropout_r, n_layers=n_lstm_layers)
        self.dec = LSTMLayer(dim_model, dropout_r=dropout_r, n_layers=n_lstm_layers)

        self.gate1 = GLU(dim_model)
        self.norm1 = nn.LayerNorm(dim_model)

        self.static_enrich_grn = GRN(dim_model, dropout_r=dropout_r)

        self.attention = []
        for i in range(n_attention_layers):
            self.attention.append([MultiHeadAttentionBlock(dim_model, dim_model, n_heads=n_heads).cuda(),
                                   nn.LayerNorm(dim_model).cuda()])

        self.norm2 = nn.LayerNorm(dim_model)

        self.positionwise_grn = GRN(dim_model, dropout_r=dropout_r)
        self.norm3 = nn.LayerNorm(dim_model)

        self.dropout = nn.Dropout(dropout_r)
        self.fc_out = nn.Linear(dim_model, n_quantiles)

    # takes input (batch_size, past_seq_len, n_variables_past)
    # , (batch_size, future_seq_len, n_variables_future)
    def forward(self, x_past_cont, x_past_disc, x_future_cont, x_future_disc):
        # Encoder
        x_past, vs_weights = self.vs_past(x_past_cont, x_past_disc)

        e, e_hidden = self.enc(x_past)
        self.dec.hidden = e_hidden
        print(e)

        e = self.dropout(e)
        x_past = self.norm1(self.gate1(e) + x_past)

        # Decoder
        x_future, _ = self.vs_future(x_future_cont, x_future_disc)

        d, _ = self.dec(x_future)
        d = self.dropout(d)
        x_future = self.norm1(self.gate1(d) + x_future)

        # Static enrichment
        x = torch.cat((x_past, x_future), dim=1)  # (batch_size, past_seq_len + future_seq_len, dim_model)
        attention_res = x_future
        x = self.static_enrich_grn(x)

        # attention layer
        a = self.attention[0][1](self.attention[0][0](x) + x)
        for at in self.attention[1:]:
            a = at[1](at[0](a) + a)

        x_future = self.norm2(a[:, x_past.shape[1]:] + x_future)

        a = self.positionwise_grn(x_future)
        x_future = self.norm3(a + x_future + attention_res)

        net_out = self.fc_out(x_future)
        return net_out, vs_weights

    def reset(self, batch_size, gpu=True):
        self.enc.reset(batch_size, gpu)
        self.dec.reset(batch_size, gpu)