#------------------#
#   File: Main.py
#   Author: Soham Deshpande
#   Date: January 2022
#   Description: Main file where modules are assembled
#
#------------------#

from Activation_functions import *
#from  Variable_selection_network import *
from GLU import *
from Dense_Network import *
from Attention_module import *
from Loss_Functions import *
from Positional_Encoder import *
from LSTM import *
from Temporal_Layer import *
from Time_Distributed import *

from PytorchForecasting import *
from Imports import *
class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer

    In this file the different modules are assembled.

    1. Variable Selection Network
    2. LSTM Encoder
    3. Normalisation
    4. GRN
    5. MutiHead Attention
    6. Normalisation
    7. GRN
    8. Normalisation
    9. Dense network
    10.Quantile outputs

    The final model being run is the Pytorch model and not the
    one made from scratch for reasons listed in the writeup

    """



    def __init__(self, n_var_past_cont, n_var_future_cont, n_var_past_disc,
            n_var_future_disc , dim_model, n_quantiles = 3, dropout_r = 0.1,
            n_lstm_layers = 1, n_attention_layers = 1, n_heads = 4):

        super(TemporalFusionTransformer, self).__init__()
        #self.vs_past = VariableSelectionNetwork(n_var_past_cont, n_var_past_disc, dim_model, dropout_r = dropout_r)
        #self.vs_future = VariableSelectionNetwork(n_var_future_cont, n_var_future_disc, dim_model, dropout_r = dropout_r)

        self.enc = LSTM(dim_model, n_layers = n_lstm_layers)
        self.dec = LSTM(dim_model, n_layers = n_lstm_layers)

        self.gate1 = GLU(dim_model)
        self.norm1 = nn.LayerNorm(dim_model)

        self.static_enrich_grn = GRN(dim_model, dropout_r = dropout_r)

        self.attention = []
        for i in range(n_attention_layers):
            self.attention.append([MultiHeadAttentionBlock(dim_model, dim_model, n_heads = n_heads).cuda(),
                                   nn.LayerNorm(dim_model).cuda()])

        self.norm2 = nn.LayerNorm(dim_model)

        self.positionwise_grn = GRN(dim_model, dropout_r = dropout_r)
        self.norm3 = nn.LayerNorm(dim_model)

        self.dropout = nn.Dropout(dropout_r)
        self.fc_out = nn.Linear(dim_model, n_quantiles)

    #takes input (batch_size, past_seq_len, n_variables_past)
    #, (batch_size, future_seq_len, n_variables_future)
    def forward(self, x_past_cont, x_past_disc, x_future_cont, x_future_disc):
        #Encoder
        x_past, vs_weights = self.vs_past(x_past_cont, x_past_disc)

        e, e_hidden = self.enc(x_past)
        self.dec.hidden = e_hidden

        e = self.dropout(e)
        x_past = self.norm1(self.gate1(e) + x_past)

        #Decoder
        x_future, _ = self.vs_future(x_future_cont, x_future_disc)

        d, _ = self.dec(x_future)
        d = self.dropout(d)
        x_future = self.norm1(self.gate1(d) + x_future)

        #Static enrichment
        x = torch.cat((x_past, x_future), dim = 1) #(batch_size, past_seq_len + future_seq_len, dim_model)
        attention_res = x_future
        x = self.static_enrich_grn(x)

        #attention layer
        a = self.attention[0][1](self.attention[0][0](x) + x)
        for at in self.attention[1:]:
            a = at[1](at[0](a) + a)

        x_future = self.norm2(a[:, x_past.shape[1]:] + x_future)

        a = self.positionwise_grn(x_future)
        x_future = self.norm3(a + x_future + attention_res)

        net_out = self.fc_out(x_future)
        return net_out, vs_weights

    def reset(self, batch_size, gpu = True):
        self.enc.reset(batch_size, gpu)
        self.dec.reset(batch_size, gpu)


TFT = tft()
