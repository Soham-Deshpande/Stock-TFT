import pandas as pd
import datetime
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import adfuller
import numpy as np
import time

import random

import plotly.graph_objects as go

header = {'open time': 1, 'open': 2, 'high': 3, 'low': 4, 'close': 5}


# load_data from file
def load_data(year, symbol, con_cols, interval="1m", normalise=True):
    if (type(year) != type([])):
        year = [year]

    if (type(symbol) != type([])):
        symbol = [symbol]

    frames = []
    for y in year:
        for s in symbol:
            data = pd.read_csv("data_{}/{}_{}.csv".format(interval, y, s))
            frames.append(data)

    data_ = pd.concat(frames)

    print("done")
    # convert timestamp into month and day numbers
    data_['Hour'] = pd.to_datetime(data_["Open Time"], unit='ms').apply(lambda x: x.hour)
    data_['Day'] = pd.to_datetime(data_["Open Time"], unit='ms').apply(lambda x: x.day - 1)
    data_['Month'] = pd.to_datetime(data_["Open Time"], unit='ms').apply(lambda x: x.month - 1)
    return data_


class Indexer():
    def __init__(self, r_bottom, r_top, batch_size, random=True, increment=1):
        self.r_bottom = r_bottom
        self.r_top = r_top
        self.random = random
        self.increment = increment
        self.batch_size = batch_size
        self.indices = [0]
        self.next()

    def next(self):
        if (self.random):
            new_indices = []
            for b in range(self.batch_size):
                new_indices.append(random.randrange(self.r_bottom, self.r_top))
            self.indices = new_indices
        else:
            new_indices = [self.indices[-1]]

            for b in range(1, self.batch_size):
                i = new_indices[-1] + self.increment
                if (i >= self.r_top):
                    new_indices.append((i - self.top) + self.r_bottom)
                else:
                    new_indices.append(i)
            self.indices = new_indices

        return self.indices


def get_batches(data_, in_seq_len, out_seq_len, con_cols, disc_cols, target_cols, batch_size=1, gpu=True,
                normalise=True, indexer=None, norm=None):
    data = data_.copy()

    given_indexer = True
    if indexer is None:
        given_indexer = False
        indexer = Indexer(1, data.shape[0] - (in_seq_len + out_seq_len + 1), batch_size)

    if normalise:
        if norm is None:
            norm = data
        data[con_cols] = (data[con_cols] - norm[con_cols].stack().mean()) / norm[con_cols].stack().std()

    # convert columns indices from dataframe to numpy darray
    con_cols = [data.columns.get_loc(x) for x in con_cols]
    disc_cols = [data.columns.get_loc(x) for x in disc_cols]
    target_cols = [data.columns.get_loc(x) for x in target_cols]

    if (not gpu):
        dtype = torch.FloatTensor
    else:
        dtype = torch.cuda.FloatTensor

    while True:
        # get batches
        n = np.array([pd.np.r_[i:(i + in_seq_len + out_seq_len)] for i in indexer.indices])
        batch_data = data.iloc[n.flatten()].values
        batch_data = torch.tensor(batch_data.reshape(batch_size, in_seq_len + out_seq_len, data.shape[-1]))

        # split up batch data
        in_seq_continuous = batch_data[:, 0:in_seq_len, con_cols]
        in_seq_discrete = batch_data[:, 0:in_seq_len, disc_cols]

        out_seq = batch_data[:, in_seq_len:in_seq_len + out_seq_len, disc_cols]
        target_seq = batch_data[:, in_seq_len:in_seq_len + out_seq_len, target_cols]

        yield (torch.tensor(in_seq_continuous).type(dtype).unsqueeze(-1),
               torch.tensor(in_seq_discrete).type(dtype),
               torch.tensor(out_seq).type(dtype),
               torch.tensor(target_seq).type(dtype))

        if (not given_indexer):
            indexer.next()


def one_hot(x, dims, gpu=True):
    out = []
    batch_size = x.shape[0]
    seq_len = x.shape[1]

    if (not gpu):
        dtype = torch.FloatTensor
    else:
        dtype = torch.cuda.FloatTensor

    for i in range(0, x.shape[-1]):
        x_ = x[:, :, i].byte().cpu().long().unsqueeze(-1)
        o = torch.zeros([batch_size, seq_len, dims[i]]).long()

        o.scatter_(-1, x_, 1)
        out.append(o.type(dtype))
    return out
