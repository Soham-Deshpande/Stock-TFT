# ---------------------------------------------------#
#
#   File       : Data-Preprocessing.py
#   Author     : Soham Deshpande
#   Date       : May 2021
#   Description: Handle CSV files
#                Normalise the dataset so that all
#                values lie in between 0 and 1
#

# ----------------------------------------------------#

import pandas as pd
import numpy as np


class CSVHandler:
    def __init__(self, filename, header):
        self.data = pd.read_csv(filename, header=header)

    # @property
    def read(self):
        return self.data

    @property
    def datatype(self):
        return type(self.data)

    @property
    def datahead(self):
        return self.data.head()

    def extractcolumns(self, column):
        return self.data[column].tolist()


def splitcolumns(filename, headers, columns):
    rawdata = CSVHandler(filename, headers)
    datalist = []
    print(len(columns))
    for i in columns:
        columni = rawdata.extractcolumns(str(i))
        datalist.append(columni)
    return datalist


class Normalise:
    def __int__(self):
        super().__init__()

    def normalise(self, rdatasplit):
        self.data = rdatasplit
        self.norm = np.linalg.norm(rdatasplit)
        self.normalised = rdatasplit / self.norm
        return self.normalised.tolist()


def normalisedata(columnnames):
    rdata = splitcolumns(r'C:\Users\soham\PycharmProjects\NEA\Data\Testing-Data.csv', 0, columnnames)
    normal = Normalise()  # rawdata
    ndata = normal.normalise(rdata)  # normalised data
    return ndata


ColumnNames = ['Open', 'Open', 'High', 'Low', 'Close']
print(normalisedata(ColumnNames))
