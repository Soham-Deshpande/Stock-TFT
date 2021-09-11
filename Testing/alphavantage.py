# import os
# import requests
# import pandas as pd
# from alpha_vantage.timeseries import TimeSeries
# import numpy as np
#
# #app = TimeSeries('VMB3LYVEVWH9PV8J')
#
#
# #help(app)
#
# #aapl = app.get_daily_adjusted('AAPL', outputsize='full')
# #print(aapl)
#
# import requests
# import alpha_vantage
# import json
#
#
# API_URL = "https://www.alphavantage.co/query"
# symbols = ['AAPL']
#
# for symbol in symbols:
#         data = { "function": "TIME_SERIES_INTRADAY",
#         "symbol": symbol,
#         "interval" : "5min",
#         "datatype": "json",
#         "outputsize":"full",
#         "apikey": "VMB3LYVEVWH9PV8J" }
#         response = requests.get(API_URL, data)
#         data = response.json()
#         print(symbol)
#         a = (data['Time Series (5min)'])
#         keys = (a.keys())
#         for key in keys:
#                 print(a[key]['2. high'] + " " + a[key]['3. low'] + " " + a[key]['5. volume'])


import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from datetime import datetime
import time
import matplotlib.pyplot as plt

def request_stock_price_hist(symbol, token, sample=False):
    """
    This function helps the user to retrieve historical stock prices for the
    specified symbol from Alpha Vantage.
    Parameters
    ----------
    symbol : String
        Stock symbol, e.g. Apple is AAPL.
    token : String
        Register an account on alphavantage.co and get your API.
    sample : Boolean, optional
        Set to True to take a sample of the data only.
    Returns
    -------
    df : Pandas DataFrame
        A Pandas DataFrame containing stock price information.
    """
    if sample == False:
        q_string = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize=full&apikey={}'
    else:
        q_string = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&apikey={}'

    print("Retrieving stock price data from Alpha Vantage (This may take a while)...")
    r = requests.get(q_string.format(symbol, token))
    print("Data has been successfully downloaded...")
    date = []
    colnames = list(range(0, 7))
    df = pd.DataFrame(columns=colnames)
    print("Sorting the retrieved data into a dataframe...")
    for i in tqdm(r.json()['Time Series (Daily)'].keys()):
        date.append(i)
        row = pd.DataFrame.from_dict(r.json()['Time Series (Daily)'][i], orient='index').reset_index().T[1:]
        df = pd.concat([df, row], ignore_index=True)
    df.columns = ["open", "high", "low", "close", "adjusted close", "volume", "dividend amount", "split cf"]
    df['date'] = date
    df.to_pickle("./avdataaapl.pkl")
    return df

#request_stock_price_hist("AAPL","VMB3LYVEVWH9PV8J")

df = pd.read_pickle("./avdataaapl.pkl")
#print(df.head)
#print(df.info)
df.open = int(df.open)
df.close = int(df.close)

df.plot(x = "open",
        y= "close",
        kind = "line")
plt.show()



# importing packages
import pandas as pd

# dictionary of data
dct = {"f1": range(6), "b1": range(6, 12)}

# forming dataframe
data = pd.DataFrame(dct)

# using to_pickle function to form file
# with name 'pickle_data'
pd.to_pickle(data,'./pickle_data.pkl')

# unpickled the data by using the
# pd.read_pickle method
unpickled_data = pd.read_pickle(r"C:\Users\soham\PycharmProjects\NEA\Data\NEAFTSE2010-21.csv")
print(unpickled_data)

