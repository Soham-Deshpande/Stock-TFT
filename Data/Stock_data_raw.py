# neural network will be built and run here
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.dates as mdates
from functools import reduce
import plotly.graph_objects as go


def graphing():
    columns = ['Date', 'Open', 'High', 'Low', 'Close']
    df = pd.read_csv(r'C:\Users\soham\PycharmProjects\NEA\NEAFTSE2010-21.csv', header = 0)
    #df = df.iloc[::-1]
    print(type(df))
    print(df.head())
    print(df.columns)
    df.isna().any()

    date_raw = df['Date']
    open = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']
    date = [datetime.datetime.strptime(d, '%d/%m/%Y') for d in date_raw]

    print(reduce(lambda x, y: (max(x[0], y - x[1]), y), close, (0, close[0]))[0])

    plt.figure(figsize=(50, 20))
    plt.plot(date, close)
    plt.xlabel("Years")
    plt.ylabel("Close price in £GBP")
    plt.gcf().autofmt_xdate()
    #plt.show()



    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    #fig.show()

    train_data, test_data = df[0:int(len(df) * 0.8)], df[int(len(df) * 0.8):]
    plt.figure(figsize=(12, 7))
    plt.title('FTSE 100 Prices')
    plt.xlabel('Dates')
    plt.ylabel('Prices')
    plt.plot(df['Open'], 'blue', label='Training Data')
    plt.plot(test_data['Open'], 'green', label='Testing Data')
    plt.xticks(np.arange(0, 1857, 300), df['Date'][0:1857:300])
    plt.legend()
    plt.show()
