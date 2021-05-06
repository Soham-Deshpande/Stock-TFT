import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns



#explain why i normalised the dataset

#time to normalise this dataset from 0 to 1

# class Normalise():
#     def __int__(self):
#         super().__init__()
#
#     def normalise(self,x):
#



columns = ['Date', 'Open', 'High', 'Low', 'Close']

df = pd.read_csv(r'C:\Users\soham\PycharmProjects\NEA\Data\Testing-Data.csv', header = 0)

#print(type(df))
#print(df.head())
date_raw = df['Date']
open = df['Open']
high = df['High']
low = df['Low']
close = df['Close']
volume = df['Volume']
date = [datetime.datetime.strptime(d, '%m/%d/%y') for d in date_raw]
#print(open)



print(volume)
norm = np.linalg.norm(volume)
#print(norm)
normalised = volume/norm
print(normalised)

plt.figure(figsize=(13,10), dpi= 150)
sns.distplot(normalised, color="darkblue",bins=400)
plt.show()