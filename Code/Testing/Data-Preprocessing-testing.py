from Util import *


#explain why i normalised the dataset

#time to normalise this dataset from 0 to 1
#
# class Normalise():
#     def __int__(self):
#         super().__init__()
#
#     def normalise(self):
#         norm = np.linalg.norm(self)
#         normalised = self/ norm
#         print(normalised)
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
#print(open)



#print(open)
norm = np.linalg.norm(open)
#print(norm)
normalised = open/norm
print(normalised)

# plt.figure(figsize=(13,10), dpi= 150)
# sns.distplot(normalised, color="darkblue", bins=400)
# plt.show()




##########################################################
#notes and code from data-preprocessing.py



# print(data.datatype())
# print(data.datahead())
#rawdata = CSVHandler(r'C:\Users\soham\PycharmProjects\NEA\Data\Testing-Data.csv', 0)
#date = rawdata.extractcolumns('Date')
#print(date)
# open = rawdata.extractcolumns('Open')
# high = rawdata.extractcolumns('High')
# low = rawdata.extractcolumns('Low')
# close = rawdata.extractcolumns('Close')
# volume = rawdata.extractcolumns('Volume')
#open = data.extractcolumns('Open')


#a = splitcolumns(r'C:\Users\soham\PycharmProjects\NEA\Data\Testing-Data.csv', 0, ColumnNames)
#print(a)


# RawData = pd.read_csv(r'C:\Users\soham\PycharmProjects\NEA\Data\Testing-Data.csv', header=0)

