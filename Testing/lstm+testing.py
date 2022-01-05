import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from datetime import datetime


df = pd.read_csv(r'C:\Users\soham\PycharmProjects\NEA\Data\ftse.csv')

# Separate dates for future plotting
train_dates = pd.to_datetime(df['Date'])
print(train_dates)

# Variables for training
cols = list(df)[1:6]

df_for_training = df[cols].astype(float)

# df_for_plot=df_for_training.tail(5000)
# df_for_plot.plot.line()

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)
print(df_for_training_scaled)
# As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
# In this example, the n_features is 2. We will make timesteps = 3.
# With this, the resultant n_samples is 5 (as the input data has 9 rows).
trainX = []
trainY = []
print(trainX,trainY)
n_future = 1  # Number of days we want to predict into the future
n_past = 30  # Number of past days we want to use to predict the future

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# define Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit model
history = model.fit(trainX, trainY, epochs=10, batch_size=16)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
# Forecasting...
# Start with the last day in training date and predict future...
n_futures = 14  # Redefining n_future to extend prediction dates beyond original n_future dates...
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_futures, freq='1d').tolist()

forecast = model.predict(trainX[-n_futures:])  # forecast
print(forecast)
# Perform inverse transformation to rescale back to original range
# Since we used 5 variables for transform, the inverse expects same dimensions
# Therefore, let us copy our values 5 times and discard them after inverse transform
forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]
#print(y_pred_future)
# Convert timestamp to date
forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())
#print(forecast_dates)

df2 = np.array(forecast_dates)
print(df2)
df2 = pd.to_datetime(df2)
print(df2)
#
df3 = np.array(y_pred_future)
print(df3)
# #sns.lineplot(df2,df3)

df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Open': y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

print("THIS WORKS",df_forecast)
# original = df[['Date', 'Open']]
# original['Date'] = pd.to_datetime(original['Date'])
# original = original.loc[original['Date'] >= '2021-5-21']
#
# sns.lineplot(original['Date'], original['Open'])
from matplotlib.dates import DateFormatter
plt.figure(figsize=(15,15))
plt.plot(df_forecast['Date'], df_forecast['Open'])
plt.xticks(rotation=45)

plt.show()



"""
2021-05-20,13.090000,13.110000,12.920000,13.060000,13.060000,46474700
2021-05-21,13.110000,13.280000,13.050000,13.230000,13.230000,53320100
2021-05-24,13.290000,13.310000,13.120000,13.180000,13.180000,39012800
2021-05-25,13.200000,13.290000,13.100000,13.120000,13.120000,47329700
2021-05-26,13.120000,13.430000,13.040000,13.400000,13.400000,60269700
2021-05-27,13.600000,14.400000,13.590000,14.350000,14.350000,184588600
2021-05-28,14.290000,14.340000,14.030000,14.060000,14.060000,72571900
2021-06-01,14.230000,14.340000,14.100000,14.150000,14.150000,50276900
2021-06-02,14.180000,14.180000,14.010000,14.090000,14.090000,39936800
2021-06-03,13.990000,14.370000,13.940000,14.090000,14.090000,63163800
2021-06-04,14.160000,14.200000,13.860000,13.960000,13.960000,63924700
"""