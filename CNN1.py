# -*- coding: utf-8 -*-
"""CNN1.ipynb

"""

# Library list
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, Flatten, Input, Activation
from keras import optimizers
import numpy as np
import pandas as pd
import tensorflow
import yfinance as yf
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Set random parameter
np.random.seed(4)
tensorflow.random.set_seed(4)

# Get the data for the stock MSFT for the last 10 years
df = yf.download('MSFT', '2014-06-01', '2024-06-01')
next_day_closed = df['Close'].shift(-1, axis=0)
next_day_closed.drop(index=next_day_closed.index[len(next_day_closed) - 1], axis=0, inplace=True)
df.drop(index=df.index[len(df) - 1], axis=0, inplace=True)
date = df.index
df.reset_index(drop=True, inplace=True)
df.tail()
open_values = df['Open'].to_numpy()
closed_values = df['Close'].to_numpy()
date_values = date.to_numpy()

# Convert as an array to be used as feature in classifier
data = df.to_numpy()
# Convert next closed data to array, will be used as label
next_day_closed_values = next_day_closed.to_numpy()
# Dataset split and number of rows (history) involved in prediction
test_split = 0.9
history_points = 500

# Normalized data
data_normaliser = preprocessing.MinMaxScaler()
data_normalised = data_normaliser.fit_transform(data)
next_day_closed_values = next_day_closed_values.reshape(-1, 1)
next_day_closed_values_normalised = data_normaliser.fit_transform(next_day_closed_values)

data_histories_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
n = int(data_histories_normalised.shape[0] * test_split)
next_day_closed_values_normalised = np.array([next_day_closed_values_normalised[i + history_points].copy() for i in range(len(next_day_closed_values_normalised) - history_points)])
next_day_closed_values = np.array([next_day_closed_values[i + history_points].copy() for i in range(len(next_day_closed_values) - history_points)])
open_values = np.array([open_values[i + history_points].copy() for i in range(len(open_values) - history_points)])
closed_values = np.array([closed_values[i + history_points].copy() for i in range(len(closed_values) - history_points)])
date_values = np.array([date_values[i + history_points].copy() for i in range(len(date_values) - history_points)])

# Train data
data_train = data_histories_normalised[:n]
y_train = next_day_closed_values_normalised[:n]

# Test data
data_test = data_histories_normalised[n:]
y_test = next_day_closed_values_normalised[n:]

unscaled_y_test = next_day_closed_values[n:]
unscaled_y = next_day_closed_values
stacking = pd.DataFrame(columns=['date', 'open', 'close', 'delta_next_day', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4', 'prediction_5'])
stacking['date'] = date_values
stacking['open'] = open_values
stacking['close'] = closed_values
stacking['nextclose'] = next_day_closed
stacking['delta_next_day'] = stacking['nextclose'] - stacking['close']
stacking.drop(['nextclose'], axis=1, inplace=True)

lr = [0.001, 0.005, 0.0005, 0.0001, 0.00001]

# CNN MODEL
for ii in range(5):
    # Model architecture
    cnn_input = Input(shape=(history_points, 6), name='cnn_input')
    x = Conv1D(filters=64, kernel_size=2, activation='relu', name='conv1d_0')(cnn_input)
    x = Dropout(0.2, name='cnn_dropout_0')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)

    model = Model(inputs=cnn_input, outputs=output)
    adam = optimizers.Adam(learning_rate=lr[ii])
    model.compile(optimizer=adam, loss='mse')

    model.fit(x=data_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)

    # evaluation
    y_test_predicted = model.predict(data_test)
    y_test_predicted = data_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict(data_histories_normalised)
    y_predicted = data_normaliser.inverse_transform(y_predicted)
    unscaled_y_test = np.reshape(unscaled_y_test, (-1, 1))
    print(unscaled_y_test.shape)
    print(y_test_predicted.shape)

    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)

    plt.gcf().set_size_inches(22, 15, forward=True)
    start = 0
    end = -1
    # real = plt.plot(unscaled_y_test[start:end], label='real')
    # pred = plt.plot(y_test_predicted[start:end], label='predicted')
    # real = plt.plot(unscaled_y[start:end], label='real')
    # pred = plt.plot(y_predicted[start:end], label='predicted')
    # plt.legend(['Real', 'Predicted'])
    # plt.show()
    # from datetime import datetime
    # model.save(f'basic_model.h5')

    # If next_day_closed_price is higher than today closed price for given threshold, then long or buy
    # If the difference is negative, short or sell, otherwise hold.
    thresh = 0.1
    hold_ = 0
    long_ = 1
    short_ = 2
    strategy = []
    closed_price_today = []
    for i in range(len(unscaled_y)):
        closed_price_today_ = unscaled_y[i]
        closed_price_today.append(closed_price_today_)
        predicted_closed_price_tomorrow = y_predicted[i]
        delta = predicted_closed_price_tomorrow - closed_price_today_
        if delta > thresh:
            strategy.append(long_)
        elif delta < 0:
            strategy.append(short_)
        else:
            strategy.append(hold_)
    if (ii == 0):
        stacking['prediction_1'] = strategy
    if (ii == 1):
        stacking['prediction_2'] = strategy
    if (ii == 2):
        stacking['prediction_3'] = strategy
    if (ii == 3):
        stacking['prediction_4'] = strategy
    if (ii == 4):
        stacking['prediction_5'] = strategy

    stacking.to_csv('LSTM_stacking_output.csv')

    # today = plt.plot(closed_price_today, label='Today')
    # tomorrow = plt.plot(y_predicted, label='Tomorrow')
    # stra = plt.plot(strategy, label='Startegy')
    #
    # plt.legend(['Today', 'Tomorrow','Strategy: hold_0/long_1/short_2'])
    #
    # plt.show()


