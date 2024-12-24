import process_data
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, models
from keras.models import Sequential
from keras.layers import Dense, LSTM


def model_lstm(range_input=30, neurons=16, n_feature=3):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(range_input, n_feature)))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mean_squared_error", optimizer=optimizers.Adam())
    return model


def train_model(b, e, n, x_train, y_train):
    model = model_lstm(
        range_input=x_train.shape[1], n_feature=x_train.shape[2], neurons=n
    )
    history = model.fit(x_train, y_train, batch_size=b, epochs=e)
    return model, history


if __name__ == "__main__":
    asset = ["GC=F", "^GSPC"]
    start = "2014-01-01"
    end = "2024-12-01"
    target_range = 2
    range_input = 30

    data = process_data.prepar_data(asset, start, end, target_range)
    data_scale = process_data.sklearn_scaled(data)
    x_train, y_train, x_test, y_test, test_close = process_data.split_data(
        data=data_scale,
        train_ratio=0.5,
        close=data["Close"]["GC=F"].to_numpy(),
        data_range=range_input,
    )
    print(x_train.shape, y_train.shape)
    model = model_lstm(
        range_input=x_train.shape[1], n_feature=x_train.shape[2], neurons=36
    )
    history = model.fit(x_train, y_train, batch_size=500, epochs=200)
