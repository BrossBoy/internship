import yfinance as yf
import numpy as np
import pandas as pd
import talib as ta

from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler(feature_range=(0, 1))


def sklearn_scaled(data):
    data = scale.fit_transform(data)
    return data


def split_data(data, close, data_range=30, train_ratio=0.8):
    split_range = int(len(data) * train_ratio)
    _, col = data.shape
    X = data[:, 12 : col - 1]
    Y = data[:, -1]

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    test_close = []

    for i in range(split_range - 30):
        x_train.append(X[i : i + data_range])
        y_train.append(Y[i + data_range - 1])
        x_test.append(X[split_range + i : split_range + i + data_range])
        y_test.append(Y[split_range + i + data_range - 1])
        test_close.append(close[split_range + i + data_range - 1])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test, test_close


def prepar_data(asset, start, end, target_range):
    data = yf.download(asset, start=start, end=end)
    data.dropna(inplace=True)

    # momentum indicator
    data["RSI"] = ta.RSI(data["Close"][asset[0]], timeperiod=14)
    data["MACD"], data["MACD_Signal"], _ = ta.MACD(
        data["Close"][asset[0]], fastperiod=12, slowperiod=26, signalperiod=9
    )
    data["Momentum"] = ta.MOM(data["Close"][asset[0]], timeperiod=10)
    data["Stochastic"] = ta.STOCH(
        data["High"][asset[0]],
        data["Low"][asset[0]],
        data["Close"][asset[0]],
        fastk_period=14,
        slowk_period=3,
    )[0]

    # volatility indicator
    data["Volatility"] = ta.ATR(
        data["High"][asset[0]],
        data["Low"][asset[0]],
        data["Close"][asset[0]],
        timeperiod=14,
    )
    data["NATR"] = ta.NATR(
        data["High"][asset[0]],
        data["Low"][asset[0]],
        data["Close"][asset[0]],
        timeperiod=14,
    )

    # overlap studies functions
    data["SMA_15"] = ta.SMA(data["Close"][asset[0]], timeperiod=15)
    data["SMA_30"] = ta.SMA(data["Close"][asset[0]], timeperiod=30)
    data["SMA_60"] = ta.SMA(data["Close"][asset[0]], timeperiod=60)
    data["BB_upper"], data["BB_middle"], data["BB_lower"] = ta.BBANDS(
        data["Close"][asset[0]], timeperiod=20
    )

    # volume indicator
    data["AD"] = ta.AD(
        data["High"][asset[1]],
        data["Low"][asset[1]],
        data["Close"][asset[1]],
        data["Volume"][asset[1]],
    )
    data["ADOSC"] = ta.ADOSC(
        data["High"][asset[1]],
        data["Low"][asset[1]],
        data["Close"][asset[1]],
        data["Volume"][asset[1]],
        fastperiod=3,
        slowperiod=10,
    )
    data["OBV"] = ta.OBV(data["Close"][asset[1]], data["Volume"][asset[1]])

    # target of predict
    data["Target"] = (
        data["Close"][asset[0]].shift(-target_range) - data["Close"][asset[0]]
    )

    data.dropna(inplace=True)
    return data
