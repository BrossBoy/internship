import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models


def backtest_diff(model, x_test, close_price):
    predict = model.predict(x_test)

    plt.plot(y_test)
    plt.plot(predict)
    plt.show()
