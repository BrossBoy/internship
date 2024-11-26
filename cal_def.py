import numpy as np


def cal_cagr(year, history):
    day = 365 * year
    if day < len(history):
        cagr = (history[-1] / history[-day]) ** (1 / year) - 1
        cagr *= 100
        return cagr
    else:
        print("cant find cagr")
        return 0


def cal_max_drawdown(history):
    max_drawdown = 0
    peak = 0
    index_peak = 0
    index_trough = 0
    index_use_peak = 0
    for i in range(len(history)):
        if history[i] > peak:
            peak = history[i]
            index_peak = i
        else:
            drawdown = (peak - history[i]) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                index_use_peak = index_peak
                index_trough = i
    return max_drawdown, index_use_peak, index_trough


def cal_sd(history):
    history = np.array(history)
    return np.std(history)
