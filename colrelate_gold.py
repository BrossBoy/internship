import yfinance as yf
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


def test():
    gold = yf.download(
        [
            "GC=F",
            "^GSPC",
            "RTX",
            "NOC",
            "BA",
            "GD",
            "XOM",
            "CVX",
            "COP",
            "NEM",
            "FNV",
            "^NDX",
            "QQQ",
            # "QQQM",
            "ITA",
            "XAR",
            "PPA",
            "^VIX",
        ],
        "2014-01-01",
        "2024-11-25",
    )
    tmp = gold["Close"]
    tmp = tmp.dropna()
    # tmp = tmp["Close"]
    # gold_p = tmp["GC=F"]
    # sp_p = tmp["^GSPC"]
    # print(np.corrcoef(gold_p, sp_p))
    # print(tmp.keys())
    # for i in tmp.keys():
    #     print(tmp[i].isna().sum(), i)
    # cor = tmp[tmp.keys()[1:]].corr()
    sb.heatmap(tmp.corr(), annot=True)
    plt.show()


test()

# gold = yf.download("GC=F", "2014-01-01", "2024-11-25")
# gold_price = gold["Close"]

# sp500 = yf.download("^GSPC", "2014-01-01", "2024-11-25")
# sp500_price = sp500["Close"]


# # print(gold_price)
# # print(sp500_price)
# print(np.corrcoef(gold_price["GC=F"], sp500_price["^GSPC"]))
