import numpy as np
import pandas as pd


def create_label(df, horizon=3, atr_window=14, threshold_scale=1.0):
    """
    3-class labeling based on volatility-adjusted future returns.

    0 = DOWN trend
    1 = SIDEWAY / NOISE
    2 = UP trend
    """

    df = df.copy()

    # ==========================
    # FUTURE RETURN
    # ==========================
    future_close = df["close"].shift(-horizon)

    ret = (future_close - df["close"]) / df["close"]

    # ==========================
    # VOLATILITY FILTER (ATR proxy)
    # ==========================
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift(1))
    low_close = np.abs(df["low"] - df["close"].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(atr_window).mean()

    # ==========================
    # NORMALIZE RETURN BY VOLATILITY
    # ==========================
    norm_ret = ret / (atr + 1e-9)

    # ==========================
    # THRESHOLDS (SYMMETRIC)
    # ==========================
    upper = threshold_scale
    lower = -threshold_scale

    # ==========================
    # LABELING
    # ==========================
    df["target"] = 1  # default = SIDEWAY

    df.loc[norm_ret > upper, "target"] = 2   # UP trend
    df.loc[norm_ret < lower, "target"] = 0   # DOWN trend

    return df