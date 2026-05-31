import pandas as pd
import numpy as np
import pandas_ta as ta


# ==========================
# FEATURE ENGINEERING
# ==========================

def add_features(df, tf):

    df = df.copy()

    df[f"ema20_{tf}"] = ta.ema(df["close"], length=20)
    df[f"ema78_{tf}"] = ta.ema(df["close"], length=78)
    df[f"ema200_{tf}"] = ta.ema(df["close"], length=200)

    df[f"rsi_{tf}"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"])
    df[f"macd_{tf}"] = macd["MACD_12_26_9"]
    df[f"hist_{tf}"] = macd["MACDh_12_26_9"]

    # ===== EMA GAP =====
    df[f"ema20_gap_{tf}"] = (df["close"] - df[f"ema20_{tf}"]) / df[f"ema20_{tf}"]
    df[f"ema78_gap_{tf}"] = (df["close"] - df[f"ema78_{tf}"]) / df[f"ema78_{tf}"]
    df[f"ema200_gap_{tf}"] = (df["close"] - df[f"ema200_{tf}"]) / df[f"ema200_{tf}"]

    # ===== SLOPE =====
    df[f"ema20_slope_{tf}"] = df[f"ema20_{tf}"].pct_change(3)
    df[f"ema78_slope_{tf}"] = df[f"ema78_{tf}"].pct_change(3)

    # ===== VOLATILITY =====
    df[f"volatility_{tf}"] = df["close"].pct_change().rolling(20).std()

    return df


# ==========================
# LOAD DATA
# ==========================

h1 = pd.read_csv("data/raw/BTCUSDT_1h.csv")
h4 = pd.read_csv("data/raw/BTCUSDT_4h.csv")
d1 = pd.read_csv("data/raw/BTCUSDT_1d.csv")

for df in [h1, h4, d1]:
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", inplace=True)


# ==========================
# FEATURES
# ==========================

h1 = add_features(h1, "1h")
h4 = add_features(h4, "4h")
d1 = add_features(d1, "1d")


# ==========================
# KEEP COLUMNS
# ==========================

h1 = h1.add_suffix("_1h")
h4 = h4.add_suffix("_4h")
d1 = d1.add_suffix("_1d")

h1.rename(columns={"time_1h": "time"}, inplace=True)
h4.rename(columns={"time_4h": "time"}, inplace=True)
d1.rename(columns={"time_1d": "time"}, inplace=True)


# ==========================
# MERGE MTF
# ==========================

df = pd.merge_asof(
    h4.sort_values("time"),
    h1.sort_values("time"),
    on="time",
    direction="backward"
)

df = pd.merge_asof(
    df.sort_values("time"),
    d1.sort_values("time"),
    on="time",
    direction="backward"
)


# ==========================
# LABEL (VOLATILITY BASED)
# ==========================

future = df["close_4h"].shift(-5)

ret = (future - df["close_4h"]) / df["close_4h"]

vol = ret.rolling(20).std()

df["target"] = 1  # SIDEWAYS default

df.loc[ret > vol, "target"] = 2   # UP
df.loc[ret < -vol, "target"] = 0  # DOWN


# ==========================
# CLEAN
# ==========================

print("Before dropna:", df.shape)

df = df.dropna()

print("After dropna:", df.shape)

# ==========================
# SAVE
# ==========================

df.to_csv("data/mtf_dataset.csv", index=False)

print("Saved: data/mtf_dataset.csv")
print("Columns:", len(df.columns))