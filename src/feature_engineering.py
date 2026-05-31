import pandas_ta as ta

def create_features(df):

    df["ema20"] = ta.ema(df["close"], length=20)

    df["ema78"] = ta.ema(df["close"], length=78)

    df["ema200"] = ta.ema(df["close"], length=200)

    df["rsi"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"])

    df["macd"] = macd["MACD_12_26_9"]

    df["hist"] = macd["MACDh_12_26_9"]

    return df