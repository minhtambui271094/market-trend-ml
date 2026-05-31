import requests
import pandas as pd
import os

SYMBOL = "BTCUSDT"

INTERVALS = [
    "15m",
    "1h",
    "4h",
    "1d"
]

LIMIT = 1000

os.makedirs("data/raw", exist_ok=True)

for interval in INTERVALS:

    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol={SYMBOL}"
        f"&interval={interval}"
        f"&limit={LIMIT}"
    )

    print(f"Downloading {interval}...")

    data = requests.get(url, timeout=30).json()

    df = pd.DataFrame(data, columns=[
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore"
    ])

    df = df[
        ["open_time", "open", "high", "low", "close", "volume"]
    ]

    df["time"] = pd.to_datetime(
        df["open_time"],
        unit="ms"
    )

    df = df[
        ["time", "open", "high", "low", "close", "volume"]
    ]

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[c] = pd.to_numeric(df[c])

    filename = f"data/raw/{SYMBOL}_{interval}.csv"

    df.to_csv(
        filename,
        index=False
    )

    print(f"Saved: {filename}")

print("Done")