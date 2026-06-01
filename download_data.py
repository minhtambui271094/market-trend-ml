import requests
import pandas as pd
import time
from datetime import datetime



BASE_URL = "https://api.binance.com/api/v3/klines"


def get_klines(symbol, interval, start_time=None, end_time=None, limit=1000):
    """
    Download klines từ Binance API
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    return data


def klines_to_df(data):
    """
    Convert raw Binance data → DataFrame
    """
    cols = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(data, columns=cols)

    df["time"] = pd.to_datetime(df["time"], unit="ms")

    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df[["time", "open", "high", "low", "close", "volume"]]


def download_full(symbol, interval, start_time):
    """
    Download full history bằng loop (pagination)
    """
    all_data = []

    while True:
        print(f"Downloading {symbol} {interval} from {start_time}")

        data = get_klines(symbol, interval, start_time=start_time)

        if not data:
            break

        all_data.extend(data)

        last_time = data[-1][0]

        # +1 ms để tránh duplicate
        start_time = last_time + 1

        print(f"Got {len(data)} candles")

        time.sleep(0.3)  # tránh rate limit

        if len(data) < 1000:
            break

    return all_data


def save_data(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved: {filename} ({len(df)} rows)")


# ==========================
# CONFIG
# ==========================

symbol = "BTCUSDT"

intervals = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}

# Start time: 2022-01-01
start_time = int(datetime(2022, 1, 1).timestamp() * 1000)


# ==========================
# MAIN DOWNLOAD
# ==========================

for name, tf in intervals.items():

    print("\n============================")
    print(f"Downloading {name}")
    print("============================")

    raw = download_full(
        symbol=symbol,
        interval=tf,
        start_time=start_time
    )

    df = klines_to_df(raw)

    save_data(df, f"data/raw/BTCUSDT_{name}.csv")

print("\nDONE ALL DATA")

import os # <-- Thêm import này ở đầu file

# ... giữ nguyên các hàm trên ...

def save_data(df, filename):
    # ✅ SỬA: Tự động tạo cây thư mục nếu chưa có sẵn
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df.to_csv(filename, index=False)
    print(f"Saved: {filename} ({len(df)} rows)")