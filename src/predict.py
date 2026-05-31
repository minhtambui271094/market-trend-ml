from data_loader import load_data
from feature_engineering import create_features

import joblib

model = joblib.load(
    "models/trend_model.pkl"
)

df = load_data("data/raw/BTCUSDT_4H.csv")

df = create_features(df)

df = df.dropna()

latest = df.iloc[-1:]

features = [
    "ema20",
    "ema78",
    "ema200",
    "rsi",
    "macd",
    "hist"
]

signal = model.predict(
    latest[features]
)

if signal[0] == 1:
    print("UP TREND")
else:
    print("DOWN TREND")

result = model.predict(X)[0]

labels = {
    0: "DOWN TREND",
    1: "SIDEWAYS",
    2: "UP TREND"
}

print(labels[result])