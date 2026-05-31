import pandas as pd
import joblib
from feature_engineering import create_features

# load latest mtf dataset
df = pd.read_csv("data/mtf_dataset.csv")

# lấy dòng cuối (latest market state)
latest = df.iloc[-1:].copy()

# load model
model = joblib.load("models/trend_model.pkl")

# tách features giống train
X = latest.drop(columns=["time", "target"], errors="ignore")

pred = model.predict(X)[0]
proba = model.predict_proba(X)[0]

label_map = {
    0: "DOWN",
    1: "SIDEWAYS",
    2: "UP"
}

print("\n===== BTC LIVE SIGNAL =====")
print("Prediction:", label_map[pred])
print("Confidence:", proba)