import numpy as np
import joblib
import pandas as pd


class TradingEngine:
    def __init__(
        self,
        model_path="models/trend_model.pkl",
        confidence_threshold=0.55
    ):
        self.model = joblib.load(model_path)
        self.conf_threshold = confidence_threshold

    # ======================
    # FEATURE PREP
    # ======================
    def prepare(self, df):
        drop_cols = ["time", "target"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ======================
    # CONFIDENCE CALC
    # ======================
    def get_confidence(self, proba):
        return np.max(proba)

    # ======================
    # DIRECTION MAP
    # ======================
    def get_direction(self, proba):
        idx = np.argmax(proba)
        return idx  # 0=down,1=side,2=up

    # ======================
    # POSITION SIZE ENGINE
    # ======================
    def position_size(self, confidence, atr, volatility):

        # normalize inputs
        atr_factor = 1 / (1 + atr)
        vol_factor = 1 / (1 + volatility)

        size = confidence * atr_factor * vol_factor

        return np.clip(size, 0, 1)

    # ======================
    # MAIN PIPELINE
    # ======================
    def run(self, df):
        df = df.copy()

        X = self.prepare(df)
        probas = self.model.predict_proba(X)

        signals = []
        confidence_list = []
        size_list = []

        for i, p in enumerate(probas):

            confidence = self.get_confidence(p)
            direction = self.get_direction(p)

            # default no trade
            signal = 0
            size = 0

            # threshold filter
            if confidence > self.conf_threshold:

                if direction == 2:
                    signal = 1
                elif direction == 0:
                    signal = -1
                else:
                    signal = 0

                # position sizing inputs
                atr = df.get("atr_4h", pd.Series(np.zeros(len(df)))).iloc[i]
                vol = df.get("volatility_4h", pd.Series(np.zeros(len(df)))).iloc[i]

                size = self.position_size(confidence, atr, vol)

            signals.append(signal)
            confidence_list.append(confidence)
            size_list.append(size)

        df["signal"] = signals
        df["confidence"] = confidence_list
        df["position_size"] = size_list

        return df


# ======================
# TEST RUN
# ======================
if __name__ == "__main__":
    df = pd.read_csv("data/mtf_dataset.csv")

    engine = TradingEngine()

    result = engine.run(df)

    print("\n===== LATEST SIGNAL =====")
    print(result[["signal", "confidence", "position_size"]].tail(10))