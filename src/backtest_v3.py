import pandas as pd
import numpy as np
import joblib


class BacktestV3:
    def __init__(
        self,
        model_path="models/trend_model.pkl",
        fee=0.0004,
        threshold=0.6,
        price_col="close_4h",
        atr_col="atr_4h",
        vol_col="volatility_4h"
    ):
        self.model = joblib.load(model_path)
        self.fee = fee
        self.threshold = threshold
        self.price_col = price_col
        self.atr_col = atr_col
        self.vol_col = vol_col

    # ======================
    # FEATURES
    # ======================
    def prepare_features(self, df):
        drop_cols = ["time", "target"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ======================
    # SIGNAL
    # ======================
    def generate_signal(self, proba):
        p_down, p_side, p_up = proba

        if p_up > self.threshold and p_up > p_down:
            return 1
        elif p_down > self.threshold and p_down > p_up:
            return -1
        return 0

    # ======================
    # POSITION SIZING
    # ======================
    def position_size(self, confidence, atr):
        """
        risk-based sizing:
        - higher confidence → bigger size
        - higher volatility (ATR) → smaller size
        """
        base = confidence

        vol_penalty = 1 / (1 + atr)

        size = base * vol_penalty

        return np.clip(size, 0, 1)

    # ======================
    # REGIME FILTER
    # ======================
    def regime_filter(self, df):
        """
        only trade when volatility is not too low
        """
        if self.vol_col not in df.columns:
            return np.ones(len(df), dtype=bool)

        vol = df[self.vol_col].fillna(0)

        # dynamic threshold (median)
        threshold = vol.rolling(50).median().fillna(vol.median())

        return vol > threshold

    # ======================
    # ATR STOP LOSS
    # ======================
    def apply_stoploss(self, df):
        stop = []

        position = 0
        entry_price = 0
        atr = df[self.atr_col].fillna(0)

        for i in range(len(df)):
            price = df[self.price_col].iloc[i]
            signal = df["signal"].iloc[i]

            # new entry
            if signal != 0 and position == 0:
                position = signal
                entry_price = price
                stop.append(0)
                continue

            # exit logic
            if position != 0:
                stop_dist = 1.5 * atr.iloc[i]

                if position == 1:
                    if price < entry_price - stop_dist:
                        position = 0
                        stop.append(-1)
                        continue

                if position == -1:
                    if price > entry_price + stop_dist:
                        position = 0
                        stop.append(-1)
                        continue

            stop.append(position)

        df["position"] = stop
        return df

    # ======================
    # BACKTEST RUN
    # ======================
    def run(self, df):
        df = df.copy()

        if self.price_col not in df.columns:
            raise ValueError(f"Missing {self.price_col}")

        X = self.prepare_features(df)
        probas = self.model.predict_proba(X)

        # signal
        signals = []
        for p in probas:
            signals.append(self.generate_signal(p))

        df["signal"] = signals

        # regime filter
        regime_ok = self.regime_filter(df)
        df["signal"] = df["signal"] * regime_ok.astype(int)

        # ATR required fallback
        if self.atr_col not in df.columns:
            df[self.atr_col] = df[self.price_col].rolling(14).std()

        # apply stoploss engine
        df = self.apply_stoploss(df)

        # returns
        df["ret"] = df[self.price_col].pct_change().shift(-1)

        # confidence approx
        df["confidence"] = np.max(probas, axis=1)

        # position sizing
        df["size"] = df.apply(
            lambda r: self.position_size(r["confidence"], r[self.atr_col]),
            axis=1
        )

        # strategy return
        df["strategy_ret"] = df["position"] * df["size"] * df["ret"]

        # fee
        df["trade"] = df["position"].diff().abs().fillna(0)
        df["fee_cost"] = df["trade"] * self.fee

        df["strategy_ret_net"] = df["strategy_ret"] - df["fee_cost"]

        # equity
        df["equity"] = (1 + df["strategy_ret_net"].fillna(0)).cumprod()

        return df

    # ======================
    # METRICS
    # ======================
    def metrics(self, df):
        total_return = df["equity"].iloc[-1] - 1

        sharpe = (
            df["strategy_ret_net"].mean()
            / (df["strategy_ret_net"].std() + 1e-9)
        ) * np.sqrt(365)

        max_dd = (df["equity"] / df["equity"].cummax() - 1).min()

        winrate = (df["strategy_ret_net"] > 0).mean()

        trades = df["trade"].sum()

        return {
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "winrate": float(winrate),
            "num_trades": float(trades)
        }


if __name__ == "__main__":
    df = pd.read_csv("data/mtf_dataset.csv")

    engine = BacktestV3(
        threshold=0.6,
        fee=0.0004
    )

    result = engine.run(df)

    print("\n========= METRICS V3 =========\n")
    for k, v in engine.metrics(result).items():
        print(f"{k}: {v}")

    result.to_csv("data/backtest_v3.csv", index=False)
    print("\nSaved: data/backtest_v3.csv")