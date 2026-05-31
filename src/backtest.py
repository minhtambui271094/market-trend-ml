import pandas as pd
import numpy as np
import joblib


class BacktestEngine:
    def __init__(
        self,
        model_path="models/trend_model.pkl",
        fee=0.0004,
        threshold=0.6,
        price_col="close_4h"
    ):
        self.model = joblib.load(model_path)
        self.fee = fee
        self.threshold = threshold
        self.price_col = price_col

    def prepare_features(self, df):
        # loại cột không phải feature
        drop_cols = ["time", "target"]

        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # đảm bảo không lỗi NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        return X

    def generate_signal(self, proba):
        """
        proba = [p_class0, p_class1, p_class2]
        giả định:
        0 = DOWN
        1 = SIDEWAY
        2 = UP
        """
        p_down, p_side, p_up = proba

        if p_up > self.threshold and p_up > p_down:
            return 1   # BUY
        elif p_down > self.threshold and p_down > p_up:
            return -1  # SELL
        else:
            return 0   # HOLD

    def run(self, df):
        df = df.copy()

        # check price column
        if self.price_col not in df.columns:
            raise ValueError(f"Missing price column: {self.price_col}")

        X = self.prepare_features(df)

        probas = self.model.predict_proba(X)

        signals = []
        for p in probas:
            signals.append(self.generate_signal(p))

        df["signal"] = signals

        # returns
        df["ret"] = df[self.price_col].pct_change().shift(-1)

        # strategy returns
        df["strategy_ret"] = df["signal"] * df["ret"]

        # fee (only when position changes)
        df["position_change"] = df["signal"].diff().abs().fillna(0)
        df["fee_cost"] = df["position_change"] * self.fee

        df["strategy_ret_net"] = df["strategy_ret"] - df["fee_cost"]

        # equity curve
        df["equity"] = (1 + df["strategy_ret_net"].fillna(0)).cumprod()

        return df

    def metrics(self, df):
        df = df.copy()

        total_return = df["equity"].iloc[-1] - 1

        sharpe = (
            df["strategy_ret_net"].mean()
            / (df["strategy_ret_net"].std() + 1e-9)
        ) * np.sqrt(365)

        max_dd = self.max_drawdown(df["equity"])

        winrate = (df["strategy_ret_net"] > 0).mean()

        trades = (df["signal"] != 0).sum()

        return {
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "winrate": float(winrate),
            "num_trades": int(trades)
        }

    def max_drawdown(self, equity):
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return dd.min()


if __name__ == "__main__":
    df = pd.read_csv("data/mtf_dataset.csv")

    engine = BacktestEngine(
        threshold=0.6,
        fee=0.0004,
        price_col="close_4h"
    )

    result = engine.run(df)

    print("\n================ METRICS ================\n")
    for k, v in engine.metrics(result).items():
        print(f"{k}: {v}")

    result.to_csv("data/backtest_result.csv", index=False)
    print("\nSaved: data/backtest_result.csv")