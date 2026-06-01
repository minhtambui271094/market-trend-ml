import os
import joblib
import numpy as np
import pandas as pd

class TradingEngine:
    def __init__(
        self,
        model_path="models/trend_model.pkl",
        confidence_threshold=0.55
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {model_path}")
            
        self.model = joblib.load(model_path)
        self.conf_threshold = confidence_threshold
        
        # Định nghĩa chính xác 51 đặc trưng mà mô hình yêu cầu (Tránh lỗi thừa/thiếu cột khi predict)
        self.expected_features = [
            'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h', 'ema20_4h_4h', 'ema78_4h_4h', 'ema200_4h_4h', 'rsi_4h_4h', 'macd_4h_4h', 'hist_4h_4h', 'ema20_gap_4h_4h', 'ema78_gap_4h_4h', 'ema200_gap_4h_4h', 'ema20_slope_4h_4h', 'ema78_slope_4h_4h', 'volatility_4h_4h',
            'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h', 'ema20_1h_1h', 'ema78_1h_1h', 'ema200_1h_1h', 'rsi_1h_1h', 'macd_1h_1h', 'hist_1h_1h', 'ema20_gap_1h_1h', 'ema78_gap_1h_1h', 'ema200_gap_1h_1h', 'ema20_slope_1h_1h', 'ema78_slope_1h_1h', 'volatility_1h_1h',
            'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d', 'ema20_1d_1d', 'ema78_1d_1d', 'ema200_1d_1d', 'rsi_1d_1d', 'macd_1d_1d', 'hist_1d_1d', 'ema20_gap_1d_1d', 'ema78_gap_1d_1d', 'ema200_gap_1d_1d', 'ema20_slope_1d_1d', 'ema78_slope_1d_1d', 'volatility_1d_1d'
        ]

    # ======================
    # FEATURE PREP
    # ======================
    def prepare(self, df):
        """Lọc và sắp xếp chính xác 51 cột đặc trưng mô hình yêu cầu"""
        # Trích xuất đúng và đủ các cột đặc trưng để tránh lỗi lệch pha hệ thống
        X = df[self.expected_features].copy()
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
        return np.argmax(proba)  # 0=down, 1=sideways, 2=up

    # ======================
    # POSITION SIZE ENGINE
    # ======================
    def position_size(self, confidence, atr, volatility):
        """Tính toán khối lượng lệnh dựa trên độ tự tin và rủi ro thị trường"""
        # Chuẩn hóa đầu vào để tránh chia cho số quá nhỏ hoặc lỗi NaN
        atr_factor = 1 / (1 + atr if atr > 0 else 1)
        vol_factor = 1 / (1 + volatility if volatility > 0 else 1)

        size = confidence * atr_factor * vol_factor
        return np.clip(size, 0.01, 1.0) # Khối lượng tối thiểu 1%, tối đa 100% tài khoản rủi ro

    # ======================
    # MAIN PIPELINE
    # ======================
    def run(self, df):
        df = df.copy()

        # TỰ ĐỘNG SỬA: Tính toán ATR 4H trực tiếp nếu dữ liệu gốc không có sẵn cột này
        if "atr_4h" not in df.columns:
            high_low = df["high_4h"] - df["low_4h"]
            high_close = (df["high_4h"] - df["close_4h"].shift(1)).abs()
            low_close = (df["low_4h"] - df["close_4h"].shift(1)).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["atr_4h"] = true_range.rolling(14).mean().fillna(method='bfill').fillna(0)

        # Chuẩn bị ma trận đầu vào và dự báo xác suất
        X = self.prepare(df)
        probas = self.model.predict_proba(X)

        signals = []
        confidence_list = []
        size_list = []

        # Xác định chính xác tên cột Volatility sau khi đã merge đa khung thời gian
        vol_col = "volatility_4h_4h" if "volatility_4h_4h" in df.columns else "volatility_4h"

        for i, p in enumerate(probas):
            confidence = self.get_confidence(p)
            direction = self.get_direction(p)

            signal = 0
            size = 0.0

            # Bộ lọc ngưỡng độ tự tin (Confidence Threshold Filter)
            if confidence > self.conf_threshold:
                if direction == 2:    # UP TREND -> BUY
                    signal = 1
                elif direction == 0:  # DOWN TREND -> SELL
                    signal = -1
                else:                 # SIDEWAYS -> STAND OUT
                    signal = 0

                # Chỉ tính toán kích thước khối lượng lệnh nếu có tín hiệu giao dịch rõ ràng
                if signal != 0:
                    atr_val = df["atr_4h"].iloc[i]
                    vol_val = df[vol_col].iloc[i] if vol_col in df.columns else 0.0
                    size = self.position_size(confidence, atr_val, vol_val)

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
    dataset_path = "data/mtf_dataset.csv"
    if os.path.exists(dataset_path):
        df_test = pd.read_csv(dataset_path)
        engine = TradingEngine(confidence_threshold=0.55)
        result = engine.run(df_test)

        print("\n===== 10 TÍN HIỆU MỚI NHẤT TRÊN ĐỒ THỊ =====")
        print(result[["time", "close_4h", "signal", "confidence", "position_size"]].tail(10))
    else:
        print(f"❌ Không tìm thấy file {dataset_path} để chạy thử nghiệm Engine.")