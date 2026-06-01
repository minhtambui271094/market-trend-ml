import os
import time
import ccxt
import joblib
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta

class LivePredictor:
    def __init__(self):
        # 1. Kiểm tra sự tồn tại của file mô hình
        if not os.path.exists("models/trend_model.pkl"):
            raise FileNotFoundError("Chưa tìm thấy file models/trend_model.pkl. Vui lòng chạy train.py trước.")
        self.model = joblib.load("models/trend_model.pkl")
        
        # Danh sách 51 đặc trưng (features) đa khung thời gian mà mô hình XGBoost yêu cầu
        self.expected_features = [
            'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h', 'ema20_4h_4h', 'ema78_4h_4h', 'ema200_4h_4h', 'rsi_4h_4h', 'macd_4h_4h', 'hist_4h_4h', 'ema20_gap_4h_4h', 'ema78_gap_4h_4h', 'ema200_gap_4h_4h', 'ema20_slope_4h_4h', 'ema78_slope_4h_4h', 'volatility_4h_4h',
            'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h', 'ema20_1h_1h', 'ema78_1h_1h', 'ema200_1h_1h', 'rsi_1h_1h', 'macd_1h_1h', 'hist_1h_1h', 'ema20_gap_1h_1h', 'ema78_gap_1h_1h', 'ema200_gap_1h_1h', 'ema20_slope_1h_1h', 'ema78_slope_1h_1h', 'volatility_1h_1h',
            'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d', 'ema20_1d_1d', 'ema78_1d_1d', 'ema200_1d_1d', 'rsi_1d_1d', 'macd_1d_1d', 'hist_1d_1d', 'ema20_gap_1d_1d', 'ema78_gap_1d_1d', 'ema200_gap_1d_1d', 'ema20_slope_1d_1d', 'ema78_slope_1d_1d', 'volatility_1d_1d'
        ]
        
        # 2. Khởi tạo kết nối sàn OKX công khai (Read-only để lấy data công khai)
        self.exchange = ccxt.okx({'enableRateLimit': True, 'timeout': 30000})
        self.symbol = 'BTC/USDT'  

    def compute_indicators_for_df(self, df):
        """Tính toán các chỉ báo kỹ thuật ĐỒNG BỘ 100% với tập dữ liệu train (mtf_dataset.py)"""
        df = df.copy()
        df['ema20'] = ta.ema(df['close'], length=20)
        df['ema78'] = ta.ema(df['close'], length=78)
        df['ema200'] = ta.ema(df['close'], length=200)
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd_df is not None:
            df['macd'] = macd_df.iloc[:, 0]
            df['hist'] = macd_df.iloc[:, 2]
        else:
            df['macd'], df['hist'] = 0.0, 0.0
            
        # Tỷ lệ khoảng cách GAP (%)
        df['ema20_gap'] = (df['close'] - df['ema20']) / df['ema20']
        df['ema78_gap'] = (df['close'] - df['ema78']) / df['ema78']
        df['ema200_gap'] = (df['close'] - df['ema200']) / df['ema200']
        
        # Độ dốc xu hướng SLOPE (% thay đổi 3 phiên)
        df['ema20_slope'] = df['ema20'].pct_change(3)
        df['ema78_slope'] = df['ema78'].pct_change(3)
        
        # Độ biến động VOLATILITY (Std của tỷ lệ thay đổi giá trong 20 phiên)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        return df

    def fetch_data_for_tf(self, tf):
        """Tải cuốn chiếu đủ lượng nến cần thiết từ sàn OKX qua API một cách an toàn"""
        try:
            all_ohlcv = []
            limit_per_request = 100  
            since = None
            max_candles = 500  # Tối ưu hóa từ 5000 xuống 500 để tránh bị sàn khóa IP do spam API
            
            while len(all_ohlcv) < max_candles:
                if since:
                    ohlcv = self.exchange.fetch_ohlcv(self.symbol, tf, since=since, limit=limit_per_request)
                else:
                    ohlcv = self.exchange.fetch_ohlcv(self.symbol, tf, limit=limit_per_request)
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                all_ohlcv = sorted(all_ohlcv, key=lambda x: x[0])
                
                # Tính toán mốc thời gian lùi về quá khứ
                timeframe_ms = self.exchange.parse_timeframe(tf) * 1000
                since = all_ohlcv[0][0] - (limit_per_request * timeframe_ms)
                
                time.sleep(0.05) # Giảm tải cho API
                if len(ohlcv) < limit_per_request:
                    break
            
            all_ohlcv = sorted(all_ohlcv, key=lambda x: x[0])
            all_ohlcv = all_ohlcv[-max_candles:]
            
            df = pd.DataFrame(all_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
        except Exception as e:
            logging.error(f"Lỗi thu thập dữ liệu khung {tf}: {e}")
            return None

    def get_signal(self):
        """Xử lý đặc trưng đa khung thời gian (MTF) và trả về dự báo kèm độ tự tin cao nhất"""
        try:
            # 1. Thu thập dữ liệu từ sàn OKX
            df_4h = self.fetch_data_for_tf('4h')
            df_1h = self.fetch_data_for_tf('1h')
            df_1d = self.fetch_data_for_tf('1d') 
            
            if df_4h is None or df_1h is None or df_1d is None:
                logging.warning("Không thể lấy đủ dữ liệu từ OKX. Đang trả về tín hiệu mặc định (Sideways).")
                return 1, 1.0

            # 2. Tính toán các chỉ báo kỹ thuật đồng bộ
            feat_4h = self.compute_indicators_for_df(df_4h)
            feat_1h = self.compute_indicators_for_df(df_1h)
            feat_1d = self.compute_indicators_for_df(df_1d)
            
            # 3. Trích xuất cây nến đóng cửa gần nhất của từng khung thời gian
            last_4h = feat_4h.iloc[-1].to_dict()
            last_1h = feat_1h.iloc[-1].to_dict()
            last_1d = feat_1d.iloc[-1].to_dict()

            # 4. Ánh xạ các đặc trưng khớp với cấu trúc cột đa khung lúc huấn luyện mô hình
            input_dict = {}
            for col in ['open', 'high', 'low', 'close', 'volume', 'ema20', 'ema78', 'ema200', 'rsi', 'macd', 'hist', 'ema20_gap', 'ema78_gap', 'ema200_gap', 'ema20_slope', 'ema78_slope', 'volatility']:
                input_dict[f'{col}_4h' if col in ['open','high','low','close','volume'] else f'{col}_4h_4h'] = last_4h.get(col, 0.0)
                input_dict[f'{col}_1h' if col in ['open','high','low','close','volume'] else f'{col}_1h_1h'] = last_1h.get(col, 0.0)
                input_dict[f'{col}_1d' if col in ['open','high','low','close','volume'] else f'{col}_1d_1d'] = last_1d.get(col, 0.0)

            # Chuyển đổi sang định dạng dữ liệu đầu vào chuẩn của mô hình
            X_input = pd.DataFrame([input_dict]).fillna(0.0)
            X_input = X_input[self.expected_features]

            # 5. Thực hiện dự báo mô hình
            pred = self.model.predict(X_input)[0]
            all_probas = self.model.predict_proba(X_input)[0]
            
            # Lấy xác suất của chính nhãn dự đoán để làm độ tự tin (Confidence Score) cho main.py sử dụng
            proba = float(all_probas[pred])
            
            return int(pred), proba
            
        except Exception as e:
            logging.error(f"Lỗi trong quá trình phân tích đặc trưng MTF Live: {e}")
            return 1, 0.0