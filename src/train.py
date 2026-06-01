# =====================================================================
# HỆ THỐNG TRADING BOT THÔNG MINH (BẢN CHẠY LOCAL TRÊN VS CODE)
# =====================================================================
import os
import sys
import time
import logging
import urllib.request
import numpy as np
import pandas as pd
import ccxt
import pandas_ta as ta
import joblib
from xgboost import XGBClassifier

# Cấu hình log hiển thị nhật ký
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Cấu hình thông số kết nối sàn OKX
exchange = ccxt.okx({'enableRateLimit': True, 'timeout': 30000})
symbol = 'BTC/USDT'
model_path = "trend_model.pkl"

# Định nghĩa các hàm cào dữ liệu và tính toán chỉ báo kỹ thuật giống hệt lúc train
def fetch_historical_candles(tf, max_candles=5000):
    all_ohlcv = []
    limit_per_request = 100
    since = None
    print(f"   -> Đang cào cuốn chiếu dữ liệu quá khứ khung {tf}...")
    while len(all_ohlcv) < max_candles:
        try:
            if since:
                ohlcv = exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit_per_request)
            else:
                ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit_per_request)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            all_ohlcv = sorted(all_ohlcv, key=lambda x: x[0])
            timeframe_ms = exchange.parse_timeframe(tf) * 1000
            since = all_ohlcv[0][0] - (limit_per_request * timeframe_ms)
            time.sleep(0.05)
            if len(ohlcv) < limit_per_request:
                break
        except Exception as e:
            print(f"   [Lưu ý] Dừng cào khung {tf} tại mốc này: {e}")
            break
    all_ohlcv = sorted(all_ohlcv, key=lambda x: x[0])[-max_candles:]
    df = pd.DataFrame(all_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

def compute_indicators(df):
    if df is None or len(df) < 200:
        return df
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

    df['ema20_gap'] = df['close'] - df['ema20']
    df['ema78_gap'] = df['close'] - df['ema78']
    df['ema200_gap'] = df['close'] - df['ema200']
    df['ema20_slope'] = df['ema20'].diff(1)
    df['ema78_slope'] = df['ema78'].diff(1)
    df['volatility'] = df['close'].rolling(window=14).std()
    return df

# ---------------------------------------------------------------------
# GIAI ĐOẠN 1: QUẢN LÝ BỘ NÃO MÔ HÌNH CHẠY (GITHUB HOẶC AUTO-TRAIN)
# ---------------------------------------------------------------------
model = None

if not os.path.exists(model_path):
    # Thử tải từ GitHub của bạn
    github_raw_url = "https://raw.githubusercontent.com/minhtambui271094/market-trend-ml/main/models/trend_model.pkl"
    print(f"\n🔄 [Bước 1] Đang thử kết nối tải mô hình từ GitHub...")
    try:
        urllib.request.urlretrieve(github_raw_url, model_path)
        print("✅ Tải thành công! Đang nạp mô hình từ GitHub vào hệ thống.")
        model = joblib.load(model_path)
    except Exception as e:
        print(f"⚠️ Không thể tải từ GitHub (Chi tiết: {e}).")
        print("🤖 [CƠ CHẾ DỰ PHÒNG CHÍNH XÁC]: Tự động chuyển sang chế độ tự huấn luyện cục bộ ngay lập tức!")

        # Tiến hành cào dữ liệu tự huấn luyện
        df_4h = compute_indicators(fetch_historical_candles('4h', 5000))
        df_1h = compute_indicators(fetch_historical_candles('1h', 5000))
        df_1d = compute_indicators(fetch_historical_candles('1d', 5000))

        print("   -> Đang đồng bộ hóa đa khung thời gian...")
        df_4h = df_4h.add_suffix('_4h').rename(columns={'time_4h': 'time'})
        df_1h = df_1h.add_suffix('_1h').rename(columns={'time_1h': 'time'})
        df_1d = df_1d.add_suffix('_1d').rename(columns={'time_1d': 'time'})

        final_df = pd.merge_asof(df_4h.sort_values('time'), df_1h.sort_values('time'), on='time', direction='backward')
        final_df = pd.merge_asof(final_df, df_1d.sort_values('time'), on='time', direction='backward')

        # Tạo nhãn Target học tập
        future_return = final_df['close_4h'].shift(-1) / final_df['close_4h'] - 1
        conditions = [ (future_return > 0.005), (future_return < -0.005) ]
        choices = [2, 0] # 2: Tăng, 0: Giảm, 1: Đi ngang
        final_df['target'] = np.select(conditions, choices, default=1)
        final_df = final_df.dropna()

        features = [c for c in final_df.columns if c not in ["time", "target"]]
        X, y = final_df[features], final_df["target"]

        print("   -> Đang huấn luyện kiến trúc thuật toán XGBoost chuyên sâu...")
        model = XGBClassifier(
            n_estimators=500, 
            learning_rate=0.03, 
            max_depth=6, 
            objective='multi:softprob', 
            num_class=3, 
            random_state=42
        )
        model.fit(X, y)
        joblib.dump(model, model_path)
        print("✅ Đã hoàn tất huấn luyện mô hình dự phòng tại chỗ!")

else:
    # Nếu file đã tồn tại sẵn ở môi trường hiện tại
    print("✅ Đang nạp mô hình có sẵn từ ổ cứng máy tính...")
    model = joblib.load(model_path)

# Trích xuất chính xác danh sách các thuộc tính mà mô hình cần dự đoán
expected_features = model.feature_names_in_

# ---------------------------------------------------------------------
# GIAI ĐOẠN 2: KHỞI CHẠY VÒNG LẶP LIVE BOT QUÉT TÍN HIỆU THỊ TRƯỜNG
# ---------------------------------------------------------------------
print("\n🚀 --- HỆ THỐNG TRADING BOT ĐÃ KHỞI ĐỘNG THÀNH CÔNG! ĐANG LIVE QUÉT... ---")

while True:
    try:
        # Khi chạy quét Live thực tế, chỉ cần tải 500 nến gần nhất để tính toán chỉ báo cho nhanh
        live_4h = compute_indicators(fetch_historical_candles('4h', 500))
        live_1h = compute_indicators(fetch_historical_candles('1h', 500))
        live_1d = compute_indicators(fetch_historical_candles('1d', 500))

        last_4h = live_4h.iloc[-1].to_dict()
        last_1h = live_1h.iloc[-1].to_dict()
        last_1d = live_1d.iloc[-1].to_dict()

        input_dict = {}
        for col in ['open', 'high', 'low', 'close', 'volume', 'ema20', 'ema78', 'ema200', 'rsi', 'macd', 'hist', 'ema20_gap', 'ema78_gap', 'ema200_gap', 'ema20_slope', 'ema78_slope', 'volatility']:
            input_dict[f'{col}_4h'] = last_4h.get(col, 0.0)
            input_dict[f'{col}_1h'] = last_1h.get(col, 0.0)
            input_dict[f'{col}_1d'] = last_1d.get(col, 0.0)

        X_input = pd.DataFrame([input_dict]).fillna(0.0)

        # Ép chặt thứ tự cột đầu vào khớp 100% với cấu trúc học của bộ não XGBoost
        X_input = X_input[expected_features]

        pred = model.predict(X_input)[0]
        probas = model.predict_proba(X_input)[0]

        pct_sell = probas[0] * 100
        pct_side = probas[1] * 100
        pct_buy  = probas[2] * 100

        print("\n" + "="*50)
        print(f" [ANALYSIS LIVE] - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        print(f" 🔴 LỰC BÁN (Down)     : [{'|'*int(pct_sell/4)}{'-'*(25-int(pct_sell/4))}] {pct_sell:.2f}%")
        print(f" 🟡 ĐI NGANG (Sideways): [{'|'*int(pct_side/4)}{'-'*(25-int(pct_side/4))}] {pct_side:.2f}%")
        print(f" 🟢 LỰC MUA (Up)       : [{'|'*int(pct_buy/4)}{'-'*(25-int(pct_buy/4))}] {pct_buy:.2f}%")
        print("-" * 50)

        max_proba = probas[pred]
        if max_proba >= 0.6:
            if pred == 2:
                print(f" ==> QUYẾT ĐỊNH: LỰC MUA CHIẾM ƯU THẾ -> KHUYẾN NGHỊ BUY | Độ tin cậy: {max_proba:.2f}")
            elif pred == 0:
                print(f" ==> QUYẾT ĐỊNH: LỰC BÁN CHIẾM ƯU THẾ -> KHUYẾN NGHỊ SELL | Độ tin cậy: {max_proba:.2f}")
            else:
                print(" ==> QUYẾT ĐỊNH: Tín hiệu Sideways cân bằng -> Đứng ngoài quan sát")
        else:
            print(f" ==> QUYẾT ĐỊNH: Độ tin cậy {max_proba:.2f} quá thấp (< 60%) -> Đứng ngoài chờ dòng tiền")
        print("==================================================\n")

    except Exception as e:
        logging.error(f"Lỗi vòng lặp Live: {e}")

    time.sleep(60)