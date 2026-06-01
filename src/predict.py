import os
import joblib
import pandas as pd

def main():
    model_path = "models/trend_model.pkl"
    dataset_path = "data/mtf_dataset.csv"

    # 1. Kiểm tra sự tồn tại của file mô hình và dữ liệu
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy mô hình tại: {model_path}. Hãy chạy train.py trước!")
        return
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy tập dữ liệu MTF tại: {dataset_path}. Hãy chạy mtf_dataset.py trước!")
        return

    # 2. Tải mô hình đã train
    print("... Đang tải mô hình XGBoost")
    model = joblib.load(model_path)

    # 3. Đọc dữ liệu đa khung thời gian (MTF)
    df = pd.read_csv(dataset_path)
    if df.empty:
        print("❌ File dữ liệu trống!")
        return

    # 4. Định nghĩa chính xác 51 đặc trưng mà mô hình yêu cầu (giống lúc train)
    features = [
        'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h', 'ema20_4h_4h', 'ema78_4h_4h', 'ema200_4h_4h', 'rsi_4h_4h', 'macd_4h_4h', 'hist_4h_4h', 'ema20_gap_4h_4h', 'ema78_gap_4h_4h', 'ema200_gap_4h_4h', 'ema20_slope_4h_4h', 'ema78_slope_4h_4h', 'volatility_4h_4h',
        'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h', 'ema20_1h_1h', 'ema78_1h_1h', 'ema200_1h_1h', 'rsi_1h_1h', 'macd_1h_1h', 'hist_1h_1h', 'ema20_gap_1h_1h', 'ema78_gap_1h_1h', 'ema200_gap_1h_1h', 'ema20_slope_1h_1h', 'ema78_slope_1h_1h', 'volatility_1h_1h',
        'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d', 'ema20_1d_1d', 'ema78_1d_1d', 'ema200_1d_1d', 'rsi_1d_1d', 'macd_1d_1d', 'hist_1d_1d', 'ema20_gap_1d_1d', 'ema78_gap_1d_1d', 'ema200_gap_1d_1d', 'ema20_slope_1d_1d', 'ema78_slope_1d_1d', 'volatility_1d_1d'
    ]

    # Kiểm tra xem file csv có đủ các cột đặc trưng này không
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        print(f"❌ Dữ liệu thiếu các cột đặc trưng sau: {missing_cols}")
        return

    # 5. Lấy dòng dữ liệu mới nhất (nến gần nhất) để thực hiện dự báo xu hướng
    latest_data = df.iloc[[-1]]
    X_input = latest_data[features]

    # 6. Tiến hành dự báo nhãn xu hướng và xác suất (Độ tự tin)
    pred_label = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    # Ánh xạ nhãn số sang chuỗi hiển thị tương ứng
    labels_map = {
        0: "DOWN TREND (Xu hướng giảm)",
        1: "SIDEWAYS (Đi ngang / Nhiễu)",
        2: "UP TREND (Xu hướng tăng)"
    }

    print("\n" + "="*40)
    print(f"⏰ THỜI GIAN NẾN: {latest_data['time'].values[0]}")
    print(f"💰 GIÁ ĐÓNG CỬA (4H): {latest_data['close_4h'].values[0]}")
    print(f"🤖 XU HƯỚNG DỰ BÁO: {labels_map[pred_label]}")
    print(f"📊 ĐỘ TỰ TIN (XÁC SUẤT):")
    print(f"   - Giảm (Class 0): {proba[0]*100:.2f}%")
    print(f"   - Đi ngang (Class 1): {proba[1]*100:.2f}%")
    print(f"   - Tăng (Class 2): {proba[2]*100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()