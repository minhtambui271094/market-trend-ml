import time
import logging
import schedule  # Thư viện giúp đặt lịch chạy chu kỳ
from datetime import datetime
from src.predict_live import LivePredictor
from src.trading_engine import TradingEngine

# Thiết lập log để theo dõi song song trên Dashboard của Render
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def generate_progress_bar(prob, length=25):
    """Hàm bổ trợ tạo thanh tiến trình dạng [|||||-----] dựa trên tỷ lệ phần trăm"""
    num_pipes = round(prob * length)
    num_dashes = length - num_pipes
    return "[" + "|" * num_pipes + "-" * num_dashes + "]"

def run_trading_cycle():
    """Hàm thực hiện chu kỳ: Đọc data tự động -> Dự báo AI -> Xuất bảng thống kê -> Giao dịch"""
    try:
        # 1. Khởi tạo instance kết nối sàn và bộ máy thực thi
        predictor = LivePredictor()
        engine = TradingEngine()
        
        # 2. Lấy dự báo từ model XGBoost (Trả về: nhãn định danh và danh sách 3 mức xác suất)
        pred, probas = predictor.get_signal() 
        
        prob_down = probas[0]      # Xác suất Lực Bán (Nhãn 0)
        prob_sideways = probas[1]  # Xác suất Đi Ngang (Nhãn 1)
        prob_up = probas[2]        # Xác suất Lực Mua (Nhãn 2)
        
        # Vẽ các thanh tiến trình tương ứng với độ dài chuẩn 25 ký tự
        bar_down = generate_progress_bar(prob_down)
        bar_sideways = generate_progress_bar(prob_sideways)
        bar_up = generate_progress_bar(prob_up)
        
        # 3. Logic lọc tín hiệu (THRESHOLD) & Xác định chuỗi văn bản Quyết định
        decision_str = ""
        proba_confidence = probas[pred]  # Độ tự tin của nhãn AI chọn cao nhất
        
        if proba_confidence > 0.6:
            if pred == 2:    # LỰC MUA CHIẾM ƯU THẾ TUYỆT ĐỐI
                engine.execute_buy()
                decision_str = f"🟢 Kích hoạt LỆNH MUA (Up) với độ tự tin {proba_confidence*100:.2f}%"
            elif pred == 0:  # LỰC BÁN CHIẾM ƯU THẾ TUYỆT ĐỐI
                engine.execute_sell()
                decision_str = f"🔴 Kích hoạt LỆNH BÁN (Down) với độ tự tin {proba_confidence*100:.2f}%"
            else:            # Hệ thống đi ngang Sideways rất mạnh (>60%)
                decision_str = "Tín hiệu Sideways cân bằng -> Đứng ngoài quan sát"
        else:
            # Nếu AI chọn xu hướng nhưng độ tự tin quá thấp (<= 60%)
            if pred == 1:
                decision_str = "Tín hiệu Sideways cân bằng -> Đứng ngoài quan sát"
            else:
                decision_str = f"Độ tự tin thấp ({proba_confidence*100:.2f}%) -> Giữ nguyên vị thế, đứng ngoài quan sát"
                
        # 4. Xuất giao diện đồ họa LIVE ANALYSIS ra màn hình Console / Render Logs
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        live_report = (
            f"\n==================================================\n"
            f" [ANALYSIS LIVE] - {current_time}\n"
            f"==================================================\n"
            f" 🔴 LỰC BÁN (Down)     : {bar_down} {prob_down*100:.2f}%\n"
            f" 🟡 ĐI NGANG (Sideways): {bar_sideways} {prob_sideways*100:.2f}%\n"
            f" 🟢 LỰC MUA (Up)       : {bar_up} {prob_up*100:.2f}%\n"
            f"--------------------------------------------------\n"
            f" ==> QUYẾT ĐỊNH: {decision_str}\n"
            f"=================================================="
        )
        
        # In ra màn hình ngay lập tức (flush=True giúp Render không bị nghẽn log buffer)
        print(live_report, flush=True)
            
    except Exception as e:
        logging.error(f"Lỗi cục bộ phát sinh trong chu kỳ giao dịch: {e}")

def main():
    logging.info("Hệ thống Trading Bot đa khung thời gian đã sẵn sàng trên Cloud!")
    
    # Kích hoạt chạy lượt đầu tiên ngay khi khởi động bot
    run_trading_cycle()
    
    # Lên lịch lặp đi lặp lại chuẩn xác mỗi 1 phút
    schedule.every(1).minutes.do(run_trading_cycle)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()