import time
import logging
import schedule # Thư viện này giúp bạn đặt lịch chạy cực chuẩn
from src.predict_live import LivePredictor
from src.trading_engine import TradingEngine

# Thiết lập log để theo dõi trên Dashboard của Render
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def run_trading_cycle():
    """Hàm này sẽ thực hiện chu kỳ: Đọc data -> Dự báo -> Giao dịch"""
    logging.info("--- Bắt đầu chu kỳ trading mới ---")
    try:
        # 1. Khởi tạo (hoặc dùng instance đã có)
        predictor = LivePredictor()
        engine = TradingEngine()
        
        # 2. Lấy dự báo từ model
        # Giả sử hàm get_signal() trả về nhãn (0, 1, 2) và độ tự tin (proba)
        pred, proba = predictor.get_signal() 
        
        logging.info(f"Dự báo: {pred} | Độ tự tin: {proba:.2f}")
        
        # 3. Logic thực thi (Cần điều kiện lọc - THRESHOLD)
        # Chỉ vào lệnh nếu độ tự tin > 0.6 (60%)
        if proba > 0.6:
            if pred == 2: # UP
                engine.execute_buy()
            elif pred == 0: # DOWN
                engine.execute_sell()
            else:
                logging.info("Tín hiệu Sideways, không giao dịch.")
        else:
            logging.info("Độ tự tin thấp, giữ nguyên vị thế.")
            
    except Exception as e:
        logging.error(f"Lỗi trong chu kỳ giao dịch: {e}")

def main():
    logging.info("Hệ thống Trading Bot đã sẵn sàng trên Render!")
    
    # Chạy chu kỳ đầu tiên ngay khi khởi động
    run_trading_cycle()
    
    # Thiết lập lịch chạy mỗi 1 phút (hoặc theo timeframe của bạn)
    schedule.every(1).minutes.do(run_trading_cycle)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()