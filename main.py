import time
import logging
import schedule
import ccxt

# Hỗ trợ cơ chế import linh hoạt tùy thuộc vào cấu trúc thư mục dự án của bạn
try:
    from src.predict_live import LivePredictor
    from src.trading_engine import TradingEngine
except ImportError:
    from predict_live import LivePredictor
    from trading_engine import TradingEngine

# Thiết lập hệ thống ghi nhật ký (Log) chuẩn mực để theo dõi trực quan trên Render Dashboard
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class LiveTradingBot:
    def __init__(self, dry_run=True):
        """
        Trung tâm đầu não điều khiển toàn bộ chu trình hoạt động của Bot Live
        :param dry_run: True = Chạy mô phỏng (An toàn, không mất tiền), False = Giao dịch tiền thật
        """
        self.dry_run = dry_run
        logging.info(f"🤖 ĐANG KHỞI TẠO TRADING BOT - CHẾ ĐỘ: {'[MÔ PHỎNG - DRY RUN]' if dry_run else '[TÀI KHOẢN THẬT - LIVE]'}")
        
        # Khởi tạo instance cố định một lần duy nhất để tối ưu hiệu năng CPU và RAM
        self.predictor = LivePredictor()
        self.engine = TradingEngine(confidence_threshold=0.60) # Lọc tín hiệu có độ tự tin từ 60% trở lên
        
        self.symbol = 'BTC/USDT'
        
        # Cấu hình bảo mật tài khoản thực (Chỉ kích hoạt khi dry_run = False)
        if not self.dry_run:
            self.exchange = ccxt.okx({
                'apiKey': 'YOUR_API_KEY',          # Thay bằng API Key OKX của bạn
                'secret': 'YOUR_SECRET_KEY',        # Thay bằng Secret Key OKX của bạn
                'password': 'YOUR_API_PASSPHRASE',  # OKX bắt buộc phải có Passphrase riêng cho API
                'enableRateLimit': True,
            })
        else:
            # Ở chế độ mô phỏng, tái sử dụng kết nối public sẵn có của predictor để tránh tốn băng thông
            self.exchange = self.predictor.exchange

    def execute_order(self, direction, confidence):
        """Hàm điều phối đặt lệnh thị trường Live, tự động tính khối lượng vị thế dựa trên AI"""
        try:
            logging.info(f"⚡ TÍN HIỆU HỢP LỆ! Hướng: {direction} | Độ tự tin mô hình: {confidence:.2f}")
            
            # Cào dữ liệu giá hiện tại từ sàn để phục vụ tính toán
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['close']
            
            # Sử dụng bộ máy TradingEngine để tối ưu khối lượng vốn đầu tư (Position Sizing)
            # Mặc định lấy tham số thị trường cơ bản để bảo vệ tài khoản (60% tự tin tương đương ~10%-20% size vị thế)
            base_size = self.engine.position_size(confidence, atr=0.02, volatility=0.01)
            
            if self.dry_run:
                logging.info(f"🔮 [MÔ PHỎNG] Thực hiện lệnh MARKET {direction} tại mức giá {current_price:,.2f} USDT.")
                logging.info(f"💡 Khối lượng lệnh được AI khuyến nghị: {base_size * 100:.1f}% trên tổng số dư tài khoản.")
                return

            # =================================================================
            # LOGIC GIAO DỊCH TIỀN THẬT (CHỈ CHẠY KHI ĐÃ ĐIỀN API KEY & TẮT DRY RUN)
            # =================================================================
            balance = self.exchange.fetch_balance()
            usdt_available = balance['free'].get('USDT', 0.0)
            
            if usdt_available < 10.0:
                logging.warning(f"❌ Số dư USDT trên sàn không đủ điều kiện tối thiểu để giao dịch ({usdt_available:.2f} USDT).")
                return
                
            # Tính toán dòng tiền phân bổ thực tế dựa vào tỷ lệ quản lý vốn AI
            allocated_capital = usdt_available * base_size
            amount_to_trade = allocated_capital / current_price
            
            logging.info(f"💰 Tài khoản khả dụng: {usdt_available:.2f} USDT | Phân bổ lệnh: {allocated_capital:.2f} USDT ({base_size*100:.1f}%)")
            
            if direction == "BUY":
                order = self.exchange.create_market_buy_order(self.symbol, amount_to_trade)
                logging.info(f"🟢 KHỚP LỆNH MUA THÀNH CÔNG! ID: {order['id']} | Giá khớp trung bình: {order['price']}")
            elif direction == "SELL":
                order = self.exchange.create_market_sell_order(self.symbol, amount_to_trade)
                logging.info(f"🔴 KHỚP LỆNH BÁN THÀNH CÔNG! ID: {order['id']} | Giá khớp trung bình: {order['price']}")
                
        except Exception as e:
            logging.error(f"❌ Lỗi nghiêm trọng phát sinh trong quá trình đặt lệnh trên sàn: {e}")

    def run_trading_cycle(self):
        """Tiến trình cốt lõi chạy tuần hoàn: Đọc dữ liệu đa khung → Dự báo AI → Sàng lọc vị thế"""
        logging.info("--- Bắt đầu chu kỳ trading live mới ---")
        try:
            # 1. Gọi mô hình phân tích đặc trưng live đa khung thời gian để lấy tín hiệu và độ xác thực
            pred, proba = self.predictor.get_signal()
            logging.info(f"🔍 Kết quả phân tích AI -> Nhãn dự báo xu hướng: {pred} | Mức độ tự tin: {proba:.2f}")
            
            # 2. Áp dụng bộ lọc ngưỡng độ tự tin (Confidence Threshold Filter)
            if proba >= self.engine.conf_threshold:
                if pred == 2:      # NHÃN 2: XU HƯỚNG TĂNG TRƯỞNG (UP TREND) -> KÍCH HOẠT MUA
                    self.execute_order("BUY", proba)
                elif pred == 0:    # NHÃN 0: XU HƯỚNG SỤT GIẢM (DOWN TREND) -> KÍCH HOẠT BÁN
                    self.execute_order("SELL", proba)
                else:              # NHÃN 1: THỊ TRƯỜNG ĐI NGANG (SIDEWAYS)
                    logging.info("⚪ Hệ thống nhận diện thị trường không rõ xu hướng (Sideways). Giữ nguyên trạng thái đứng ngoài.")
            else:
                logging.info(f"⏳ Độ tự tin của mô hình AI ({proba:.2f}) chưa đạt ngưỡng tối thiểu ({self.engine.conf_threshold:.2f}). Bỏ qua phiên này.")
                
        except Exception as e:
            logging.error(f"💥 Lỗi không xác định xảy ra trong chu kỳ quét tín hiệu: {e}")

def main():
    # 🚨 LƯU Ý: Để chạy tiền thật, hãy thay đổi thành bot = LiveTradingBot(dry_run=False)
    bot = LiveTradingBot(dry_run=True)
    
    # Thực thi chu kỳ kiểm tra đầu tiên ngay lập tức khi khởi động để xác minh tính toàn vẹn của code
    bot.run_trading_cycle()
    
    # Thiết lập lịch trình lặp lại tự động đều đặn mỗi 1 phút (Đồng bộ nhịp tim với máy chủ Cloud)
    schedule.every(1).minutes.do(bot.run_trading_cycle)
    
    logging.info("🚀 Hệ thống Trading Bot đã lên lịch thành công và đang hoạt động ngầm...")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logging.info("🛑 Người dùng gửi lệnh ngắt thủ công. Tiến trình Bot đang được dừng an toàn.")
            break
        except Exception as e:
            logging.error(f"⚠️ Cảnh báo lỗi vòng lặp hệ thống: {e}")
            time.sleep(5) # Tránh lặp vô hạn gây nghẽn log khi có sự cố hệ điều hành

if __name__ == "__main__":
    main()