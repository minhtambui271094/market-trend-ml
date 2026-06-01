import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading_engine import TradingEngine

class AdvancedBacktesterV3:
    def __init__(self, initial_balance=10000, fee=0.0005, sl_mult=2.0, tp_mult=3.0):
        """
        Bộ máy Backtest chuyên nghiệp tích hợp quản lý vốn AI
        :param initial_balance: Số dư ban đầu (USD)
        :param fee: Phí giao dịch mỗi đầu lệnh (0.0005 = 0.05%)
        :param sl_mult: Hệ số nhân ATR để đặt Stop Loss
        :param tp_mult: Hệ số nhân ATR để đặt Take Profit
        """
        self.initial_balance = initial_balance
        self.fee = fee
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult

    def run(self, dataset_path="data/mtf_dataset.csv"):
        if not os.path.exists(dataset_path):
            print(f"❌ Không tìm thấy tập dữ liệu tại: {dataset_path}. Vui lòng kiểm tra lại!")
            return None, []

        print("... Đang đọc dữ liệu và khởi chạy Trading Engine để quét tín hiệu")
        df = pd.read_csv(dataset_path)
        
        # Gọi công cụ TradingEngine để lấy tín hiệu từ mô hình AI
        engine = TradingEngine()
        df_signals = engine.run(df)

        balance = self.initial_balance
        position = None       # Trạng thái vị thế: None, 'LONG', 'SHORT'
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        trade_units = 0.0     # Khối lượng hợp đồng/coin nắm giữ
        allocated_capital = 0.0
        
        trades_history = []
        balance_curve = []

        print(f"... Đang tiến hành mô phỏng giao dịch chi tiết trên {len(df_signals)} nến 4H")

        for i in range(len(df_signals)):
            row = df_signals.iloc[i]
            current_close = row['close_4h']
            high_price = row['high_4h']
            low_price = row['low_4h']
            atr = row['atr_4h']
            signal = row['signal']
            pos_size_factor = row['position_size']
            timestamp = row['time']

            # --- 1. KIỂM TRA ĐIỀU KIỆN THOÁT LỆNH (ĐÃ CÓ VỊ THẾ) ---
            if position == 'LONG':
                # Kiểm tra dính Stop Loss trước (Ưu tiên quản trị rủi ro)
                if low_price <= sl_price:
                    exit_price = sl_price
                    pnl = (exit_price - entry_price) * trade_units
                    fee_paid = (entry_price + exit_price) * trade_units * self.fee
                    balance += (pnl - fee_paid)
                    trades_history.append({
                        'type': 'LONG', 'result': 'SL', 'entry_time': entry_time, 'exit_time': timestamp,
                        'entry': entry_price, 'exit': exit_price, 'pnl': pnl - fee_paid, 'balance': balance
                    })
                    position = None
                # Kiểm tra dính Take Profit
                elif high_price >= tp_price:
                    exit_price = tp_price
                    pnl = (exit_price - entry_price) * trade_units
                    fee_paid = (entry_price + exit_price) * trade_units * self.fee
                    balance += (pnl - fee_paid)
                    trades_history.append({
                        'type': 'LONG', 'result': 'TP', 'entry_time': entry_time, 'exit_time': timestamp,
                        'entry': entry_price, 'exit': exit_price, 'pnl': pnl - fee_paid, 'balance': balance
                    })
                    position = None
                # Thoát sớm nếu AI đảo chiều báo hiệu Short ngược lại
                elif signal == -1:
                    exit_price = current_close
                    pnl = (exit_price - entry_price) * trade_units
                    fee_paid = (entry_price + exit_price) * trade_units * self.fee
                    balance += (pnl - fee_paid)
                    trades_history.append({
                        'type': 'LONG', 'result': 'EARLY_EXIT', 'entry_time': entry_time, 'exit_time': timestamp,
                        'entry': entry_price, 'exit': exit_price, 'pnl': pnl - fee_paid, 'balance': balance
                    })
                    position = None

            elif position == 'SHORT':
                # Kiểm tra dính Stop Loss
                if high_price >= sl_price:
                    exit_price = sl_price
                    pnl = (entry_price - exit_price) * trade_units
                    fee_paid = (entry_price + exit_price) * trade_units * self.fee
                    balance += (pnl - fee_paid)
                    trades_history.append({
                        'type': 'SHORT', 'result': 'SL', 'entry_time': entry_time, 'exit_time': timestamp,
                        'entry': entry_price, 'exit': exit_price, 'pnl': pnl - fee_paid, 'balance': balance
                    })
                    position = None
                # Kiểm tra dính Take Profit
                elif low_price <= tp_price:
                    exit_price = tp_price
                    pnl = (entry_price - exit_price) * trade_units
                    fee_paid = (entry_price + exit_price) * trade_units * self.fee
                    balance += (pnl - fee_paid)
                    trades_history.append({
                        'type': 'SHORT', 'result': 'TP', 'entry_time': entry_time, 'exit_time': timestamp,
                        'entry': entry_price, 'exit': exit_price, 'pnl': pnl - fee_paid, 'balance': balance
                    })
                    position = None
                # Thoát sớm nếu AI đảo chiều báo hiệu Long ngược lại
                elif signal == 1:
                    exit_price = current_close
                    pnl = (entry_price - exit_price) * trade_units
                    fee_paid = (entry_price + exit_price) * trade_units * self.fee
                    balance += (pnl - fee_paid)
                    trades_history.append({
                        'type': 'SHORT', 'result': 'EARLY_EXIT', 'entry_time': entry_time, 'exit_time': timestamp,
                        'entry': entry_price, 'exit': exit_price, 'pnl': pnl - fee_paid, 'balance': balance
                    })
                    position = None

            # --- 2. KIỂM TRA ĐIỀU KIỆN VÀO LỆNH (KHI ĐANG ĐỨNG NGOÀI) ---
            if position is None and signal != 0 and pos_size_factor > 0:
                entry_price = current_close
                entry_time = timestamp
                
                # Quản lý vốn: Kích thước lệnh phụ thuộc vào độ tự tin của AI
                allocated_capital = balance * pos_size_factor
                trade_units = allocated_capital / entry_price

                if signal == 1:
                    position = 'LONG'
                    sl_price = entry_price - (self.sl_mult * atr if atr > 0 else entry_price * 0.02)
                    tp_price = entry_price + (self.tp_mult * atr if atr > 0 else entry_price * 0.04)
                elif signal == -1:
                    position = 'SHORT'
                    sl_price = entry_price + (self.sl_mult * atr if atr > 0 else entry_price * 0.02)
                    tp_price = entry_price - (self.tp_mult * atr if atr > 0 else entry_price * 0.04)

            balance_curve.append(balance)

        df_signals['backtest_balance'] = balance_curve
        self._print_metrics_summary(trades_history, balance)
        self._plot_equity_curve(df_signals)

        return df_signals, trades_history

    def _print_metrics_summary(self, trades, final_balance):
        """Tính toán các chỉ số thống kê hiệu suất chuẩn quỹ đầu tư"""
        print("\n" + "="*45)
        print("📊 BÁO CÁO KẾT QUẢ KIỂM TRA QUÁ KHỨ (BACKTEST V3)")
        print("="*45)
        
        total_trades = len(trades)
        print(f"💰 Vốn ban đầu      : {self.initial_balance:,.2f} USDT")
        print(f"💵 Vốn cuối cùng     : {final_balance:,.2f} USDT")
        
        total_return_pct = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        print(f"📈 Tổng lợi nhuận (%): {total_return_pct:.2f}%")
        print(f"🏁 Tổng số lệnh đã đi: {total_trades} lệnh")

        if total_trades > 0:
            df_tr = pd.DataFrame(trades)
            winning_trades = df_tr[df_tr['pnl'] > 0]
            losing_trades = df_tr[df_tr['pnl'] <= 0]
            
            win_rate = (len(winning_trades) / total_trades) * 100
            print(f"🎯 Tỷ lệ thắng (Win): {win_rate:.2f}%")
            
            gross_profit = winning_trades['pnl'].sum()
            gross_loss = abs(losing_trades['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            print(f"⚖️ Profit Factor     : {profit_factor:.2f}")
            print(f"🟢 Lệnh thắng lớn nhất: {winning_trades['pnl'].max():+,.2f} USDT")
            print(f"🔴 Lệnh lỗ nặng nhất : {losing_trades['pnl'].min():+,.2f} USDT")
        else:
            print("📭 Không có lệnh nào được thực hiện trong suốt giai đoạn này.")
        print("="*45 + "\n")

    def _plot_equity_curve(self, df):
        """Vẽ biểu đồ tăng trưởng tài sản (Equity Curve)"""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(df['time']), df['backtest_balance'], label='Tài sản Bot AI', color='#1f77b4', linewidth=2)
            plt.axhline(y=self.initial_balance, color='gray', linestyle='--', label='Vốn gốc ban đầu')
            plt.title('BIỂU ĐỒ TĂNG TRƯỞNG TÀI SẢN CHI TIẾT (EQUITY CURVE) - BOT V3', fontsize=14, fontweight='bold')
            plt.xlabel('Thời gian', fontsize=12)
            plt.ylabel('Số dư tài khoản (USDT)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Tạo thư mục reports nếu chưa có để lưu kết quả
            os.makedirs("reports", exist_ok=True)
            plt.savefig("reports/equity_curve_v3.png", dpi=300)
            print("💾 Đã lưu biểu đồ tăng trưởng tài sản vào thư mục: reports/equity_curve_v3.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Không thể xuất biểu đồ do thiếu thư viện hiển thị đồ họa: {e}")

if __name__ == "__main__":
    # Chạy kiểm thử bộ máy Backtest offline
    backtester = AdvancedBacktesterV3(initial_balance=10000, fee=0.0005, sl_mult=2.0, tp_mult=3.0)
    backtester.run(dataset_path="data/mtf_dataset.csv")