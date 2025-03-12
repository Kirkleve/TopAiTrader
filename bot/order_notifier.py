import time
from config import TELEGRAM_CHAT_ID

class OrderNotifier:
    def __init__(self, bot, trader, interval=60):
        self.bot = bot
        self.trader = trader
        self.interval = interval
        self.notified_positions = set()

    def check_positions(self):
        symbols = self.bot.coin_manager.get_current_coins()

        for symbol in symbols:
            positions = self.trader.exchange.fetch_positions([symbol])
            for pos in positions:
                symbol_name = pos['symbol']
                pos_amt = float(pos['contracts'])

                if pos_amt == 0 and symbol_name in self.notified_positions:
                    pnl = float(pos['info'].get('realizedPnl', 0))
                    msg = f"⚠️ Позиция по {symbol_name} закрылась по TP или SL.\nPNL: {pnl:.2f}$ {'✅ Профит!' if pnl > 0 else '🔴 Убыток!'}"
                    self.bot.send_message(TELEGRAM_CHAT_ID, msg)
                    self.notified_positions.remove(symbol_name)

                elif pos_amt > 0 and symbol_name not in self.notified_positions:
                    self.notified_positions.add(symbol_name)

    def start_monitoring(self):
        print("🟢 Запуск мониторинга ордеров TP и SL...")
        while True:
            try:
                self.check_positions()
                time.sleep(self.interval)
            except Exception as e:
                print(f"❌ Ошибка мониторинга позиций: {e}")
                time.sleep(self.interval)
