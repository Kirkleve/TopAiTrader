import time
from config import TELEGRAM_CHAT_ID

class OrderNotifier:
    def __init__(self, bot, trader, interval=60):
        self.bot = bot
        self.trader = trader
        self.interval = interval
        self.open_positions = {}

    def check_positions(self):
        symbols = self.bot.coin_manager.get_current_coins()

        for symbol in symbols:
            position = self.trader.get_position(symbol)

            if position and symbol not in self.open_positions:
                # Позиция открылась
                side = position['side'].upper()
                amount = position['amount']
                entry_price = position['entry_price']
                self.open_positions[symbol] = position

                msg = (
                    f"🚀 Открыта новая позиция по {symbol}\n"
                    f"• Направление: {side}\n"
                    f"• Цена входа: {entry_price:.2f}$\n"
                    f"• Объём: {amount}"
                )
                self.bot.send_message(TELEGRAM_CHAT_ID, msg)

            elif not position and symbol in self.open_positions:
                # Позиция закрылась
                trade_history = self.trader.get_trade_history(symbol, limit=1)
                if trade_history:
                    trade = trade_history[0]
                    entry_price = trade['price']
                    exit_price = self.trader.exchange.fetch_ticker(symbol)['last']
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100 if trade['side'] == 'buy' else ((entry_price - exit_price) / entry_price) * 100

                    msg = (
                        f"⚠️ Позиция по {symbol} закрыта.\n"
                        f"• Цена входа: {entry_price:.2f}$\n"
                        f"• Цена выхода: {exit_price:.2f}$\n"
                        f"• PNL: {pnl_percent:+.2f}% {'✅ Профит!' if pnl_percent > 0 else '🔴 Убыток'}"
                    )
                    self.bot.send_message(TELEGRAM_CHAT_ID, msg)

                self.open_positions.pop(symbol)

    def start_monitoring(self):
        print("🟢 Запуск мониторинга позиций и PNL...")
        while True:
            try:
                self.check_positions()
                time.sleep(self.interval)
            except Exception as e:
                print(f"❌ Ошибка мониторинга позиций: {e}")
                time.sleep(self.interval)
