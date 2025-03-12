import threading
import time
import telebot
import json
from bot.manage_coins import CoinManager
from bot.handlers.manage import handle_manage
from bot.handlers.sentiment import handle_sentiment
from bot.handlers.market import handle_market
from bot.handlers.topnews import handle_topnews
from bot.handlers.predict import handle_predict
from bot.handlers.accuracy import handle_accuracy
from bot.handlers.autotrade import handle_autotrade
from bot.handlers.help import handle_help
from trading.binance_trader import BinanceTrader


class TelegramBot:
    def __init__(self, token, lstm_model, dqn_agent, scaler, data_fetcher, sentiment_analyzer,
                 news_fetcher, summarizer, market_analyzer, symbols_to_trade):

        self.bot = telebot.TeleBot(token)
        self.lstm_model = lstm_model
        self.dqn_agent = dqn_agent  # ← Новая модель DQN!
        self.scaler = scaler
        self.data_fetcher = data_fetcher
        self.sentiment_analyzer = sentiment_analyzer
        self.news_fetcher = news_fetcher
        self.summarizer = summarizer
        self.market_analyzer = market_analyzer
        self.trader = BinanceTrader()
        self.coin_manager = CoinManager(self.trader, self.data_fetcher)
        self.autotrade_file = 'autotrade_state.json'
        self.is_autotrading = self.load_autotrade_state()
        self.symbols_to_trade = symbols_to_trade

        self.save_autotrade_state()

    def save_autotrade_state(self):
        with open(self.autotrade_file, 'w') as f:
            json.dump({"is_autotrading": self.is_autotrading}, f)

    def load_autotrade_state(self):
        try:
            with open(self.autotrade_file, 'r') as f:
                state = json.load(f)
                return state.get("is_autotrading", False)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def send_message(self, chat_id, text, parse_mode='Markdown', **kwargs):
        self.bot.send_message(chat_id, text, parse_mode=parse_mode, **kwargs)

    def start_autotrade(self, chat_id):
        if self.is_autotrading:
            self.send_message(chat_id, "⚠️ Автотрейдинг уже запущен!")
            return

        self.is_autotrading = True
        self.save_autotrade_state()
        self.send_message(chat_id, "🤖 Автотрейдинг запущен!")

        threading.Thread(target=self.run_autotrade_loop, args=(chat_id,), daemon=True).start()

    def stop_autotrade(self, chat_id):
        self.is_autotrading = False
        self.save_autotrade_state()

        symbols = self.coin_manager.get_current_coins()
        for symbol in symbols:
            self.trader.close_all_positions(symbol)
            self.trader.cancel_all_orders(symbol)

        self.send_message(chat_id, "🛑 Автотрейдинг остановлен, все позиции и ордера закрыты!")

    def run_autotrade_loop(self, chat_id):
        while self.is_autotrading:
            # ← Передаём ОБЕ модели в handle_autotrade
            handle_autotrade(self, chat_id)
            time.sleep(900)

    def send_welcome_message(self, chat_id):
        commands_text = (
            "🚀 Доступные команды:\n"
            "/manage — 💎 Управление монетами\n"
            "/predict — 📈 Прогноз BTC\n"
            "/accuracy — 📊 Точность модели\n"
            "/sentiment — 📰 Анализ новостей\n"
            "/market — 📌 Обзор рынка\n"
            "/topnews — 🔥 Новости\n"
            "/autotrade — 🤖 Запуск автотрейда\n"
            "/stop — 🛑 Остановить автотрейд\n"
            "/help — 📖 Помощь"
        )
        self.send_message(chat_id, commands_text)

    def start(self, chat_id):
        self.bot.message_handler(commands=['manage'])(lambda msg: handle_manage(self, msg))
        self.bot.message_handler(commands=['predict'])(lambda msg: handle_predict(self, msg))
        self.bot.message_handler(commands=['accuracy'])(lambda msg: handle_accuracy(self, msg))
        self.bot.message_handler(commands=['sentiment'])(lambda msg: handle_sentiment(self, msg))
        self.bot.message_handler(commands=['market'])(lambda msg: handle_market(self, msg))
        self.bot.message_handler(commands=['topnews'])(lambda msg: handle_topnews(self, msg))
        self.bot.message_handler(commands=['autotrade'])(lambda msg: self.start_autotrade(msg.chat.id))
        self.bot.message_handler(commands=['stop'])(lambda msg: self.stop_autotrade(msg.chat.id))
        self.bot.message_handler(commands=['help'])(lambda msg: handle_help(self, msg))

        self.send_welcome_message(chat_id)
        print(f"🚀 Телеграм-бот успешно запущен и готов торговать: {', '.join(self.symbols_to_trade)}")

        self.bot.polling()
