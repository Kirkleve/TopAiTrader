import telebot
import json
import threading
import time
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
    def __init__(self, token, models, data_fetcher, sentiment_analyzer,
                 news_fetcher, summarizer, market_analyzer, symbols_to_trade, chat_id, xgb_predictor=None):

        self.bot = telebot.TeleBot(token)
        self.models = models
        self.data_fetcher = data_fetcher
        self.sentiment_analyzer = sentiment_analyzer
        self.news_fetcher = news_fetcher
        self.summarizer = summarizer
        self.market_analyzer = market_analyzer
        self.symbols_to_trade = symbols_to_trade
        self.chat_id = chat_id
        self.trader = BinanceTrader()
        self.coin_manager = CoinManager(self.trader, self.data_fetcher)
        self.autotrade_file = 'autotrade_state.json'
        self.is_autotrading = self.load_autotrade_state()
        self.xgb_predictor = xgb_predictor
        self.handlers()

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

    def send_message(self, text, parse_mode=None):
        try:
            if parse_mode:
                self.bot.send_message(self.chat_id, text, parse_mode=parse_mode)
            else:
                self.bot.send_message(self.chat_id, text)
        except Exception as e:
            print(f"❌ Ошибка отправки сообщения в Telegram: {e}")

    def start_autotrade(self, message):
        if self.is_autotrading:
            self.send_message("⚠️ Автотрейдинг уже запущен!")
            return

        self.is_autotrading = True
        self.save_autotrade_state()
        self.send_message("🤖 Автотрейдинг запущен!")
        threading.Thread(target=self.run_autotrade_loop, daemon=True).start()

    def stop_autotrade(self, message):
        self.is_autotrading = False
        self.save_autotrade_state()

        symbols = self.coin_manager.get_current_coins()
        for symbol in symbols:
            self.trader.close_all_positions(symbol)
            self.trader.cancel_all_orders(symbol)

        self.send_message("🛑 Автотрейдинг остановлен, все позиции закрыты!")

    def run_autotrade_loop(self):
        while self.is_autotrading:
            try:
                handle_autotrade(self, self.chat_id)
                time.sleep(900)
            except Exception as e:
                print(f"⚠️ Ошибка автотрейдинга: {e}")
                time.sleep(900)

    def send_welcome_message(self):
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
        self.send_message(commands_text)

    def handlers(self):
        self.bot.message_handler(commands=['manage'])(lambda msg: handle_manage(self, msg))
        self.bot.message_handler(commands=['predict'])(lambda msg: handle_predict(self, msg))
        self.bot.message_handler(commands=['accuracy'])(lambda msg: handle_accuracy(self, msg))
        self.bot.message_handler(commands=['sentiment'])(lambda msg: handle_sentiment(self, msg))
        self.bot.message_handler(commands=['market'])(lambda msg: handle_market(self, msg))
        self.bot.message_handler(commands=['topnews'])(lambda msg: handle_topnews(self, msg))
        self.bot.message_handler(commands=['autotrade'])(self.start_autotrade)
        self.bot.message_handler(commands=['stop'])(self.stop_autotrade)
        self.bot.message_handler(commands=['help'])(lambda msg: handle_help(self, msg))

    def start(self):
        self.send_welcome_message()
        print(f"🚀 Телеграм-бот успешно запущен и готов к торговле: {', '.join(self.coin_manager.get_current_coins())}")

        self.bot.polling()
