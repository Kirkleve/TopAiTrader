import telebot
import threading

from bot.handlers.accuracy import handle_accuracy
from bot.handlers.autotrade import handle_autotrade
from bot.handlers.help import handle_help
from bot.handlers.manage.manage import handle_manage
from bot.handlers.manage.manage_coins import CoinManager
from bot.handlers.market import handle_market
from bot.handlers.market_sentiment import handle_market_sentiment
from bot.handlers.predict import handle_predict
from bot.handlers.topnews import handle_topnews
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, BYBIT_API_KEY, BYBIT_API_SECRET
from data.fetch_data import CryptoDataFetcherBybit
from data.market_analyzer import MarketAnalyzer
from data.news_fetcher import NewsFetcher
from model.news_sentiment import SentimentAnalyzer
from trading.binance_trader import BinanceTrader
from trainer.model_manager.lstm_manager import LSTMModelManager
from trainer.model_manager.neuralprophet_manager import NeuralProphetManager
from trainer.model_manager.xgb_manager import XGBModelManager
from trainer.model_manager.ppo_manager import PPOModelManager
from .order_notifier import OrderNotifier


class TelegramBot:
    def __init__(self):
        self.bot = telebot.TeleBot(TELEGRAM_TOKEN)
        self.chat_id = TELEGRAM_CHAT_ID

        self.trader = BinanceTrader()
        self.coin_manager = CoinManager(self.trader, MarketAnalyzer(SentimentAnalyzer()))

        self.data_fetcher = CryptoDataFetcherBybit(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_analyzer = MarketAnalyzer(self.news_fetcher)

        features = ['close', 'rsi', 'ema', 'adx', 'atr', 'volume', 'cci', 'williams_r', 'momentum', 'mfi', 'mass_index']

        # Менеджеры и модели
        self.lstm_manager = LSTMModelManager('BTC_USDT', features)
        self.np_manager = NeuralProphetManager('BTC_USDT')
        self.xgb_manager = XGBModelManager('BTC_USDT')
        self.ppo_manager = PPOModelManager('BTC_USDT')

        self.lstm_models = {tf: self.lstm_manager.load_model(tf) for tf in ['15m', '1h', '4h', '1d']}
        self.lstm_scalers = {tf: self.lstm_manager.load_scaler(tf) for tf in ['15m', '1h', '4h', '1d']}
        self.np_models = {tf: self.np_manager.load_model(tf) for tf in ['15m', '1h', '4h', '1d']}
        self.xgb_model, self.xgb_scaler_X, self.xgb_scaler_y = self.xgb_manager.load_model_and_scalers()
        self.ppo_agent = self.ppo_manager.load_model()

        # Регистрация команд
        self.bot.message_handler(commands=['start', 'help'])(self.handle_help)
        self.bot.message_handler(commands=['predict'])(self.handle_predict)
        self.bot.message_handler(commands=['autotrade'])(self.handle_autotrade)
        self.bot.message_handler(commands=['manage'])(self.handle_manage)
        self.bot.message_handler(commands=['accuracy'])(self.handle_accuracy)
        self.bot.message_handler(commands=['market'])(self.handle_market)
        self.bot.message_handler(commands=['sentiment'])(self.handle_market_sentiment)
        self.bot.message_handler(commands=['topnews'])(self.handle_topnews)

        # Мониторинг ордеров в отдельном потоке
        self.order_notifier = OrderNotifier(self, self.trader)
        threading.Thread(target=self.order_notifier.start_monitoring, daemon=True).start()

    def handle_help(self, message):
        handle_help(self, message)

    def handle_predict(self, message):
        handle_predict(self, message)

    def handle_autotrade(self, _):
        symbol = 'BTC/USDT'

        # Корректное получение данных
        unified_data = self.data_fetcher.fetch_historical_data_multi_timeframe(symbol)
        current_step = -1

        # Корректное получение sentiment-данных
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(symbol.split('/')[0])
        historical_sentiment_scores = [
            (int(item['label'][0]) - 3) / 2 for item in sentiment_result
        ]

        # Корректное получение Fear & Greed Index
        fg_index, _ = self.market_analyzer.get_full_analysis()['fear_and_greed'].values()
        fg_scaled = fg_index / 100 if fg_index else 0.5
        historical_fg_scores = [fg_scaled] * len(unified_data)

        balance = self.trader.exchange.fetch_balance()['free']['USDT']

        handle_autotrade(
            self, self.chat_id, symbol, unified_data, current_step,
            historical_sentiment_scores, historical_fg_scores,
            self.lstm_manager.features, balance
        )

    def handle_manage(self, message):
        handle_manage(self, message)

    def handle_accuracy(self, message):
        handle_accuracy(self, message)

    def handle_market(self, message):
        handle_market(self, message)

    def handle_market_sentiment(self, message):
        handle_market_sentiment(self, message)

    def handle_topnews(self, message):
        handle_topnews(self, message)