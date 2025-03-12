import os
import threading
import logging
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
from bot.manage_coins import CoinManager
from bot.order_notifier import OrderNotifier
from data.fetch_data import CryptoDataFetcher
from data.cryptopanic_data import CryptoNewsFetcher
from data.market_analyzer import MarketAnalyzer
from data.news_summary import NewsSummarizer
from trainer.train import train
from trainer.model_loader import load_models
from trading.binance_trader import BinanceTrader
from bot.telegram_bot import TelegramBot
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logging.getLogger("transformers").setLevel(logging.ERROR)

# Инициализация компонентов
data_fetcher = CryptoDataFetcher()
trader = BinanceTrader()
coin_manager = CoinManager(trader, data_fetcher)

# Проверка и добавление монет
symbols_to_trade = coin_manager.get_current_coins()
if not symbols_to_trade:
    default_coin = 'BTC/USDT'
    coin_manager.add_coin('BTC')
    symbols_to_trade = [default_coin]

# Проверяем модели и запускаем обучение, если нужно
for symbol in symbols_to_trade:
    symbol_dir = symbol.replace('/', '_')
    lstm_model_dir = f'trainer/models/{symbol_dir}'
    lstm_model_path = f'{lstm_model_dir}/{symbol_dir}_1h_lstm.pth'
    dqn_model_path = f'trading/trained_agent_{symbol_dir}.pth'

    if not os.path.exists(lstm_model_path) or not os.path.exists(dqn_model_path):
        print(f"⚠️ Модели для {symbol} не найдены, запускаю обучение...")
        train(symbol)

# Загрузка моделей для основной монеты (BTC/USDT)
symbol_dir = 'BTC_USDT'
lstm_model, dqn_agent = load_models('BTC/USDT')

# Scaler (для LSTM)
market_data = data_fetcher.fetch_historical_data_multi_timeframe('BTC/USDT')['1h']
features = ['close', 'rsi', 'ema', 'adx', 'atr', 'volume']
scaler = MinMaxScaler()
scaler.fit(market_data[features])

# Анализаторы
sentiment_analyzer = pipeline("sentiment-analysis", model='nlptown/bert-base-multilingual-uncased-sentiment')
translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer = NewsSummarizer(translator_pipeline, summarization_pipeline)

news_fetcher = CryptoNewsFetcher(summarizer)
market_analyzer = MarketAnalyzer(news_fetcher)

# Инициализация Telegram-бота (полная версия)
telegram_bot = TelegramBot(
    TELEGRAM_TOKEN,
    lstm_model,
    dqn_agent,
    scaler,
    data_fetcher,
    sentiment_analyzer,
    news_fetcher,
    summarizer,
    market_analyzer,
    symbols_to_trade  # ← обязательный параметр, не забывай!
)

telegram_bot.coin_manager = coin_manager
telegram_bot.start(TELEGRAM_CHAT_ID)

# Запуск потока проверки позиций
order_notifier = OrderNotifier(telegram_bot, telegram_bot.trader, interval=60)
order_thread = threading.Thread(target=order_notifier.check_positions, daemon=True)
order_thread.start()
