import threading
import logging
from bot.manage_coins import CoinManager
from bot.order_notifier import OrderNotifier
from components import initialize_components
from trading.binance_trader import BinanceTrader
from predictor_loader import load_models_for_symbols

from bot.telegram_bot import TelegramBot
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logging.getLogger("transformers").setLevel(logging.ERROR)

# Инициализация компонентов
data_fetcher, sentiment_analyzer, news_fetcher, summarizer, market_analyzer = initialize_components()

# Binance trader и менеджер монет
trader = BinanceTrader()
coin_manager = CoinManager(trader, data_fetcher)

symbols_to_trade = coin_manager.get_current_coins()

if not symbols_to_trade:
    default_coin = 'BTC/USDT'
    coin_manager.add_coin('BTC')
    symbols_to_trade = [default_coin]
    print(f"⚠️ Список монет пуст, добавлена {default_coin}")

# Загрузка моделей и агентов
models = load_models_for_symbols(symbols_to_trade)

# TelegramBot
telegram_bot = TelegramBot(
    token=TELEGRAM_TOKEN,
    chat_id=TELEGRAM_CHAT_ID,
    models=models,
    data_fetcher=data_fetcher,
    sentiment_analyzer=sentiment_analyzer,
    news_fetcher=news_fetcher,
    summarizer=summarizer,
    market_analyzer=market_analyzer,
    symbols_to_trade=symbols_to_trade
)

telegram_bot.coin_manager = CoinManager(BinanceTrader(), data_fetcher)
telegram_bot.start()

# OrderNotifier
order_notifier = OrderNotifier(telegram_bot, telegram_bot.trader, interval=60)
threading.Thread(target=order_notifier.check_positions, daemon=True).start()
