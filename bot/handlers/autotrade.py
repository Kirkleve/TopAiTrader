from typing import Literal
import torch
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from model.news_sentiment import SentimentAnalyzer
from trading.agent import DQNAgent
from trading.binance_trader import BinanceTrader
from data.fear_and_greed import FearGreedIndexFetcher
from utils.adaptive_strategy import adapt_strategy
from trainer.train import train
from data.fetch_data import CryptoDataFetcher
from model.lstm_price_predictor import LSTMPricePredictor
from data.atr import calculate_atr


def handle_autotrade(bot, chat_id):
    symbols = bot.coin_manager.get_current_coins()
    trader = BinanceTrader()  # ✅ Инициализация трейдера!
    data_fetcher = CryptoDataFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    strategy_params = adapt_strategy()

    bot.send_message(chat_id, f"🤖 Запускаю автотрейдинг для: {', '.join(symbols)}")

    balance = trader.exchange.fetch_balance()['free']['USDT']
    risk_percent = 0.02
    seq_length = 20
    features = ['close', 'rsi', 'ema', 'adx', 'atr', 'volume']

    for symbol in symbols:
        historical_data = data_fetcher.fetch_historical_data_multi_timeframe(symbol)
        if not historical_data or historical_data['1h'].empty:
            bot.send_message(chat_id, f"⚠️ Нет данных по {symbol}, пропускаем.")
            continue

        symbol_dir = symbol.replace('/', '_')
        lstm_model_path = f'trainer/models/{symbol_dir}/{symbol_dir}_1h_lstm.pth'
        dqn_model_path = f'trading/trained_agent_{symbol_dir}.pth'

        if not os.path.exists(lstm_model_path) or not os.path.exists(dqn_model_path):
            bot.send_message(chat_id, f"⏳ Обучаю модели для {symbol} (~3-5 мин.)...")
            train(symbol)
            bot.send_message(chat_id, f"✅ Модели для {symbol} успешно обучены!")

        lstm_model = LSTMPricePredictor(input_size=len(features))
        lstm_model.load_state_dict(torch.load(lstm_model_path))
        lstm_model.eval()

        agent = DQNAgent(state_size=len(features) + 3, action_size=3)
        agent.model.load_state_dict(torch.load(dqn_model_path))
        agent.model.eval()

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(historical_data['1h'][features])

        sentiment_score = sentiment_analyzer.analyze_sentiment([symbol.split('/')[0]])
        fear_greed_value, _ = FearGreedIndexFetcher.fetch_current_index()
        fear_greed_scaled = fear_greed_value / 100 if fear_greed_value else 0.5

        lstm_input = torch.tensor(scaled_features[-seq_length:], dtype=torch.float32).unsqueeze(0)
        predicted_price = lstm_model(lstm_input).detach().item()

        current_state = np.hstack([
            scaled_features[-1],
            predicted_price,
            sentiment_score,
            fear_greed_scaled
        ])

        state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)

        action = agent.act(state_tensor)

        if action == 0:
            bot.send_message(chat_id, f"⏳ {symbol}: агент решил не торговать.")
            continue

        side: Literal['buy', 'sell'] = 'buy' if action == 1 else 'sell'

        atr = calculate_atr(historical_data['1h'])

        position = trader.get_position(symbol)
        if position:
            trader.close_all_positions(symbol)
            bot.send_message(chat_id, f"🔄 Закрыта текущая позиция по {symbol}.")

        trader.create_order_with_sl_tp(
            symbol=symbol,
            side=side,
            balance=balance,
            risk_percent=risk_percent,  # ← теперь чётко используется!
            atr=atr,
            sl_multiplier=strategy_params.get('sl_multiplier', 1.5),
            tp_multiplier=strategy_params.get('tp_multiplier', 3),
            adaptive=True
        )

        message = (
            f"🚀 Открыта позиция {side.upper()} по {symbol}\n"
            f"💵 Цена: {historical_data['1h']['close'].iloc[-1]:.2f}$\n"
            f"📊 ATR: {atr:.4f}\n"
            f"⚙️ Размер позиции: адаптивный\n"
            f"🔄 Порог стратегии: {strategy_params['threshold_percent']:.2%}\n"
            f"🔮 Sentiment порог: {strategy_params['sentiment_threshold']:.2f}"
        )

        bot.send_message(chat_id, message)

    bot.send_message(chat_id, "✅ Автотрейдинг по всем монетам завершён!")
    bot.send_message(chat_id, f"🔄 Параметры стратегии адаптированы:\n"
                              f"threshold_percent: {strategy_params['threshold_percent']:.4f}\n"
                              f"sentiment_threshold: {strategy_params['sentiment_threshold']:.2f}")
