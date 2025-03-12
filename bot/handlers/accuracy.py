import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from data.fear_and_greed import FearGreedIndexFetcher
from trading.crypto_env import CryptoTradingEnv


def handle_accuracy(bot, message, symbol='BTC/USDT', timeframe='1h'):
    df = bot.data_fetcher.fetch_historical_data_multi_timeframe(symbol)[timeframe]

    if df.empty:
        bot.send_message(message.chat.id, f"⚠️ Нет данных для {symbol}.")
        return

    features = ['close', 'rsi', 'ema', 'adx', 'atr', 'volume']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])

    # Готовим состояние (с учетом sentiment и fear&greed)
    sentiment_result = bot.sentiment_analyzer(symbol.split('/')[0])[0]
    sentiment_score = sentiment_result['score'] if sentiment_result['label'].lower() == 'positive' else -sentiment_result['score']

    fear_greed_value, _ = FearGreedIndexFetcher.fetch_current_index()
    fear_greed_scaled = fear_greed_value / 100 if fear_greed_value else 0.5

    state_data = []
    seq_length = 20

    for i in range(seq_length, len(scaled_features)):
        lstm_input = torch.tensor(scaled_features[i - seq_length:i], dtype=torch.float32).unsqueeze(0)
        predicted_price = bot.lstm_model(lstm_input).detach().item()

        state = np.hstack([scaled_features[i], predicted_price, sentiment_score, fear_greed_scaled])
        state_data.append(state)

    env = CryptoTradingEnv(np.array(state_data), [sentiment_score]*len(state_data))
    agent = bot.dqn_agent

    state, _ = env.reset()
    done = False
    total_profit = 0.0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = agent.act(state_tensor, epsilon=0.0)  # Без случайности для оценки точности!
        next_state, reward, done, _, _ = env.step(action)

        total_profit += reward
        state = next_state

    bot.send_message(
        message.chat.id,
        f"🤖 *Оценка LSTM+DQN модели для {symbol}*\n"
        f"📊 Итоговая прибыль симуляции: {total_profit:.2f}% (PNL)\n"
        f"🚩 Оценка проведена на исторических данных таймфрейма 1h."
    )
