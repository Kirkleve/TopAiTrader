import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from data.fear_and_greed import FearGreedIndexFetcher
from model.news_sentiment import SentimentAnalyzer
from trading.crypto_env import CryptoTradingEnv
from trading.agent import DQNAgent
from data.fetch_data import CryptoDataFetcher


def train_agent(symbol: str, episodes=50, batch_size=32):
    features = ['close', 'rsi', 'ema', 'adx', 'atr', 'volume']
    timeframes = ['15m', '1h', '4h', '1d']
    sentiment_analyzer = SentimentAnalyzer()

    fetcher = CryptoDataFetcher()
    combined_data = []

    for timeframe in timeframes:
        df_dict = fetcher.fetch_historical_data_multi_timeframe(symbol, [timeframe])
        df = df_dict[timeframe]

        if df is None or df.empty:
            print(f"⚠️ Нет данных для {symbol} [{timeframe}], пропускаем.")
            continue

        # Масштабирование признаков
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[features])

        # Анализ sentiment для текущего таймфрейма
        sentiment_score = sentiment_analyzer.analyze_sentiment([symbol])
        sentiment_column = np.full((scaled_features.shape[0], 1), sentiment_score)

        # Fear & Greed Index
        fear_greed_value, _ = FearGreedIndexFetcher.fetch_current_index()
        fear_greed_scaled = (fear_greed_value / 100) if fear_greed_value is not None else 0.5
        fear_greed_column = np.full((scaled_features.shape[0], 1), fear_greed_scaled)

        # Объединяем все данные
        combined_features = np.hstack((scaled_features, sentiment_column, fear_greed_column))
        combined_data.append(combined_features)

    if len(combined_data) == 0:
        raise ValueError(f"⚠️ Нет данных для {symbol} на всех таймфреймах.")

    combined_data = np.vstack(combined_data)

    sentiment_scores = combined_data[:, -2]  # предпоследний столбец sentiment
    env = CryptoTradingEnv(combined_data, sentiment_scores)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n  # type: ignore

    agent = DQNAgent(state_size, action_size)

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            agent.memory.append((state, action, reward, next_state_tensor, done))

            state = next_state_tensor
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                print(f"Эпизод {episode + 1}/{episodes}, общая прибыль: {total_reward:.2f}")

    # Сохранение модели
    os.makedirs('trading', exist_ok=True)
    torch.save(agent.model.state_dict(), f'trading/trained_agent_{symbol.replace("/", "_")}.pth')
    print(f"✅ Агент для {symbol} обучен и модель сохранена!")

    return agent
