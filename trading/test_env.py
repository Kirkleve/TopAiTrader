import numpy as np
from data.fetch_data import CryptoDataFetcher
from main import prepare_lstm_data
from trading.crypto_env import CryptoTradingEnv

data_fetcher = CryptoDataFetcher()
market_data = data_fetcher.fetch_historical_data(['BTC/USDT'])
historical_data = market_data['BTC/USDT']

# Подготовка данных
train_data, train_labels, scaler = prepare_lstm_data(historical_data)

sentiment_dummy = np.zeros(train_data.shape[0])
env = CryptoTradingEnv(train_data.numpy(), sentiment_dummy)

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()