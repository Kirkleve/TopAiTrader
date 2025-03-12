from data.fetch_data import CryptoDataFetcher
from model.news_sentiment import SentimentAnalyzer
from data.fear_and_greed import FearGreedIndexFetcher
from trainer.lstm_trainer import LSTMTrainer
from trainer.dqn_trainer import DQNTrainer
import numpy as np
import torch

class UniversalTrainer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.features = ['close', 'rsi', 'ema', 'adx', 'atr', 'volume']
        self.fetcher = CryptoDataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()

    def run_training(self):
        data_multi = self.fetcher.fetch_historical_data_multi_timeframe(self.symbol)
        lstm_trainer = LSTMTrainer(self.features)

        lstm_models = {}
        scalers = {}

        # Сначала полностью обучаем LSTM для всех таймфреймов
        for timeframe, df in data_multi.items():
            if df.empty:
                print(f"⚠️ Нет данных для {timeframe}, пропускаем обучение.")
                continue

            print(f"⏳ Обучаю LSTM модель [{timeframe}] для {self.symbol}...")
            lstm_model, scaler = lstm_trainer.train_and_save(df, self.symbol, timeframe)
            lstm_models[timeframe] = lstm_model
            scalers[timeframe] = scaler
            print(f"✅ LSTM модель [{timeframe}] обучена и сохранена!")

        # И только теперь (после всех LSTM) обучаем DQN для 1h
        if '1h' in data_multi and not data_multi['1h'].empty:
            df = data_multi['1h']
            sentiment_score = self.sentiment_analyzer.analyze_sentiment([self.symbol.split('/')[0]])
            fear_greed_index, _ = FearGreedIndexFetcher.fetch_current_index()
            fg_scaled = fear_greed_index / 100 if fear_greed_index else 0.5

            scaled_data = scalers['1h'].transform(df[self.features])
            state_data = []

            for i in range(lstm_trainer.seq_length, len(scaled_data)):
                lstm_input = torch.tensor(
                    scaled_data[i - lstm_trainer.seq_length:i], dtype=torch.float32
                ).unsqueeze(0)
                predicted_price = lstm_models['1h'](lstm_input).detach().item()

                state = np.hstack([scaled_data[i], predicted_price, sentiment_score, fg_scaled])
                state_data.append(state)

            state_data = np.array(state_data)
            sentiment_scores = [float(sentiment_score)] * len(state_data)

            print(f"🚀 Начинаю обучение DQN агента для {self.symbol}...")
            dqn_trainer = DQNTrainer(state_data, sentiment_scores)
            dqn_trainer.train_and_save(self.symbol)

        print(f"✅ Полное обучение (LSTM всех таймфреймов + DQN) завершено!")

