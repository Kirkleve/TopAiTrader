import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from data.fetch_data import CryptoDataFetcher
from data.fear_and_greed import FearGreedIndexFetcher
from model.CNN.cnn_data_preprocessor import CNNDataPreprocessor
from trainer.cnn_trainer import CNNTrainer
from trainer.lstm_trainer import LSTMTrainer
from trainer.ppo_trainer import PPOTrainer
from trainer.model_manager.cnn_model import CNNModelManager
from trainer.model_manager.lstm_model import LSTMModelManager
from trainer.model_manager.xgb_model import XGBModelManager
from trainer.model_manager.ppo_model import PPOModelManager
from model.xgboost_predictor import XGBoostPredictor
from model.news_sentiment import SentimentAnalyzer


class UniversalTrainer:
    def __init__(self, symbol, device='cpu'):
        self.symbol = symbol
        self.features = [
            'close', 'rsi', 'ema', 'adx', 'atr', 'volume',
            'cci', 'williams_r', 'momentum', 'mfi', 'mass_index'
        ]
        self.device = device
        self.fetcher = CryptoDataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer(model_type='finbert')

        self.cnn_manager = CNNModelManager(symbol)
        self.lstm_manager = LSTMModelManager(symbol, self.features)
        self.xgb_manager = XGBModelManager(symbol)
        self.ppo_manager = PPOModelManager(symbol)

        self.timeframes = ['15m', '1h', '4h', '1d']

    def run_training(self):
        data_multi = self.fetcher.fetch_historical_data_multi_timeframe(self.symbol, self.timeframes)

        # CNN preprocessing
        preprocessor = CNNDataPreprocessor(save_dir=os.path.join("models", self.symbol.replace('/', '_'), "cnn"))
        preprocessor.generate_candlestick_images(data_multi, self.symbol)

        cnn_models = {}
        lstm_models = {}
        scalers = {}

        # CNN обучение
        for tf in self.timeframes:
            cnn_model = self.cnn_manager.load_model(tf)
            if not cnn_model:
                cnn_trainer = CNNTrainer(self.symbol, timeframe=tf, epochs=10, device=self.device)
                image_dir = os.path.join("models", self.symbol.replace('/', '_'), "cnn",
                                         f"{self.symbol.replace('/', '_')}_{tf}")
                cnn_model = cnn_trainer.train(image_dir=image_dir)
            cnn_models[tf] = cnn_model

        # LSTM обучение
        for tf in self.timeframes:
            lstm_model = self.lstm_manager.load_model(tf)
            scaler = self.lstm_manager.load_scaler(tf)
            if not lstm_model or not scaler:
                print(f"🚀 LSTM обучение [{tf}]...")
                lstm_trainer = LSTMTrainer(self.symbol, self.features, tf, epochs=50, device=self.device)
                lstm_model, scaler = lstm_trainer.train(data_multi[tf])

            lstm_models[tf] = lstm_model
            scalers[tf] = scaler

        # XGB обучение
        xgb_model, scaler_X, scaler_y = self.xgb_manager.load_model_and_scalers()

        if not xgb_model or not scaler_X or not scaler_y:
            print("🚀 XGBoost обучение [1h]...")
            df_1h = data_multi["1h"]
            X = df_1h[self.features].iloc[:-1].values
            y = df_1h['close'].shift(-1).dropna().values.reshape(-1, 1)

            scaler_X, scaler_y = StandardScaler(), StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y).flatten()

            xgb_model = XGBoostPredictor(self.symbol)
            xgb_model.train(X_scaled, y_scaled)

            self.xgb_manager.save_model_and_scalers(xgb_model, scaler_X, scaler_y)

            # Явная загрузка моделей и скейлеров после обучения
            xgb_model, scaler_X, scaler_y = self.xgb_manager.load_model_and_scalers()
            print("✅ XGBoost-модель и скейлеры загружены после обучения.")
        else:
            print("✅ XGBoost-модель загружена, обучение пропущено.")

        state_data, sentiment_scores = self.prepare_ppo_state_data(data_multi, scalers, lstm_models)

        ppo_trainer = PPOTrainer(
            symbol=self.symbol,
            state_data=state_data,
            sentiment_scores=sentiment_scores,
            lstm_models=lstm_models,
            cnn_model=cnn_models,
            cnn_scaler=None,
            xgb_model=xgb_model,
            xgb_scaler_X=scaler_X,
            xgb_scaler_y=scaler_y,
            model_manager=self.ppo_manager,
            episodes=100000
        )

        ppo_model = self.ppo_manager.load_model()

        if ppo_model:
            print("✅ PPO-модель найдена, начинаем дообучение...")
            agent, env = ppo_trainer.train(existing_model=ppo_model)
        else:
            print("🚀 Новое обучение PPO-агента...")
            agent, env = ppo_trainer.train()

        ppo_trainer.evaluate_agent(agent, env)
        print("✅ PPO-агент готов к использованию.")

    def prepare_ppo_state_data(self, data_multi, scalers, lstm_models):
        combined_data = []
        sentiment_scores = []

        min_length = min(len(df) for df in data_multi.values() if not df.empty)

        symbol_news = [self.symbol.split('/')[0]]
        sentiment_score = self.sentiment_analyzer.analyze_sentiment(symbol_news)

        fg_index, _ = FearGreedIndexFetcher.fetch_current_index()
        fg_scaled = fg_index / 100 if fg_index else 0.5

        for i in range(20, min_length):
            state_row = []
            for tf in self.timeframes:
                df_tf = data_multi[tf]
                if df_tf.empty:
                    continue

                scaler = scalers[tf]

                current_row = df_tf[self.features].iloc[i:i + 1]
                scaled_features = scaler.transform(current_row)

                lstm_input_df = df_tf[self.features].iloc[i - 20:i]
                lstm_input_scaled = scaler.transform(lstm_input_df)

                lstm_input = torch.tensor(lstm_input_scaled, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    lstm_pred = lstm_models[tf](lstm_input).item()

                state_row.extend(scaled_features.flatten().tolist())
                state_row.append(lstm_pred)

            state_row.extend([sentiment_score, fg_scaled])
            combined_data.append(state_row)
            sentiment_scores.append(sentiment_score)

        return np.array(combined_data), np.array(sentiment_scores)
