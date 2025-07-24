import pandas as pd
from config import BYBIT_API_KEY, BYBIT_API_SECRET
from data.fetch_data import CryptoDataFetcherBybit
from data.data_preparation import DataPreparation
from data.fear_and_greed import FearGreedIndexFetcher
from data.news_fetcher import NewsFetcher
from trainer.lstm_trainer import train_lstm_models_for_timeframes
from trainer.neuralprophet_trainer import NeuralProphetTrainer
from trainer.model_manager.state_manager import StateManager
from trainer.xgboost_trainer import train_xgboost_model
from trainer.ppo_trainer import PPOTrainer
from trainer.model_manager.ppo_manager import PPOModelManager
from model.news_sentiment import SentimentAnalyzer


class UniversalTrainer:
    def __init__(self, symbol, device='cpu'):
        self.symbol = symbol
        self.features = [
            'close', 'rsi', 'ema', 'adx', 'atr', 'volume',
            'cci', 'williams_r', 'momentum', 'mfi', 'mass_index'
        ]
        self.device = device
        self.fetcher = CryptoDataFetcherBybit(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
        self.sentiment_analyzer = SentimentAnalyzer(model_type='finbert')
        self.news_fetcher = NewsFetcher()
        self.prep = DataPreparation(symbol, self.features)
        self.timeframes = ['15m', '1h', '4h', '1d']
        self.ppo_manager = PPOModelManager(symbol)

    def run_training(self):
        combined_data = self.prep.load_and_prepare_data(self.fetcher, self.timeframes)

        # NeuralProphet с возвратом скейлеров
        print("Начало обучения NeuralProphet:")
        np_models, np_scalers = {}, {}
        for tf, df in combined_data.items():
            trainer = NeuralProphetTrainer(self.symbol, self.features, timeframe=tf)
            np_model, scaler, _ = trainer.train_model(df.reset_index())
            trainer.evaluate_model(np_model, df.reset_index())
            np_models[tf] = np_model
            np_scalers[tf] = scaler

        # LSTM
        print("Начало обучения LSTM:")
        lstm_models, lstm_scalers = train_lstm_models_for_timeframes(
            symbol=self.symbol,
            features=self.features,
            combined_data=combined_data,
            device=self.device
        )

        # XGBoost
        print("Начало обучения XGBoost:")
        xgb_predictor = train_xgboost_model(
            symbol=self.symbol,
            features=self.features,
            df_1h=combined_data["1h"],
            optimize=True,
            trials=50
        )

        today = pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()
        unique_dates = pd.to_datetime(combined_data['1h'].index.date).unique()
        filtered_dates = [date for date in unique_dates if date <= today]

        historical_sentiment_scores = self.sentiment_analyzer.get_historical_sentiment_scores(
            dates=filtered_dates,
            fetch_news_for_date=self.news_fetcher.fetch_news_for_date
        )

        historical_fg_scores = FearGreedIndexFetcher.get_historical_fg_scores(dates=filtered_dates)

        sentiment_scores = {
            'historical': historical_sentiment_scores,
            'fear_greed': historical_fg_scores
        }

        # State Manager создание
        state_manager = StateManager(
            lstm_models=lstm_models,
            lstm_scalers=lstm_scalers,
            np_models=np_models,
            np_scalers=np_scalers,
            xgb_model=xgb_predictor.model,
            xgb_scaler_X=xgb_predictor.scaler_X,
            xgb_scaler_y=xgb_predictor.scaler_y,
            historical_sentiment_scores=historical_sentiment_scores,
            scaler_dict=self.prep.scalers,
            historical_fg_scores=historical_fg_scores,
            features=self.features,
            timeframes=self.timeframes
        )

        # PPO обучение (передача готового StateManager)
        if self.ppo_manager.model_exists():
            print("✅ PPO-модель уже обучена. Пропускаю обучение PPO.")
            agent = self.ppo_manager.load_model()
        else:
            print("🚀 Начинается обучение PPO-модели...")

            ppo_trainer = PPOTrainer(
                symbol=self.symbol,
                combined_data=combined_data,
                dates=filtered_dates,
                state_manager=state_manager,
                model_manager=self.ppo_manager,
                sentiment_scores=sentiment_scores,
                sentiment_model="finbert"
            )

            agent, env = ppo_trainer.train()
            self.ppo_manager.save_model(agent)
            ppo_trainer.evaluate_agent(agent, env)

        print("🎉 Полный цикл обучения успешно завершён!")

