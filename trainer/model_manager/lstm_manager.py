import os
import torch
import pickle
from sklearn.preprocessing import StandardScaler

from logger_config import setup_logger
from model.lstm_price_predictor import LSTMPricePredictor

logger = setup_logger()


class LSTMModelManager:
    def __init__(self, symbol, features):
        self.symbol = symbol.replace('/', '_')
        self.features = features
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_dir = os.path.join(self.base_dir, "models", self.symbol, "lstm")
        self.scaler_dir = os.path.join(self.model_dir, 'scalers')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)

    def save_model(self, model, timeframe):
        model_path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}_lstm.pth")
        torch.save(model.state_dict(), model_path)

    def load_model(self, timeframe):
        model_path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}_lstm.pth")
        if os.path.exists(model_path):
            model = LSTMPricePredictor(input_size=len(self.features))
            model.load_state_dict(torch.load(model_path))
            model.eval()
            logger.info(f"✅ LSTM-модель загружена: {model_path}")
            return model
        else:
            logger.info(f"⚠️ LSTM-модель для [{timeframe}] не найдена: {model_path}")
            return None

    def save_scaler(self, scaler, timeframe):
        scaler_path = os.path.join(
            self.scaler_dir, f'scaler_{self.symbol}_{timeframe}.pkl'
        )
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    def load_scaler(self, timeframe):
        scaler_path = os.path.join(
            self.scaler_dir, f'scaler_{self.symbol}_{timeframe}.pkl'
        )
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"✅ Scaler загружен: {scaler_path}")
        else:
            scaler = StandardScaler()
            logger.info(f"⚠️ Scaler не найден, создан новый StandardScaler для {self.symbol} [{timeframe}]")
        return scaler