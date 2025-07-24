import os
import pickle
import joblib
from neuralprophet import NeuralProphet

from logger_config import setup_logger

logger = setup_logger()

class NeuralProphetManager:
    def __init__(self, symbol: str):
        self.symbol = symbol.replace('/', '_')
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_dir = os.path.join(self.base_dir, "models", self.symbol, "neuralprophet")
        self.scaler_dir = os.path.join(self.model_dir, 'scalers')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)

    def get_model_filepath(self, timeframe: str):
        return os.path.join(self.model_dir, f'{self.symbol}_{timeframe}_np_model.pkl')

    def get_scaler_filepath(self, timeframe: str):
        return os.path.join(self.scaler_dir, f'{self.symbol}_{timeframe}_scaler.pkl')

    def save_model(self, model: NeuralProphet, timeframe: str):
        filepath = self.get_model_filepath(timeframe)
        joblib.dump(model, filepath)
        logger.info(f'💾 Модель NeuralProphet [{timeframe}] успешно сохранена: {filepath}')

    def load_model(self, timeframe: str):
        filepath = self.get_model_filepath(timeframe)
        if os.path.exists(filepath):
            model = joblib.load(filepath)
            logger.info(f'✅ Модель NeuralProphet [{timeframe}] загружена: {filepath}')
            return model
        logger.info(f'⚠️ Модель NeuralProphet не найдена по пути: {filepath}')
        return None

    def save_scaler(self, scaler, timeframe: str):
        filepath = self.get_scaler_filepath(timeframe)
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f'💾 Scaler NeuralProphet [{timeframe}] сохранён: {filepath}')

    def load_scaler(self, timeframe: str):
        filepath = self.get_scaler_filepath(timeframe)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f'✅ Scaler NeuralProphet [{timeframe}] загружен: {filepath}')
            return scaler
        logger.info(f'⚠️ Scaler не найден по пути: {filepath}')
        return None
