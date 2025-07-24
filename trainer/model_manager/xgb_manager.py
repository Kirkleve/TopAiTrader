import os
import pickle
import xgboost as xgb
from logger_config import setup_logger

logger = setup_logger()


class XGBModelManager:
    def __init__(self, symbol):
        self.symbol = symbol.replace('/', '_')
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_dir = os.path.join(self.base_dir, "models", self.symbol, "xgb")
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model_and_scalers(self, model, scaler_X, scaler_y):
        # сохраняем XGB-модель
        model_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_model.json")
        model.save_model(model_path)
        logger.info(f"✅ XGB-модель сохранена: {model_path}")

        # сохраняем scaler_X
        scaler_X_path = os.path.join(self.model_dir, f"{self.symbol}_scaler_X.pkl")
        with open(scaler_X_path, 'wb') as f:
            pickle.dump(scaler_X, f)
        logger.info(f"✅ Scaler_X сохранён: {scaler_X_path}")

        # сохраняем scaler_y
        scaler_y_path = os.path.join(self.model_dir, f"{self.symbol}_scaler_y.pkl")
        with open(scaler_y_path, 'wb') as f:
            pickle.dump(scaler_y, f)
        logger.info(f"✅ Scaler_y сохранён: {scaler_y_path}")

    def load_model_and_scalers(self):
        model_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_model.json")
        scaler_X_path = os.path.join(self.model_dir, f"{self.symbol}_scaler_X.pkl")
        scaler_y_path = os.path.join(self.model_dir, f"{self.symbol}_scaler_y.pkl")

        model = scaler_X = scaler_y = None

        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            logger.info(f"✅ XGB-модель загружена: {model_path}")
        else:
            logger.info(f"⚠️ XGB-модель не найдена: {model_path}")

        if os.path.exists(scaler_X_path):
            with open(scaler_X_path, 'rb') as f:
                scaler_X = pickle.load(f)
            logger.info(f"✅ Scaler_X загружен: {scaler_X_path}")
        else:
            logger.info(f"⚠️ Scaler_X не найден: {scaler_X_path}")

        if os.path.exists(scaler_y_path):
            with open(scaler_y_path, 'rb') as f:
                scaler_y = pickle.load(f)
            logger.info(f"✅ Scaler_y загружен: {scaler_y_path}")
        else:
            logger.info(f"⚠️ Scaler_y не найден: {scaler_y_path}")

        return model, scaler_X, scaler_y