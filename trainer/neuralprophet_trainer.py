import numpy as np
from sklearn.metrics import mean_absolute_error

from model.neuralprophet_model import create_neuralprophet_model
from trainer.model_manager.neuralprophet_manager import NeuralProphetManager
from data.data_preparation import DataPreparation


class NeuralProphetTrainer:
    def __init__(self, symbol, features, timeframe, scaler_type='standard'):
        self.symbol = symbol.replace('/', '_')
        self.features = features
        self.timeframe = timeframe
        self.prep = DataPreparation(symbol, features, scaler_type=scaler_type)
        self.model_manager = NeuralProphetManager(self.symbol)
        self.scaler = self.prep.get_scaler()

    def train_model(self, df, freq='1h'):
        np_model = self.model_manager.load_model(self.timeframe)
        scaler = self.model_manager.load_scaler(self.timeframe)

        if np_model and scaler:
            print(f"📦 Модель NeuralProphet [{self.timeframe}] и scaler уже загружены.")
            return np_model, scaler, None

        # 🔥 Вот здесь убедись, что возвращается именно price_scaler!
        df_prepared, price_scaler, feature_scalers = self.prep.get_neuralprophet_data(df, fit_scaler=True)

        # 👇 Сохраняем только ОДИН раз и правильно!
        self.model_manager.save_scaler((price_scaler, feature_scalers), self.timeframe)

        model = create_neuralprophet_model()

        reserved_columns = ['ds', 'y', 'y_scaled']
        df_prepared = df_prepared.drop(columns=['y_scaled'], errors='ignore')

        for feature in df_prepared.columns:
            if feature not in reserved_columns:
                model.add_future_regressor(name=feature)

        metrics = model.fit(df_prepared, freq=freq)

        self.model_manager.save_model(model, self.timeframe)

        print(f"✅ NeuralProphet [{self.timeframe}] обучен и сохранён.")

        return model, (price_scaler, feature_scalers), metrics

    def predict(self, model, df_future):
        # БЕЗ МАСШТАБИРОВАНИЯ вообще! (если модель обучена на немасштабированных данных)
        forecast = model.predict(df_future)
        return forecast

    def evaluate_model(self, model, df_test):
        df_prepared, price_scaler, feature_scalers = self.prep.get_neuralprophet_data(df_test, fit_scaler=False)
        df_prepared = df_prepared.drop(columns=['y_scaled'], errors='ignore')

        forecast = model.predict(df_prepared)

        min_length = min(len(df_prepared['y']), len(forecast['yhat1']))
        y_true = df_prepared['y'].iloc[-min_length:].reset_index(drop=True)
        y_pred = forecast['yhat1'].iloc[-min_length:].reset_index(drop=True)

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0 or len(y_pred_clean) == 0:
            print(f"⚠️ После очистки NaN не осталось данных для оценки MAE [{self.timeframe}]")
            return np.nan

        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        print(f"MAE NeuralProphet [{self.timeframe}]:", mae)
        return mae


def train_all_timeframes(symbol: str, features: list[str], combined_data: dict, scaler_type='standard') -> dict:
    np_models = {}
    freq_map = {'15m': '15min', '1h': '1H', '4h': '4H', '1d': '1D'}

    for tf, df in combined_data.items():
        print(f"🚀 Запуск NeuralProphet тренера для таймфрейма [{tf}]...")
        trainer = NeuralProphetTrainer(symbol, features, timeframe=tf, scaler_type=scaler_type)
        freq = freq_map.get(tf, '1h')

        df_reset = df.reset_index()  # без rename!
        np_model, _, _ = trainer.train_model(df_reset, freq=freq)

        np_models[tf] = np_model

    return np_models

