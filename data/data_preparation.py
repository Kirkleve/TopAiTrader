import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch

from data.neuralprophet_data_preparation import prepare_neuralprophet_data


class DataPreparation:
    def __init__(self, symbol, features, scaler_type='standard'):
        self.symbol = symbol
        self.features = features
        self.scaler_type = scaler_type
        self.scalers = {}

    def get_scaler(self):
        if self.scaler_type == 'robust':
            return RobustScaler()
        return StandardScaler()

    def load_and_prepare_data(self, fetcher, timeframes):
        combined_data = fetcher.fetch_historical_data_multi_timeframe(self.symbol, timeframes)
        prepared_data = {}

        for tf, df in combined_data.items():
            df_cleaned = df[self.features].replace([np.inf, -np.inf], np.nan).dropna()
            prepared_data[tf] = df_cleaned

        return prepared_data

    def resample_data(self, data, original_tf, target_tf='1h'):
        if original_tf == target_tf:
            return data

        if pd.Timedelta(original_tf) < pd.Timedelta(target_tf):
            df_resampled = data.resample(target_tf).agg({
                'close': 'last', 'volume': 'sum', 'high': 'max', 'low': 'min',
                'rsi': 'last', 'ema': 'last', 'adx': 'last', 'atr': 'mean',
                'cci': 'last', 'williams_r': 'last', 'momentum': 'mean',
                'mfi': 'last', 'mass_index': 'mean'
            }).dropna()
        else:
            df_resampled = data.reindex(
                pd.date_range(data.index.min(), data.index.max(), freq=target_tf),
                method='ffill'
            ).dropna()
        return df_resampled

    def get_lstm_data(self, df, seq_length=20):
        df = df.replace([np.inf, -np.inf], np.nan).dropna()  # очистка данных перед LSTM
        scaler = self.get_scaler()
        data_scaled = scaler.fit_transform(df[self.features])
        X, y = [], []

        for i in range(seq_length, len(data_scaled)):
            X.append(data_scaled[i - seq_length:i])
            y.append(data_scaled[i, 0])  # close

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        return X_tensor, y_tensor, scaler

    def get_neuralprophet_data(self, df, fit_scaler=False):
        # Убираем всё лишнее и оставляем только вызов правильной функции!
        df_np, price_scaler, feature_scalers = prepare_neuralprophet_data(df, features=self.features)

        # Возвращаем данные БЕЗ дополнительного масштабирования!
        return df_np, price_scaler, feature_scalers

    def get_xgboost_data(self, df):
        df = df.replace([np.inf, -np.inf], np.nan).dropna()  # очистка данных перед XGBoost
        scaler_X, scaler_y = StandardScaler(), StandardScaler()

        X = df[self.features].iloc[:-1].values
        y = df['close'].shift(-1).dropna().values.reshape(-1, 1)

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y).flatten()

        return X_scaled, y_scaled, scaler_X, scaler_y

    def get_ppo_state_data(self, df):
        scaler = RobustScaler()
        state_data_scaled = scaler.fit_transform(df[self.features].values)
        state_data_scaled = np.nan_to_num(state_data_scaled)
        return state_data_scaled, scaler