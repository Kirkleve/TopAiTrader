import numpy as np
import pandas as pd
import torch

from data.neuralprophet_data_preparation import calculate_fibonacci_levels
from logger_config import setup_logger


class StateManager:
    def __init__(self, lstm_models, lstm_scalers, np_models, np_scalers, xgb_model, xgb_scaler_X, xgb_scaler_y,
                 historical_sentiment_scores, scaler_dict, historical_fg_scores, features, timeframes=None):
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']
        self.lstm_models = lstm_models
        self.lstm_scalers = lstm_scalers
        self.np_models = np_models
        self.np_scalers = np_scalers
        self.xgb_model = xgb_model
        self.xgb_scaler_X = xgb_scaler_X
        self.xgb_scaler_y = xgb_scaler_y
        self.historical_sentiment_scores = historical_sentiment_scores
        self.historical_fg_scores = historical_fg_scores
        self.features = features
        self.timeframes = timeframes
        self.scaler_dict = scaler_dict

    def prepare_single_state(self, unified_data, current_step, fib_window=100):
        state_row = []
        np_pred, xgb_pred, lstm_pred = None, None, None
        logger = setup_logger()

        required_lags = 48  # NeuralProphet обучен на окне 48 шагов
        np_pred_scaled = 0.0


        for tf in self.timeframes:
            df_tf = unified_data[tf]

            # 1. LSTM блок (без изменений, корректен)
            scaler_lstm = self.lstm_scalers[tf]
            current_row = df_tf[self.features].iloc[current_step:current_step + 1]
            scaled_features = scaler_lstm.transform(current_row)

            lstm_input_df = df_tf[self.features].iloc[max(0, current_step - 20):current_step]
            lstm_input_scaled = scaler_lstm.transform(lstm_input_df)
            lstm_input = torch.tensor(lstm_input_scaled, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                lstm_pred = self.lstm_models[tf](lstm_input).item()

            # 2. NeuralProphet блок (полностью корректен и безопасен)
            start_idx = current_step - required_lags
            if start_idx < 0:
                np_pred_scaled = df_tf['close'].iloc[current_step]
                logger.warning(
                    f"⚠️ Недостаточно данных ({current_step}/{required_lags}) для NeuralProphet [{tf}] на шаге {current_step}. Используем текущую цену.")
            else:
                np_input_df = pd.DataFrame({
                    'ds': df_tf.index[start_idx:current_step],
                    'y': df_tf['close'].iloc[start_idx:current_step].values
                })

                # ОБЯЗАТЕЛЬНО: добавляем признаки для масштабирования
                for feature in self.features:
                    if feature != 'close':
                        np_input_df[feature] = df_tf[feature].iloc[start_idx:current_step].values

                # Масштабируем цену и признаки
                price_scaler, feature_scalers = self.np_scalers[tf]
                np_input_df['y'] = price_scaler.transform(np_input_df[['y']].values)

                for feature in self.features:
                    if feature != 'close':
                        scaler = feature_scalers.get(feature)
                        if scaler:
                            np_input_df[feature] = scaler.transform(np_input_df[[feature]].values)
                        else:
                            raise ValueError(f"Scaler для признака '{feature}' не найден!")

                # Генерируем уровни Фибоначчи
                fib_cols = ['fib_23_6', 'fib_38_2', 'fib_50', 'fib_61_8', 'fib_78_6']
                for col in fib_cols:
                    np_input_df[col] = np.nan

                fib_window = min(fib_window, len(np_input_df))
                if len(np_input_df) >= fib_window:
                    for i in range(fib_window, len(np_input_df)):
                        min_price = np_input_df['y'].iloc[i - fib_window:i].min()
                        max_price = np_input_df['y'].iloc[i - fib_window:i].max()
                        fib_levels = calculate_fibonacci_levels(min_price, max_price)
                        for col in fib_cols:
                            np_input_df.at[i, col] = fib_levels[col]

                np_input_df[fib_cols] = np_input_df[fib_cols].fillna(method='bfill').fillna(method='ffill')
                np_input_df.dropna(inplace=True)
                np_input_df.reset_index(drop=True, inplace=True)

                # Подготовка будущего периода
                future_period = np_input_df['ds'].iloc[-1] + pd.Timedelta(self._get_timedelta_for_tf(tf))
                future_df = pd.DataFrame({'ds': [future_period], 'y': [None]})

                for feature in self.features + fib_cols:
                    if feature != 'close':
                        future_df[feature] = np_input_df[feature].iloc[-1]

                full_input_df = pd.concat([np_input_df, future_df])

                # Прогноз NeuralProphet
                try:
                    forecast = self.np_models[tf].predict(full_input_df)
                    np_pred_scaled = forecast['yhat1'].iloc[-1]
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка прогноза NeuralProphet ({tf}): {e}")
                    np_pred_scaled = np_input_df['y'].iloc[-1]

            state_row.extend(scaled_features.flatten().tolist())
            state_row.extend([lstm_pred, np_pred_scaled])

            # 3. XGB блок (без изменений, он корректен)
            if tf == '1h':
                xgb_input_scaled = self.xgb_scaler_X.transform(current_row)
                xgb_pred_scaled = self.xgb_model.predict(xgb_input_scaled).item()
                xgb_pred = self.xgb_scaler_y.inverse_transform([[xgb_pred_scaled]])[0, 0]
                state_row.append(xgb_pred)

        # 4. Sentiment и Fear & Greed (без изменений, корректно)
        sentiment_score = self.historical_sentiment_scores[current_step] if current_step < len(
            self.historical_sentiment_scores) else 0
        fg_score = self.historical_fg_scores[current_step] if current_step < len(self.historical_fg_scores) else 0.5

        state_row.extend([sentiment_score, fg_score])

        # Итоговый лог
        logger.info("\n🔍 Проверка данных перед PPO:")
        logger.debug(f"state_row: {state_row}")
        logger.info(f"Sentiment: {sentiment_score}, Fear & Greed: {fg_score}")
        logger.info(f"LSTM pred: {lstm_pred}, NeuralProphet pred: {np_pred_scaled}, XGB pred: {xgb_pred}")

        return np.nan_to_num(state_row, nan=0.0, posinf=0.0, neginf=0.0)

    def create_state(self, unified_data, current_step):
        return self.prepare_single_state(unified_data, current_step)

    def _get_timedelta_for_tf(self, timeframe):
        return {
            '15m': '15min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }.get(timeframe, '1h')
