import numpy as np
import pandas as pd
import torch
from data.neuralprophet_data_preparation import prepare_neuralprophet_data


def precompute_model_predictions(env, lookahead=50):
    env.cached_lstm_preds = {}
    env.cached_np_preds = {}
    freq_map = {'15m': '15min', '1h': '1H', '4h': '4H', '1d': '1D'}

    for future_step in range(env.current_step, env.current_step + lookahead):
        if future_step >= len(env.data) or future_step < 48:
            continue

        env.cached_lstm_preds[future_step] = [
            model(torch.tensor(env.data[future_step - 20:future_step, :11], dtype=torch.float32).unsqueeze(0)).item()
            for model in env.lstm_models.values()
        ]

        np_preds_step = []
        required_lags = 48

        for tf, np_model in env.np_models.items():
            original_df = env.df_original_dict[tf]

            if future_step < required_lags:
                continue

            df_slice = original_df.iloc[future_step - required_lags:future_step].copy()

            if len(df_slice) < required_lags:
                continue

            input_df, price_scaler, feature_scalers = prepare_neuralprophet_data(
                df=df_slice,
                features=env.feature_names,
                fib_window=100
            )

            input_df['ds'] = pd.to_datetime(input_df['ds'])
            input_df = input_df.set_index('ds').asfreq(freq_map[tf]).fillna(method='ffill').reset_index()

            # Масштабируем признаки (кроме 'y')
            for feature, scaler in feature_scalers.items():
                input_df[feature] = scaler.transform(input_df[[feature]])

            numeric_cols = input_df.select_dtypes(include=[np.number]).columns
            if input_df[numeric_cols].isna().any().any() or np.isinf(input_df[numeric_cols].values).any():
                raise ValueError(f"⚠️ NaN или Inf в данных NeuralProphet! Таймфрейм: {tf}")

            print("\n🟡 Таймфрейм:", tf)
            print("🟡 Input DF перед прогнозом:\n", input_df.tail())
            print("🟡 Частота (freq):", pd.infer_freq(input_df['ds']))
            print("🟡 Количество строк данных:", len(input_df))

            forecast = np_model.predict(input_df)

            if forecast is None or forecast.empty or 'yhat1' not in forecast.columns:
                raise ValueError(f"⚠️ NeuralProphet не смог предсказать! Таймфрейм: {tf}")

            np_preds_step.append(forecast['yhat1'].iloc[-1])

        env.cached_np_preds[future_step] = np_preds_step


# Исправлено закрытие противоположной позиции перед открытием новой
def calculate_reward(env, action, current_price, atr, adx):
    reward = 0.0

    volatility_adj = np.clip(atr / current_price, 0.01, 0.1) if current_price != 0 else 0.01

    risk_percentage = 0.01 + (adx / 1000)  # Плавный градиент риска
    risk_amount = env.balance * risk_percentage

    # Закрытие позиции
    if action == 0 and env.position and env.position_price > 0:
        profit = (current_price - env.position_price) if env.position == 'long' else (
                    env.position_price - current_price)
        profit_pct = (profit / env.position_price) * 100
        reward += profit_pct * (1 + adx / 50) * volatility_adj

        holding_time = env.current_step - env.position_open_step
        time_penalty = holding_time / 1000
        reward -= time_penalty if profit <= 0 else -time_penalty

        if profit > 0:
            env.consecutive_profitable_trades += 1
            streak_bonus = 0.5 + (env.consecutive_profitable_trades - 1) * 0.2
            reward += streak_bonus
            env.consecutive_losses = 0
        else:
            env.consecutive_losses += 1
            loss_streak_penalty = 0.5 + (env.consecutive_losses - 1) * 0.2
            reward -= loss_streak_penalty
            env.consecutive_profitable_trades = 0

        env.balance += profit - env.trading_fee
        env.pnl += profit
        env.position = env.position_price = env.position_open_step = None
        env.pyramid_count = 0

    elif action == 0 and not env.position:
        reward -= 0.05

    # Открытие новой или противоположной позиции
    if action in [1, 2]:
        requested_position = 'long' if action == 1 else 'short'
        if env.position is None:
            env.position = requested_position
            env.position_price = current_price
            env.position_open_step = env.current_step
            env.position_size = risk_amount / atr if atr > 0 else 0.0
            reward -= env.trading_fee / 10
        elif env.position != requested_position and env.position_price is not None:
            profit = (
                (current_price - env.position_price)
                if env.position == 'long'
                else (env.position_price - current_price)
            )
            env.balance += profit - env.trading_fee
            env.pnl += profit
            reward -= 0.1
            env.position = requested_position
            env.position_price = current_price
            env.position_open_step = env.current_step
            env.position_size = risk_amount / atr if atr > 0 else 0.0
            env.consecutive_profitable_trades = 0
            env.consecutive_losses = 0
            env.pyramid_count = 0
        else:
            reward -= 0.1

    # Улучшенный пирамидинг
    if env.position and env.pyramid_count < 3 and adx > 35:
        current_profit = (current_price - env.position_price) if env.position == 'long' else (
                    env.position_price - current_price)
        max_risk_percentage = 0.05
        available_risk = env.balance * max_risk_percentage

        if current_profit > atr and available_risk > env.position_size * current_price * 0.1:
            additional_size = risk_amount / atr
            env.position_price = (env.position_price * env.position_size + current_price * additional_size) / (
                        env.position_size + additional_size)
            env.position_size += additional_size
            env.pyramid_count += 1
            reward += 0.2 + (adx - 35) / 100

    # Адаптивный трейлинг стоп и TP
    if env.position and env.position_price > 0:
        current_profit = (current_price - env.position_price) if env.position == 'long' else (
                    env.position_price - current_price)
        profit_multiplier = max(1.0, min(3.0, abs(current_profit / atr)))
        sl_multiplier = (1.2 if adx > 50 else 1.5 if adx > 30 else 2.0) / profit_multiplier
        tp_multiplier = 3 if adx > 50 else (3.5 if adx > 30 else 4)

        trailing_sl = (current_price - atr * sl_multiplier) if env.position == 'long' else (
                    current_price + atr * sl_multiplier)
        tp_price = (env.position_price + atr * tp_multiplier) if env.position == 'long' else (
                    env.position_price - atr * tp_multiplier)

        if (env.position == 'long' and current_price <= trailing_sl) or (
                env.position == 'short' and current_price >= trailing_sl):
            profit = (current_price - env.position_price) if env.position == 'long' else (
                        env.position_price - current_price)
            profit_pct = abs(profit) / env.position_price * 100
            reward += profit_pct * volatility_adj * (1 if profit > 0 else -1)

            env.balance += profit * env.position_size
            env.pnl += profit
            env.position = env.position_price = env.position_open_step = None
            env.pyramid_count = 0

        elif (env.position == 'long' and current_price >= tp_price) or (
                env.position == 'short' and current_price <= tp_price):
            profit = (current_price - env.position_price) if env.position == 'long' else (
                        env.position_price - current_price)
            profit_pct = abs(profit) / env.position_price * 100
            reward += profit_pct * volatility_adj + 0.5

            env.balance += profit * env.position_size
            env.pnl += profit
            env.position = env.position_price = env.position_open_step = None
            env.pyramid_count = 0

    balance_ratio = env.balance / env.initial_balance
    reward += (balance_ratio - 1) * 0.5 if balance_ratio > 1 else (balance_ratio - 1)

    reward = np.clip(reward, -20, 20)

    if env.balance < env.initial_balance * 0.30:
        reward -= 5

    if np.isnan(reward) or np.isinf(reward):
        print(f"🚨 reward превратился в NaN/inf. Присваиваем 0.")
        reward = 0.0

    return reward


def get_observation(env):
    if env.current_step < len(env.data):
        obs = env.data[env.current_step]
    else:
        obs = np.zeros(env.data.shape[1])  # если за пределами — безопасно

    if np.isnan(obs).any() or np.isinf(obs).any():
        print(f"🚨 NaN или Inf в наблюдениях на шаге {env.current_step}. Заменяю на нули.")
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    return obs  # уже нормализованные state_data




