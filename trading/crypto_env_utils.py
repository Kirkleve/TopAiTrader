import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def precompute_model_predictions(env, lookahead=50):
    env.cached_lstm_preds = {}
    env.cached_cnn_preds = {}

    for future_step in range(env.current_step, env.current_step + lookahead):
        if future_step >= len(env.data):
            break

        # LSTM предсказания
        env.cached_lstm_preds[future_step] = [
            model(torch.tensor(env.data[max(0, future_step - 20):future_step, :11],
                               dtype=torch.float32).unsqueeze(0)).item()
            for model in env.lstm_models.values()
        ]

        # CNN кэширование
        cnn_preds_step = []
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        for tf, cnn_model in env.cnn_models.items():
            image_path = os.path.join("models", env.symbol.replace('/', '_'), "cnn",
                                      f"{env.symbol.replace('/', '_')}_{tf}", f"{future_step}.png")
            try:
                image = Image.open(image_path).convert('RGB')
            except FileNotFoundError:
                image = Image.new('RGB', (128, 128), color='black')

            cnn_input = transform(image).unsqueeze(0)
            with torch.no_grad():
                cnn_preds_step.append(cnn_model(cnn_input).item())
        env.cached_cnn_preds[future_step] = cnn_preds_step


def calculate_reward(env, action, current_price, atr, adx):
    reward = 0.0

    # Защита от деления на ноль
    volatility_adj = np.clip(atr / current_price, 0.01, 0.1) if current_price != 0 else 0.01

    # Рассчитываем риск на сделку
    risk_percentage = 0.015  # Базовый риск 1.5% от депозита
    if adx > 30:  # Если сильный тренд, увеличиваем риск
        risk_percentage = 0.02  # Риск 2% от депозита

    # Рассчитываем сумму, которую агент готов потерять
    loss_in_balance = env.balance * risk_percentage

    # Закрытие позиции
    if action == 0 and env.position and env.position_price > 0:
        profit = (current_price - env.position_price) if env.position == 'long' else (
            env.position_price - current_price)

        # Увеличиваем вес прибыли
        profit_pct = (profit / env.position_price) * 100
        reward = profit_pct * (1 + adx / 50) * volatility_adj  # Учитываем тренд и волатильность

        # Если прибыльная сделка, то добавляем бонус
        if profit > 0:
            reward += 0.1  # Небольшой бонус за прибыльную сделку

        # Штраф за большие потери
        if profit < 0:
            loss_pct = abs(profit) / env.position_price * 100
            reward -= loss_pct * 0.5  # Чем больше убытки, тем сильнее штраф

        env.balance += profit - env.trading_fee
        env.pnl += profit
        env.position = env.position_price = env.position_open_step = None

    # Штраф за бездействие
    elif action == 0 and not env.position:
        reward = -0.05  # Штраф за бездействие

    # Открытие позиции
    if action in [1, 2] and not env.position:
        env.position = 'long' if action == 1 else 'short'
        env.position_price = current_price
        env.position_open_step = env.current_step
        reward -= env.trading_fee / 10  # Штраф за открытие позиции, чтобы избежать переторговки

        # Рассчитываем количество единиц (например, количество криптовалюты), которое можно купить/продать
        position_size = loss_in_balance / atr  # Можно использовать ATR для оценки волатильности сделки
        env.position_size = position_size  # Добавляем это в состояние агента

    elif action in [1, 2] and env.position:
        reward -= 0.1  # Штраф за повторное открытие позиции

    # Условия SL и TP с ADX и ATR
    if env.position and env.position_price > 0:
        sl_price = (env.position_price - atr * env.sl_multiplier) if env.position == 'long' else (
                   env.position_price + atr * env.sl_multiplier)

        tp_multiplier = 4 if adx > 30 else 2 if adx < 15 else env.tp_multiplier
        tp_price = (env.position_price + atr * tp_multiplier) if env.position == 'long' else (
                   env.position_price - atr * tp_multiplier)

        # Убыток при достижении SL
        if (env.position == 'long' and current_price <= sl_price) or (
                env.position == 'short' and current_price >= sl_price):
            loss_pct = abs(current_price - env.position_price) / env.position_price * 100
            reward -= loss_pct * volatility_adj  # Большой штраф за убыточную сделку
            env.position = env.position_price = env.position_open_step = None

        # Профит при достижении TP
        elif (env.position == 'long' and current_price >= tp_price) or (
                env.position == 'short' and current_price <= tp_price):
            profit_pct = abs(current_price - env.position_price) / env.position_price * 100
            reward += profit_pct * volatility_adj  # Большее вознаграждение за прибыльную сделку
            reward += 0.5  # Бонус за достижение TP
            env.position = env.position_price = env.position_open_step = None

    # Бонус за следование за трендом (если ADX высокий)
    if adx > 30:
        reward += 0.2  # Бонус за следование тренду

    # Штраф за слишком долгие позиции (если удерживаем больше 50 шагов)
    if env.position_open_step and (env.current_step - env.position_open_step) > 50:
        reward -= 0.1  # Увеличиваем штраф за длительное удержание позиции

    # Штраф за большие убытки (например, потеря 25% баланса)
    if env.balance < env.initial_balance * 0.50:
        reward -= 20  # Штраф за потерю более 50% баланса

    # Бонус за эффективное распределение капитала (избегать концентрации)
    balance_used_ratio = abs(env.balance - env.initial_balance) / env.initial_balance
    if balance_used_ratio < 0.25:  # Если агент использует менее 25% баланса
        reward += 0.2  # Бонус за безопасное распределение

    # Применяем окончательную обрезку вознаграждения
    reward = np.clip(reward, -500, 500)

    return reward



def get_observation(env):
    current_features = np.nan_to_num(env.data[env.current_step][1:12], nan=0.0)
    lstm_preds = env.cached_lstm_preds.get(env.current_step, [0.0] * len(env.lstm_models))
    cnn_preds = env.cached_cnn_preds.get(env.current_step, [0.0] * len(env.cnn_models))

    try:
        xgb_input_scaled = env.scaler_X.transform([current_features])
        xgb_pred_scaled = env.xgb_model.predict(xgb_input_scaled).item()
        xgb_pred = env.scaler_y.inverse_transform([[xgb_pred_scaled]])[0, 0]
    except Exception as e:
        print(f"🚨 Ошибка XGB-прогноза на шаге {env.current_step}: {e}")
        xgb_pred = 0.0

    sentiment_score = (
        float(env.sentiment_scores[env.current_step])
        if env.sentiment_scores is not None and len(env.sentiment_scores) > env.current_step
        else 0.0
    )

    extra_features = [
        sentiment_score,
        float(env.position_open_step or 0),
        float(env.balance),
        float(env.pnl),
        float(abs(env.pnl / env.initial_balance) > 0.05),
        getattr(env, 'fear_greed_scaled', 0.5),
        env.trading_fee,
        env.sl_multiplier,
        env.tp_multiplier,
        xgb_pred
    ]

    obs = np.concatenate(
        [current_features, lstm_preds, cnn_preds, extra_features]
    ).astype(np.float32)

    # ✅ ОБЯЗАТЕЛЬНАЯ нормализация всех признаков
    obs_normalized = (obs - obs.mean()) / (obs.std() + 1e-8)

    return obs_normalized
