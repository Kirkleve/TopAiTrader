from typing import Literal, cast
import torch
from trainer.model_manager.state_manager import StateManager
from strategy.adaptive_strategy import adapt_risk_params
from trading.binance_trader import BinanceTrader


def handle_autotrade(
    bot, chat_id, symbol, unified_data, current_step,
    historical_sentiment_scores, historical_fg_scores,
    features, balance
):
    bot.send_message(chat_id, f"🤖 Запускаю автотрейдинг для {symbol}")

    trader = BinanceTrader()

    # Загрузка всех моделей и PPO агента
    lstm_models = bot.lstm_models
    lstm_scalers = bot.lstm_scalers
    np_models = bot.np_models
    xgb_model, xgb_scaler_X, xgb_scaler_y = bot.xgb_model, bot.xgb_scaler_X, bot.xgb_scaler_y
    ppo_agent = bot.ppo_agent

    # Подготовка состояния через StateManager
    state_manager = StateManager(
        lstm_models, lstm_scalers, np_models,
        xgb_model, xgb_scaler_X, xgb_scaler_y,
        historical_sentiment_scores,
        historical_fg_scores,
        features
    )

    # Текущее состояние
    state = state_manager.create_state(unified_data, current_step)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # Прогноз sentiment для текущего состояния
    sentiment_score = historical_sentiment_scores[-1]

    # PPO агент принимает решение
    action, _ = ppo_agent.predict(state_tensor, deterministic=True)

    # Адаптация параметров риска и sentiment threshold
    risk_params = adapt_risk_params()

    # Проверка sentiment threshold после PPO решения
    if action != 0 and abs(sentiment_score) < risk_params["sentiment_threshold"]:
        bot.send_message(chat_id, f"⏳ Сделка отменена по sentiment ({sentiment_score:.2f} < {risk_params['sentiment_threshold']:.2f}).")
        return

    if action == 0:
        bot.send_message(chat_id, f"⏳ PPO-агент решил не открывать сделку ({symbol}).")
        return

    # Получаем ATR для текущего символа
    atr = unified_data[current_step, features.index('atr')]

    # Проверка и закрытие предыдущей позиции (если открыта)
    position = trader.get_position(symbol)
    if position:
        trader.close_all_positions(symbol)
        bot.send_message(chat_id, f"🔄 Закрыта предыдущая позиция по {symbol}.")

    # Открытие позиции с адаптивными SL и TP
    side: Literal['buy', 'sell'] = cast(Literal['buy', 'sell'], 'buy' if action == 1 else 'sell')

    trader.create_order_with_sl_tp(
        symbol=symbol,
        side=side,
        balance=balance,
        risk_percent=risk_params["risk_percent"],
        atr=atr,
        sl_multiplier=risk_params["sl_multiplier"],
        tp_multiplier=risk_params["tp_multiplier"],
        adaptive=True
    )

    bot.send_message(
        chat_id,
        f"🚀 Открыта позиция {side.upper()} по {symbol}\n\n"
        f"🎯 Параметры сделки:\n"
        f"• Риск на сделку: {risk_params['risk_percent']*100:.2f}% от баланса\n"
        f"• Sentiment-порог: {risk_params['sentiment_threshold']:.2f}\n"
        f"• TP multiplier: {risk_params['tp_multiplier']}\n"
        f"• SL multiplier: {risk_params['sl_multiplier']}"
    )

    bot.send_message(chat_id, "✅ Автотрейдинг успешно завершён!")
