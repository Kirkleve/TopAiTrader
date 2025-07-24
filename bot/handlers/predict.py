from datetime import datetime

import numpy as np
import pandas as pd

from trainer.model_manager.state_manager import StateManager
from data.atr import get_combined_atr


def handle_predict(bot, message, symbol='BTC/USDT', timeframe='1h'):
    chat_id = message.chat.id

    # Отправляем пользователю сообщение об ожидании анализа
    waiting_message = bot.bot.send_message(chat_id, "⏳ Пожалуйста, подождите немного — идёт подробный анализ...")

    combined_data = bot.data_fetcher.fetch_historical_data_multi_timeframe(symbol)

    if timeframe not in combined_data or combined_data[timeframe].empty:
        bot.bot.edit_message_text(f"⚠️ Нет данных для {symbol} на таймфрейме {timeframe}.", chat_id, waiting_message.message_id)
        return

    current_step = len(combined_data[timeframe]) - 1

    lstm_models = {tf: bot.lstm_manager.load_model(tf) for tf in ['15m', '1h', '4h', '1d']}
    lstm_scalers = {tf: bot.lstm_manager.load_scaler(tf) for tf in ['15m', '1h', '4h', '1d']}
    np_models = {tf: bot.np_manager.load_model(tf) for tf in ['15m', '1h', '4h', '1d']}
    xgb_model, xgb_scaler_X, xgb_scaler_y = bot.xgb_manager.load_model_and_scalers()
    ppo_agent = bot.ppo_manager.load_model()

    # === Проверка корректности загруженных скейлеров ===
    print("\n🔍 Проверка загруженных скейлеров:")

    # LSTM скейлеры
    for tf, scaler in lstm_scalers.items():
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'var_'):
            print(f"✅ LSTM scaler [{tf}]: mean={scaler.mean_}, var={scaler.var_}")
        else:
            print(f"❌ LSTM scaler [{tf}] пустой или некорректный!")

    # XGB скейлеры
    if hasattr(xgb_scaler_X, 'mean_') and hasattr(xgb_scaler_X, 'var_'):
        print(f"✅ XGB scaler_X: mean={xgb_scaler_X.mean_}, var={xgb_scaler_X.var_}")
    else:
        print("❌ XGB scaler_X пустой или некорректный!")

    if hasattr(xgb_scaler_y, 'mean_') and hasattr(xgb_scaler_y, 'var_'):
        print(f"✅ XGB scaler_y: mean={xgb_scaler_y.mean_}, var={xgb_scaler_y.var_}")
    else:
        print("❌ XGB scaler_y пустой или некорректный!")

    print("🔍 Проверка завершена\n")
    # ==============================================

    if None in (*lstm_models.values(), *np_models.values(), xgb_model, ppo_agent):
        bot.edit_message_text("⚠️ Не все модели загружены, проверь обученные модели.", chat_id, waiting_message.message_id)
        return

    # Корректное получение sentiment данных
    sentiment_analyzer = bot.sentiment_analyzer

    unique_dates = pd.to_datetime(combined_data[timeframe].index.date)
    filtered_dates = [date for date in unique_dates.unique() if date.date() <= datetime.utcnow().date()]

    def fetch_news_for_date(date):
        return bot.market_analyzer.news_fetcher.fetch_news_for_date(date=date.strftime('%Y-%m-%d'), symbol='BTC')

    historical_sentiment_scores = sentiment_analyzer.get_historical_sentiment_scores(
        filtered_dates, fetch_news_for_date
    )

    historical_sentiment_scores_full = sentiment_analyzer.map_sentiment_to_candles(
        combined_data[timeframe], filtered_dates, historical_sentiment_scores
    )

    fg_index, _ = bot.market_analyzer.get_full_analysis()['fear_and_greed'].values()
    fg_scaled = fg_index / 100 if fg_index else 0.5
    historical_fg_scores = [fg_scaled] * len(combined_data[timeframe])

    state_manager = StateManager(
        lstm_models, lstm_scalers, np_models,
        xgb_model, xgb_scaler_X, xgb_scaler_y,
        historical_sentiment_scores_full, historical_fg_scores, bot.lstm_manager.features
    )

    state = state_manager.create_state(combined_data, current_step)

    action, _ = ppo_agent.predict(state, deterministic=True)
    action = action.item() if isinstance(action, np.ndarray) else action  # добавь эту строчку
    decision = {0: "⏳ Удерживать позицию", 1: "📈 Покупать", 2: "📉 Продавать"}[action]

    current_price = combined_data[timeframe]['close'].iloc[-1]
    atr = get_combined_atr(combined_data)
    adx = combined_data[timeframe]['adx'].iloc[-1]

    lstm_pred = state[-5]
    np_pred = state[-4]
    xgb_pred = state[-3]
    sentiment_score = state[-2]
    fg_scaled = state[-1]

    reasons = [
        f"LSTM прогноз: {'выше 📈' if lstm_pred > current_price else 'ниже 📉'} текущей цены",
        f"NeuralProphet прогноз: {'выше 📈' if np_pred > current_price else 'ниже 📉'} текущей цены",
        f"XGBoost прогноз: {'выше 📈' if xgb_pred > current_price else 'ниже 📉'} текущей цены",
        f"Sentiment-анализ: {sentiment_score:.2f}",
        f"Fear & Greed индекс: {fg_scaled:.2f}",
        f"Тренд (ADX): {'сильный 🚀' if adx > 25 else 'слабый 🐢'} ({adx:.2f})",
        f"Волатильность (ATR): {atr:.2f}$"
    ]

    final_message = (
        f"🔮 <b>Анализ {symbol}</b>\n\n"
        f"📍 Текущая цена: <b>{current_price:.2f}$</b>\n"
        f"🧠 LSTM прогноз: <b>{lstm_pred:.2f}$</b>\n"
        f"📅 NeuralProphet прогноз: <b>{np_pred:.2f}$</b>\n"
        f"🚀 XGBoost прогноз: <b>{xgb_pred:.2f}$</b>\n"
        f"📰 Sentiment-анализ: <b>{sentiment_score:.2f}</b>\n"
        f"📌 Fear & Greed индекс: <b>{fg_scaled:.2f}</b>\n"
        f"📈 Тренд (ADX): <b>{adx:.2f}</b>\n"
        f"🌊 Волатильность (ATR): <b>{atr:.2f}$</b>\n\n"
        f"🎯 Итоговое решение PPO-агента: <b>{decision}</b>\n\n"
        f"🔍 Причины принятого решения:\n- " + "\n- ".join(reasons)
    )

    # Заменяем сообщение о загрузке на итоговый анализ
    bot.bot.edit_message_text(final_message, chat_id, waiting_message.message_id, parse_mode='HTML')

