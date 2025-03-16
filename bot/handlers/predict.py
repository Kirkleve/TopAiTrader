import numpy as np
from data.fetch_data import CryptoDataFetcher

def handle_predict(bot, message, symbol='BTC/USDT'):
    chat_id = message.chat.id
    models = bot.models.get(symbol.replace('/', '_'))

    if not models:
        bot.send_message("⚠️ Модели не найдены.")
        return

    data_fetcher = CryptoDataFetcher()
    prepared_data = data_fetcher.prepare_full_state(symbol, models)

    if not prepared_data:
        bot.send_message("⚠️ Ошибка при подготовке данных.")
        return

    state = prepared_data['state']
    current_price = prepared_data['current_price']
    lstm_preds = prepared_data['lstm_preds']
    xgb_pred = prepared_data['xgb_pred']
    sentiment_score = prepared_data['sentiment_score']
    sentiment_label = prepared_data['sentiment_label']
    fg_scaled = prepared_data['fg_scaled']
    fear_greed_value = prepared_data['fear_greed_value']
    fear_greed_desc = prepared_data['fear_greed_desc']

    sentiment_dir = {
        'positive': "🟢 позитивный",
        'negative': "🔴 негативный",
        'neutral': "⚪ нейтральный"
    }.get(sentiment_label.lower(), "⚪ нейтральный")

    # PPO прогнозирует решение
    action, _ = models['ppo'].predict(state, deterministic=True)

    decision = {
        0: "⏳ Удерживать позицию",
        1: "📈 Покупать",
        2: "📉 Продавать"
    }.get(action, "⏳ Удерживать позицию")

    reasons = [
        f"LSTM прогноз: {'выше 📈' if prepared_data['lstm_pred'] > prepared_data['current_price'] else 'ниже 📉'} текущей цены",
        f"XGBoost прогноз: {'выше 📈' if xgb_pred > prepared_data['current_price'] else 'ниже 📉'} текущей цены",
        f"Sentiment-анализ: {sentiment_label} ({sentiment_score:.2f})",
        f"Fear & Greed: {fear_greed_desc} ({fear_greed_value})"
    ]

    final_message = (
        f"🔮 <b>Анализ {symbol}</b>\n\n"
        f"📍 Текущая цена: <b>{prepared_data['current_price']:.2f}$</b>\n\n"
        f"🧠 LSTM прогноз: <b>{prepared_data['lstm_pred']:.2f}$</b>\n"
        f"🚀 XGBoost прогноз: <b>{xgb_pred:.2f}$</b>\n"
        f"📰 Sentiment-анализ: <b>{sentiment_label}</b> ({sentiment_score:.2f})\n"
        f"📌 Fear & Greed: <b>{fear_greed_value}</b> ({fear_greed_desc}, scaled: {fg_scaled:.2f})\n\n"
        f"🎯 Итоговое решение PPO-агента: <b>{decision}</b>\n\n"
        f"🔍 Причины принятого решения:\n- " + "\n- ".join(reasons)
    )

    bot.send_message(final_message, parse_mode='HTML')
