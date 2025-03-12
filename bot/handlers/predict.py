import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.fear_and_greed import FearGreedIndexFetcher


def handle_predict(bot, message, symbol='BTC/USDT'):
    df_main = bot.data_fetcher.fetch_historical_data_multi_timeframe(symbol)['1h']

    if df_main.empty:
        bot.send_message(message.chat.id, f"⚠️ Нет данных для {symbol}.")
        return

    features = ['close', 'rsi', 'ema', 'adx', 'atr', 'volume']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_main[features])

    seq_length = 20
    X_tensor = torch.tensor(scaled_features[-seq_length:], dtype=torch.float32).unsqueeze(0)

    prediction_norm = bot.lstm_model(X_tensor).item()

    dummy = np.zeros((1, len(features)), dtype=np.float32)
    dummy[0][0] = prediction_norm # type: ignore
    predicted_price = scaler.inverse_transform(dummy)[0, 0]

    current_price = df_main['close'].iloc[-1]

    # Анализ sentiment
    sentiment_result = bot.sentiment_analyzer(symbol.split('/')[0])[0]
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']

    if sentiment_label.lower() in ['negative', 'негативный']:
        sentiment_score = -sentiment_score
    elif sentiment_label.lower() in ['neutral', 'нейтральный']:
        sentiment_score = 0

    fear_greed_value, fear_greed_desc = FearGreedIndexFetcher.fetch_current_index()
    fear_greed_scaled = fear_greed_value / 100 if fear_greed_value else 0.5

    action = 'покупать 📈' if predicted_price > current_price else 'продавать 📉'

    reasons = [
        f"RSI: {df_main['rsi'].iloc[-1]:.2f}",
        f"EMA: {df_main['ema'].iloc[-1]:.2f}",
        f"ADX: {df_main['adx'].iloc[-1]:.2f}",
        f"ATR: {df_main['atr'].iloc[-1]:.4f}",
        f"Sentiment: {'позитивный 🟢' if sentiment_score > 0 else 'негативный 🔴' if sentiment_score < 0 else 'нейтральный 🟡'}",
        f"Fear & Greed: {fear_greed_value} ({fear_greed_desc}, scaled: {fear_greed_scaled:.2f})"
    ]

    final_message = (
        f"🔮 *Прогноз по {symbol}*\n"
        f"📍 Текущая цена: {current_price:.2f}$\n"
        f"🎯 Прогнозируемая цена: {predicted_price:.2f}$\n"
        f"🤖 Решение: {action}\n\n"
        f"📌 Причины принятого решения:\n"
        f"{chr(10).join(reasons)}"
    )

    bot.send_message(message.chat.id, final_message, parse_mode='Markdown')
