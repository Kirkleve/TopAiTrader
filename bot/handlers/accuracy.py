import numpy as np

from trainer.model_manager.state_manager import StateManager
from trading.crypto_env import CryptoTradingEnv


def handle_accuracy(bot, message, symbol='BTC/USDT', timeframe='1h'):
    combined_data = bot.data_fetcher.fetch_historical_data_multi_timeframe(symbol)

    if timeframe not in combined_data or combined_data[timeframe].empty:
        bot.send_message(message.chat.id, f"⚠️ Нет данных для {symbol} на таймфрейме {timeframe}.")
        return

    features = bot.models.features

    lstm_models = {tf: bot.lstm_manager.load_model(tf) for tf in ['15m', '1h', '4h', '1d']}
    lstm_scalers = {tf: bot.lstm_manager.load_scaler(tf) for tf in ['15m', '1h', '4h', '1d']}

    np_models = {tf: bot.np_manager.load_model(tf) for tf in ['15m', '1h', '4h', '1d']}

    xgb_model, xgb_scaler_X, xgb_scaler_y = bot.xgb_manager.load_model_and_scalers()

    ppo_agent = bot.ppo_manager.load_model()

    if None in (*lstm_models.values(), *np_models.values(), xgb_model, ppo_agent):
        bot.send_message(message.chat.id, "⚠️ Не все модели загружены, проверь обученные модели.")
        return

    historical_sentiment_scores = bot.sentiment_analyzer.get_historical_sentiment_scores()
    historical_fg_scores = bot.market_analyzer.get_historical_fg_scores()

    state_manager = StateManager(
        lstm_models, lstm_scalers, np_models,
        xgb_model, xgb_scaler_X, xgb_scaler_y,
        historical_sentiment_scores, historical_fg_scores, features
    )

    min_length = min(len(df) for df in combined_data.values())
    total_rewards = []
    total_profit = 0

    env = CryptoTradingEnv(
        symbol=symbol,
        data=np.array([state_manager.create_state(combined_data, i) for i in range(20, min_length)]),
        sentiment_scores=historical_sentiment_scores,
        lstm_models=lstm_models,
        np_models=np_models,
        xgb_model=xgb_model,
        xgb_scaler_X=xgb_scaler_X,
        xgb_scaler_y=xgb_scaler_y,
        observation_scaler=bot.ppo_manager.observation_scaler
    )

    state, _ = env.reset()
    done = False

    while not done:
        action, _ = ppo_agent.predict(state, deterministic=True)
        state, reward, done, _, _ = env.step(action)
        total_rewards.append(reward)
        total_profit += reward

    profitable_trades = len([r for r in total_rewards if r > 0])
    total_trades = len(total_rewards)
    avg_profit_per_trade = total_profit / total_trades if total_trades else 0
    win_rate = (profitable_trades / total_trades) * 100 if total_trades else 0

    bot.send_message(
        message.chat.id,
        f"🤖 *Оценка эффективности PPO агента ({symbol})*\n"
        f"📊 Итоговая прибыль: {total_profit:.2f}%\n"
        f"📈 Средний профит на сделку: {avg_profit_per_trade:.2f}%\n"
        f"🚩 Процент прибыльных сделок: {win_rate:.2f}%\n"
        f"🗂️ Количество сделок в симуляции: {total_trades}\n"
        f"✅ Таймфрейм оценки: {timeframe}"
    )
