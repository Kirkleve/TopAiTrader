from trainer.model_manager import ModelManager
from trading.binance_trader import BinanceTrader
from data.atr import get_combined_atr
from strategy.adaptive_strategy import adapt_strategy
import torch


def handle_autotrade(bot, chat_id):
    symbols = bot.coin_manager.get_current_coins()
    trader = BinanceTrader()
    strategy_params = adapt_strategy()

    bot.send_message(chat_id, f"🤖 Запускаю автотрейдинг для: {', '.join(symbols)}")

    balance = trader.exchange.fetch_balance()['free']['USDT']
    risk_percent = 0.02

    for symbol in symbols:
        manager = ModelManager(symbol)
        agent = manager.load_trained_model()
        if agent is None:
            bot.send_message(chat_id, f"⚠️ PPO-модель для {symbol} не найдена, пропускаем.")
            continue

        state_data, sentiment_score, fear_greed_scaled = manager.get_state_data()
        if state_data is None:
            bot.send_message(chat_id, f"⚠️ Нет данных для {symbol}, пропускаем.")
            continue

        state_tensor = torch.tensor(state_data, dtype=torch.float32).unsqueeze(0)
        action = agent.predict(state_tensor, deterministic=True)[0]

        if action == 0:
            bot.send_message(chat_id, f"⏳ {symbol}: агент решил не торговать.")
            continue

        side = 'buy' if action == 1 else 'sell'
        atr = get_combined_atr(manager.get_historical_data())

        position = trader.get_position(symbol)
        if position:
            trader.close_all_positions(symbol)
            bot.send_message(chat_id, f"🔄 Закрыта текущая позиция по {symbol}.")

        trader.create_order_with_sl_tp(
            symbol=symbol,
            side=side,
            balance=balance,
            risk_percent=risk_percent,
            atr=atr,
            sl_multiplier=strategy_params.get('sl_multiplier', 1.5),
            tp_multiplier=strategy_params.get('tp_multiplier', 3),
            adaptive=True
        )

        bot.send_message(chat_id, f"🚀 Открыта позиция {side.upper()} по {symbol}")

    bot.send_message(chat_id, "✅ Автотрейдинг завершён!")
