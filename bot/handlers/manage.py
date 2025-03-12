import os
from trainer.train import train  # твой новый универсальный модуль!
from data.cmc_data import MarketDataFetcher
from bot.manage_coins import CoinManager

def handle_manage(bot, message):
    coins = bot.coin_manager.get_current_coins()

    pnl_report = "\n".join([
        f"{coin.split('/')[0]}: Сегодня {bot.trader.get_pnl_period(coin, 'day')}%, "
        f"Неделя {bot.trader.get_pnl_period(coin, 'week')}%, "
        f"Месяц {bot.trader.get_pnl_period(coin, 'month')}%, "
        f"Год {bot.trader.get_pnl_period(coin, 'year')}%"
        for coin in coins
    ])

    bot.send_message(
        message.chat.id,
        f"💎 Какую монету будем торговать?\n\n📌 Текущие монеты и их PNL:\n{pnl_report}"
    )

    bot.bot.register_next_step_handler(
        message, lambda msg: process_coin_choice(bot, msg)
    )

def process_coin_choice(bot, message):
    coin = message.text.strip().upper().replace('/USDT', '')
    coin_full = coin + '/USDT'

    market_fetcher = MarketDataFetcher()
    top_100_coins = market_fetcher.fetch_top_100()

    if coin not in top_100_coins:
        bot.send_message(
            message.chat.id,
            "🚫 Извини, такой монеты нет или она не в топ-100 CoinMarketCap."
        )
        return

    coin_manager: CoinManager = bot.coin_manager

    if coin_full in coin_manager.get_current_coins():
        bot.send_message(
            message.chat.id,
            f"⚠️ Монета {coin} уже добавлена и торгуется! Просто запусти /autotrade."
        )
        return

    if coin_manager.coin_limit_reached():
        pnl_report = "\n".join([
            f"{c.split('/')[0]} ({bot.trader.get_pnl_period(c, 'day'):+.2f}%)"
            for c in coin_manager.get_current_coins()
        ])

        bot.send_message(
            message.chat.id,
            f"🚫 Уже торгуется максимальное число монет (5):\n{pnl_report}\n\n"
            f"📌 Напиши название монеты, которую хочешь заменить на {coin}:"
        )
        bot.bot.register_next_step_handler(
            message, lambda msg: replace_coin_choice(bot, msg, coin)
        )
    else:
        result = coin_manager.add_coin(coin)
        bot.send_message(message.chat.id, result)

        # Проверка наличия ОБЕИХ моделей (LSTM+DQN)
        lstm_model_dir = f'trainer/models/{coin}_USDT'
        dqn_model_path = f'trading/trained_agent_{coin}_USDT.pth'

        if not os.path.exists(lstm_model_dir) or not os.path.exists(dqn_model_path):
            bot.send_message(message.chat.id, f"⏳ Обучаю модели для {coin}/USDT (~3-5 мин.)...")
            train(coin_full)  # Универсальное обучение
            bot.send_message(message.chat.id, f"✅ Модели для {coin}/USDT успешно обучены!")
        else:
            bot.send_message(message.chat.id, f"✅ Модели для {coin}/USDT уже существуют!")

def replace_coin_choice(bot, message, new_coin):
    old_coin = message.text.strip().upper()
    old_coin_full = old_coin + '/USDT'

    coin_manager: CoinManager = bot.coin_manager

    if old_coin_full not in coin_manager.get_current_coins():
        bot.send_message(
            message.chat.id,
            f"⚠️ Монета {old_coin} не найдена в списке торгуемых."
        )
        return

    result = coin_manager.replace_coin(old_coin, new_coin)
    bot.send_message(message.chat.id, result)

    lstm_model_dir = f'trainer/models/{new_coin}_USDT'
    dqn_model_path = f'trading/trained_agent_{new_coin}_USDT.pth'

    if not os.path.exists(lstm_model_dir) or not os.path.exists(dqn_model_path):
        bot.send_message(message.chat.id, f"⏳ Обучаю модели для {new_coin}/USDT (~3-5 мин.)...")
        train(new_coin + '/USDT')  # Универсальное обучение
        bot.send_message(message.chat.id, f"✅ Модели для {new_coin}/USDT успешно обучены!")
    else:
        bot.send_message(message.chat.id, f"✅ Модели для {new_coin}/USDT уже существуют!")
