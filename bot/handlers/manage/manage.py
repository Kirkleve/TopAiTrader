import os

from bot.handlers.manage.manage_coins import CoinManager
from trainer.train import train
from data.cmc_data import MarketDataFetcher
from .smart_coin_selector import SmartCoinSelector


def handle_manage(bot, message):
    smart_selector = SmartCoinSelector(
        sentiment_analyzer=bot.sentiment_analyzer,
        trader=bot.trader
    )

    recommended_coins = smart_selector.select_best_coins(final_count=5)
    recommended_list = "\n".join([f"• {coin}" for coin in recommended_coins])

    coins = bot.coin_manager.get_current_coins()

    pnl_report = "\n".join([
        f"{coin.split('/')[0]}: Сегодня {bot.trader.get_pnl_period(coin, 'day')}%, "
        f"Неделя {bot.trader.get_pnl_period(coin, 'week')}%, "
        f"Месяц {bot.trader.get_pnl_period(coin, 'month')}%, "
        f"Год {bot.trader.get_pnl_period(coin, 'year')}%"
        for coin in coins
    ])

    response = (
        f"💎 <b>Какую монету будем торговать?</b>\n\n"
        f"📌 <b>Текущие монеты и их PNL:</b>\n{pnl_report}\n\n"
        f"🔥 <b>Лучшие монеты (умный отбор):</b>\n{recommended_list}\n\n"
        f"Напиши название монеты из списка выше или укажи свою:"
    )

    bot.send_message(message.chat.id, response, parse_mode="HTML")
    bot.bot.register_next_step_handler(
        message, lambda msg: process_coin_choice(bot, msg, recommended_coins)
    )

def process_coin_choice(bot, message, recommended_coins):
    coin = message.text.strip().upper().replace('/USDT', '')
    coin_full = coin + '/USDT'

    market_fetcher = MarketDataFetcher()
    top_100_coins = market_fetcher.fetch_top_100()

    if coin_full not in recommended_coins and coin not in top_100_coins:
        bot.send_message(
            message.chat.id,
            "🚫 Извини, такой монеты нет среди рекомендованных и она не в топ-100 CoinMarketCap."
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
        check_and_train_models(bot, message, coin)


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
    check_and_train_models(bot, message, new_coin)


def check_and_train_models(bot, message, coin):
    model_paths = [
        os.path.exists(f'models/{coin}_USDT/lstm'),
        os.path.exists(f'models/{coin}_USDT/neuralprophet'),
        os.path.exists(f'models/{coin}_USDT/xgb'),
        os.path.exists(f'models/{coin}_USDT/ppo')
    ]

    if not all(model_paths):
        bot.send_message(message.chat.id, f"⏳ Обучаю модели для {coin}/USDT (~3-5 мин.)...")
        train(coin + '/USDT')
        bot.send_message(message.chat.id, f"✅ Модели для {coin}/USDT успешно обучены!")
    else:
        bot.send_message(message.chat.id, f"✅ Модели для {coin}/USDT уже существуют!")
