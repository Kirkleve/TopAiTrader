def handle_help(bot, message):
    commands = {
        "/predict": "📈 Прогноз цены BTC на основе всех моделей (LSTM, NeuralProphet, XGB) и PPO-агента.",
        "/accuracy": "📊 Проверить точность и эффективность текущей стратегии торговли.",
        "/sentiment": "🗞️ Общий сантимент рынка: новости + индекс страха и жадности.",
        "/market": "📌 Подробный обзор текущего состояния крипторынка и трендов.",
        "/topnews": "🚀 ТОП новости крипторынка из CoinMarketCap и CryptoPanic.",
        "/manage": "💎 Управление монетами для торговли (добавить, заменить, удалить).",
        "/autotrade": "🤖 Запустить или остановить автоматическую торговлю ботом.",
        "/help": "📖 Показать этот список команд."
    }

    response = "✨ *Доступные команды бота:*\n\n"
    for command, description in commands.items():
        response += f"{command} — {description}\n\n"

    bot.bot.send_message(
        message.chat.id,
        response,
        parse_mode="Markdown"
    )
