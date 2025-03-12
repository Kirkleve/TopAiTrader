def handle_help(bot, message):
    commands = {
        "/predict": "📈 Получить прогноз цены BTC на следующий период.",
        "/accuracy": "📊 Показать текущую точность модели (RMSE).",
        "/trade": "💹 Совершить сделку (BUY, SELL или HOLD) на основе прогноза.",
        "/sentiment": "😊 Анализ настроений рынка на основе новостей.",
        "/market": "📊 Подробный анализ крипторынка и трендов.",
        "/topnews": "📰 ТОП-5 новостей крипторынка за сутки."
    }

    response = "✨ *Доступные команды:*\n\n"
    for command, description in commands.items():
        response += f"{command} — {description}\n\n"

    bot.bot.send_message(
        message.chat.id,
        response,
        parse_mode="Markdown"
    )