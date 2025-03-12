def handle_feargreed(bot, message):
    index, classification = bot.market_analyzer.get_full_analysis()['fear_and_greed'].values()

    if index is None:
        bot.send_message(message.chat.id, "⚠️ Не удалось получить индекс страха и жадности.")
        return

    bot.send_message(
        message.chat.id,
        f"📊 Индекс страха и жадности: {index}/100\n"
        f"Настроение рынка: {classification}"
    )
