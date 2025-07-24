def handle_topnews(bot, message):
    chat_id = message.chat.id

    # Собираем новости из CoinMarketCap
    cmc_data = bot.market_analyzer.get_full_analysis()
    cmc_news = cmc_data.get("news", [])[:3]  # Берём топ-3 новости

    # Собираем новости из CryptoPanic
    cryptopanic_news = bot.crypto_news_fetcher.fetch_news()[:3]  # Топ-3 новости с кратким саммари

    # Если нет новостей
    if not cmc_news and not cryptopanic_news:
        bot.send_message(chat_id, "⚠️ Новости не найдены.")
        return

    def escape_markdown(text):
        escape_chars = r'_*[\]()~`>#+-=|{}.!'
        for char in escape_chars:
            text = text.replace(char, f"\\{char}")
        return text

    response = "🗞️ *ТОП Новости крипторынка сейчас:*\n\n"

    # Новости из CoinMarketCap
    if cmc_news:
        response += "🚀 *CoinMarketCap:*\n"
        for news_item in cmc_news:
            if " - " in news_item:
                title, url = news_item.rsplit(" - ", 1)
                response += f"• [{escape_markdown(title.strip())}]({url.strip()})\n"
            else:
                response += f"• {escape_markdown(news_item)}\n"
        response += "\n"

    # Новости из CryptoPanic (с кратким описанием)
    if cryptopanic_news:
        response += "🔮 *CryptoPanic:*\n"
        for news_item in cryptopanic_news:
            if " - " in news_item:
                summary, url = news_item.rsplit(" - ", 1)
                response += f"• [{escape_markdown(summary.strip())}]({url.strip()})\n"
            else:
                response += f"• {escape_markdown(news_item)}\n"

    # Отправка итогового сообщения
    bot.bot.send_message(
        chat_id, response,
        parse_mode="Markdown",
        disable_web_page_preview=True
    )
