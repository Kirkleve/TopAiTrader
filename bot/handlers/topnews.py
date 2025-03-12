def handle_topnews(bot, message):
    cmc_data = bot.market_analyzer.get_full_analysis()

    news_list = cmc_data.get("news", [])
    if not news_list:
        bot.send_message(message.chat.id, "⚠️ Новости не найдены.")
        return

    def escape_markdown(text):
        escape_chars = r'_*[\]()~`>#+-=|{}.!'
        for char in escape_chars:
            text = text.replace(char, f"\\{char}")
        return text

    response = "📰 *ТОП-5 новостей за 24 часа:*\n\n"

    for news_item in news_list[:5]:
        if " - " in news_item:
            title, url = news_item.rsplit(" - ", 1)
            response += f"• [{escape_markdown(title.strip())}]({url.strip()})\n"
        else:
            response += f"• {escape_markdown(news_item)}\n"

    bot.bot.send_message(message.chat.id, response, parse_mode="Markdown", disable_web_page_preview=True)