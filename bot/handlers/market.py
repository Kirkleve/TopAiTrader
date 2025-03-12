def handle_market(bot, message):
    symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'LTC', 'DOGE', 'ADA']
    analysis = bot.market_analyzer.market_data_fetcher.fetch_market_data(symbols=symbols)

    def escape_markdown(text):
        escape_chars = r'_*[\]()~`>#+-=|{}.!'
        for char in escape_chars:
            text = text.replace(char, f"\\{char}")
        return text

    response = "📊 *Подробный обзор рынка:*\n\n"

    for symbol in symbols:
        data = analysis.get(symbol, {})
        price = data.get('price', 'нет данных')
        change = data.get('change_24h', 'нет данных')
        volume = data.get('volume_24h', 'нет данных')

        if price != 'нет данных' and change != 'нет данных' and volume != 'нет данных':
            response += (
                f"💰 *{escape_markdown(symbol)}*\n"
                f"├ Цена: *{price:,.2f}$*\n"
                f"├ Изменение за 24ч: *{change:.2f}%*\n"
                f"└ Объём (24ч): *{volume:,.0f}$*\n\n"
            )
        else:
            response += f"💰 *{escape_markdown(symbol)}*: нет данных 🔸\n\n"

    # Тренды по росту и падению
    prices_sorted = sorted(
        analysis.items(), key=lambda x: x[1].get('change_24h', 0), reverse=True
    )

    top_gainers = [sym for sym, _ in prices_sorted[:3]]
    top_losers = [sym for sym, _ in prices_sorted[-3:]]

    response += "🔸 *Тренды рынка:*\n"
    response += f"📈 Растут: {', '.join(top_gainers)}\n"
    response += f"📉 Падают: {', '.join(top_losers)}\n"

    bot.bot.send_message(
        message.chat.id,
        response,
        parse_mode="Markdown",
        disable_web_page_preview=True
    )