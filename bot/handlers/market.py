def handle_market(bot, message):
    symbols = bot.coin_manager.get_current_coins()  # Динамический выбор торгуемых монет
    symbols_simple = [symbol.split('/')[0] for symbol in symbols]

    analysis = bot.market_analyzer.market_data_fetcher.fetch_market_data(symbols=symbols_simple)
    fg_value, fg_desc = bot.market_analyzer.get_recent_fg_index()

    def escape_markdown(text):
        escape_chars = r'_*[\]()~`>#+-=|{}.!'
        for char in escape_chars:
            text = text.replace(char, f"\\{char}")
        return text

    response = f"📊 *Подробный обзор рынка*\n📌 Fear & Greed: *{fg_value}* ({fg_desc})\n\n"

    for symbol in symbols_simple:
        data = analysis.get(symbol, {})
        price = data.get('price', 'нет данных')
        change = data.get('change_24h', 'нет данных')
        volume = data.get('volume_24h', 'нет данных')

        if price != 'нет данных' and change != 'нет данных' and volume != 'нет данных':
            trend_emoji = "📈" if change >= 0 else "📉"
            response += (
                f"💰 *{escape_markdown(symbol)}* {trend_emoji}\n"
                f"├ Цена: *{price:,.2f}$*\n"
                f"├ Изменение за 24ч: *{change:+.2f}%*\n"
                f"└ Объём (24ч): *{volume:,.0f}$*\n\n"
            )
        else:
            response += f"💰 *{escape_markdown(symbol)}*: нет данных 🔸\n\n"

    # Тренды по росту и падению
    prices_sorted = sorted(
        analysis.items(), key=lambda x: x[1].get('change_24h', 0), reverse=True
    )

    top_gainers = [f"{sym} (+{data.get('change_24h', 0):.2f}%)" for sym, data in prices_sorted[:3]]
    top_losers = [f"{sym} ({data.get('change_24h', 0):.2f}%)" for sym, data in prices_sorted[-3:]]

    response += "🔸 *Тренды рынка:*\n"
    response += f"🚀 Растут: {', '.join(top_gainers)}\n"
    response += f"🔻 Падают: {', '.join(top_losers)}\n"

    bot.bot.send_message(
        message.chat.id,
        response,
        parse_mode="Markdown",
        disable_web_page_preview=True
    )
