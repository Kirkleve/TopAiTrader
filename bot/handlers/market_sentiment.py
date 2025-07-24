def handle_market_sentiment(bot, message):
    chat_id = message.chat.id

    # Анализ новостей
    news_list = bot.news_fetcher.fetch_news()

    if not news_list:
        news_sentiment = "⚠️ Новости не найдены."
    else:
        sentiment_result = bot.sentiment_analyzer(news_list)

        scores = []
        for sentiment in sentiment_result:
            label = sentiment['label'].lower()
            if 'negative' in label or '1' in label or '2' in label:
                scores.append(-sentiment['score'])
            elif 'neutral' in label or '3' in label:
                scores.append(0)
            else:
                scores.append(sentiment['score'])

        if not scores:
            news_sentiment = "😐 Не удалось определить тональность новостей."
        else:
            avg_sentiment = sum(scores) / len(scores)

            if avg_sentiment > 0.1:
                news_sentiment = f"😊 Позитивный ({avg_sentiment:.2f})"
            elif avg_sentiment < -0.1:
                news_sentiment = f"😟 Негативный ({avg_sentiment:.2f})"
            else:
                news_sentiment = f"😐 Нейтральный ({avg_sentiment:.2f})"

    # Индекс страха и жадности
    fg_index, fg_classification = bot.market_analyzer.get_recent_fg_index()

    if fg_index is None:
        fg_message = "⚠️ Индекс страха и жадности не получен."
    else:
        fg_message = f"{fg_index}/100 ({fg_classification})"

    # Советы на основе FG index
    if fg_index is not None:
        if fg_index < 20:
            advice = "📌 Высокий страх, хорошее время для покупок."
        elif fg_index > 75:
            advice = "📌 Высокая жадность, будь осторожен с покупками."
        else:
            advice = "📌 Рынок умеренный, оценивай ситуацию внимательно."
    else:
        advice = "📌 Не удалось получить совет по текущей ситуации."

    # Формирование ответа
    response = (
        f"📰 <b>Настроение рынка сейчас:</b>\n\n"
        f"• <b>Новости:</b> {news_sentiment}\n"
        f"• <b>Fear & Greed индекс:</b> {fg_message}\n\n"
        f"{advice}"
    )

    bot.send_message(chat_id, response, parse_mode='HTML')
