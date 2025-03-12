def handle_sentiment(bot, message):
    news_list = bot.news_fetcher.fetch_news()

    print(f"Полученные новости: {news_list}")  # Проверь, есть ли вообще новости.

    if not news_list:
        bot.send_message(message.chat.id, "⚠️ Новости не найдены.")
        return

    sentiment_result = bot.sentiment_analyzer(news_list)

    print(f"Анализ тональности: {sentiment_result}")  # Проверь результат анализа.

    scores = []
    for sentiment in sentiment_result:
        label = sentiment['label']
        if '1' in label or '2' in label:
            scores.append(-1 * sentiment['score'])
        elif label == '3 stars':
            scores.append(0)
        else:
            scores.append(sentiment['score'])

    if not scores:
        bot.send_message(message.chat.id, "⚠️ Новости не найдены или не удалось определить тональность.")
        return

    avg_sentiment = sum(scores) / len(scores)

    if avg_sentiment > 0:
        sentiment_status = "😊 Позитивный (рост вероятен)"
    elif avg_sentiment < 0:
        sentiment_status = "😟 Негативный (возможен спад)"
    else:
        sentiment_status = "😐 Нейтральный"

    response = f"📰 Настроение рынка сейчас: {sentiment_status}\nОценка: {avg_sentiment:.2f}"
    bot.send_message(message.chat.id, response)