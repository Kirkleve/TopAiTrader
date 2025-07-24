import numpy as np
import pandas as pd
from transformers import pipeline
from typing import Literal

class SentimentAnalyzer:
    def __init__(self, model_type="bert"):
        if model_type == "finbert":
            self.classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')
        else:
            self.classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

    def analyze_sentiment(self, news_list):
        sentiments = self.classifier(news_list)
        scores = []
        for sentiment in sentiments:
            label = sentiment['label']
            score = sentiment['score']
            if 'negative' in label.lower():
                scores.append(-1 * score)
            elif 'neutral' in label.lower():
                scores.append(0)
            else:
                scores.append(score)
        return np.mean(scores)

    def get_historical_sentiment_scores(self, dates, fetch_news_for_date):
        """
        dates: список дат (datetime)
        fetch_news_for_date: функция, принимающая дату и возвращающая список новостей
        Возвращает список сентимент-скоров по каждой дате.
        """
        historical_scores = []
        for date in dates:
            daily_news = fetch_news_for_date(date)
            daily_score = self.analyze_sentiment(daily_news)
            historical_scores.append(daily_score)
        return historical_scores

    @staticmethod
    def map_sentiment_to_candles(
            candles_df,
            sentiment_dates,
            sentiment_scores,
            fill_method: Literal["backfill", "bfill", "ffill", "pad"] = 'ffill'
    ):
        sentiment_map = {
            pd.to_datetime(date).strftime('%Y-%m-%d'): score
            for date, score in zip(sentiment_dates, sentiment_scores)
        }

        mapped_scores = pd.Series(candles_df.index.strftime('%Y-%m-%d').map(sentiment_map))

        if fill_method:
            mapped_scores = mapped_scores.fillna(method=fill_method)

        return mapped_scores.tolist()