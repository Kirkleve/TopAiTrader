import numpy as np
from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self, model_type="bert"):
        """
        model_type: "bert" (стандартный) или "finbert" (адаптированный для финансовых новостей)
        """
        if model_type == "finbert":
            self.classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')
        else:
            self.classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

    def analyze_sentiment(self, news_list):
        """
        Анализирует список новостей и возвращает средний score от -1 (негатив) до 1 (позитив).
        """
        sentiments = self.classifier(news_list)
        scores = []

        for sentiment in sentiments:
            label = sentiment['label']
            score = sentiment['score']
            if 'negative' in label.lower():
                scores.append(-1 * score)  # Негативный
            elif 'neutral' in label.lower():
                scores.append(0)
            else:
                scores.append(score)  # Позитивный

        return np.mean(scores)
