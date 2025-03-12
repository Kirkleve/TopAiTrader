import numpy as np
from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self):
        self.classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

    def analyze_sentiment(self, news_list):
        sentiments = self.classifier(news_list)

        scores = []
        for sentiment in sentiments:
            label = sentiment['label']
            score = sentiment['score']
            if '1' in label or '2' in label:
                scores.append(-1 * sentiment['score'])  # негатив
            elif label == '3 stars':
                scores.append(0)
            else:
                scores.append(sentiment['score'])  # позитивный

        avg_sentiment = np.mean(scores)
        return avg_sentiment
