# initializer/components.py
from transformers import pipeline
from data.fetch_data import CryptoDataFetcher
from data.cryptopanic_data import CryptoNewsFetcher
from data.news_summary import NewsSummarizer
from data.market_analyzer import MarketAnalyzer

def initialize_components():
    data_fetcher = CryptoDataFetcher()
    sentiment_analyzer = pipeline("sentiment-analysis", model='nlptown/bert-base-multilingual-uncased-sentiment')
    translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    summarizer = NewsSummarizer(translator=translator_pipeline, summarizer=summarization_pipeline)
    news_fetcher = CryptoNewsFetcher(summarizer)
    market_analyzer = MarketAnalyzer(news_fetcher)

    return data_fetcher, sentiment_analyzer, news_fetcher, summarizer, market_analyzer
