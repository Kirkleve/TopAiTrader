from data.cmc_data import MarketDataFetcher
from data.fear_and_greed import FearGreedIndexFetcher


class MarketAnalyzer:
    def __init__(self, news_fetcher):
        self.news_fetcher = news_fetcher  # Передаём готовый объект
        self.market_data_fetcher = MarketDataFetcher()

    def get_full_analysis(self):
        news = self.news_fetcher.fetch_news()
        market_data = self.market_data_fetcher.fetch_market_data()
        fear_greed_value, fear_greed_classification = FearGreedIndexFetcher.fetch_current_index()

        return {
            "news": news,
            "market": market_data,
            "fear_and_greed": {
                "value": fear_greed_value,
                "classification": fear_greed_classification
            }
        }