from datetime import datetime

from data.cmc_data import MarketDataFetcher
from data.fear_and_greed import FearGreedIndexFetcher


class MarketAnalyzer:
    def __init__(self, news_fetcher):
        self.news_fetcher = news_fetcher
        self.market_data_fetcher = MarketDataFetcher()

    def get_full_analysis(self, symbol='BTC'):
        date_today = datetime.today().strftime('%Y-%m-%d')

        try:
            news = self.news_fetcher.fetch_news_for_date(date_today, symbol)
        except Exception as e:
            print(f"⚠️ Ошибка при получении новостей: {e}")
            news = []

        try:
            market_data = self.market_data_fetcher.fetch_market_data()
        except Exception as e:
            print(f"⚠️ Ошибка при получении данных рынка: {e}")
            market_data = {}

        try:
            fear_greed_value, fear_greed_classification = FearGreedIndexFetcher.fetch_current_index()
        except Exception as e:
            print(f"⚠️ Ошибка при получении индекса страха и жадности: {e}")
            fear_greed_value, fear_greed_classification = None, "нет данных"

        return {
            "news": news,
            "market": market_data,
            "fear_and_greed": {
                "value": fear_greed_value,
                "classification": fear_greed_classification
            }
        }
