import requests
from config import CRIPTOPANIC_API_KEY

class CryptoNewsFetcher:
    def __init__(self, summarizer):
        self.api_key = CRIPTOPANIC_API_KEY
        self.url = "https://cryptopanic.com/api/v1/posts/"
        self.summarizer = summarizer

    def fetch_news(self, symbols=None):
        if symbols is None:
            symbols = ['BTC', 'ETH']

        params = {
            "auth_token": self.api_key,
            "currencies": ",".join(symbols),
            "public": "true"
        }

        try:
            response = requests.get(self.url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("results", [])
            news_with_summary = []

            for item in data[:5]:
                title = item.get('title', '')
                url = item.get('url', '')
                short_summary = self.summarizer.summarize_and_translate(title)
                news_with_summary.append(f"{short_summary} - {url}")

            return news_with_summary
        except requests.RequestException as e:
            print(f"Ошибка запроса: {e}")
            return []
