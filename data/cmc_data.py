import requests
from config import COINMARKETCAP_API_KEY

class MarketDataFetcher:
    def __init__(self):
        self.api_key = COINMARKETCAP_API_KEY
        self.quote_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        self.listings_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

    def fetch_market_data(self, symbols=None):
        """Получает актуальные данные по указанным монетам с CoinMarketCap"""
        if symbols is None:
            symbols = ['BTC', 'ETH', 'BNB']

        headers = {"X-CMC_PRO_API_KEY": self.api_key}
        params = {"symbol": ",".join(symbols), "convert": "USDT"}

        try:
            response = requests.get(self.quote_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", {})
        except requests.RequestException as e:
            print(f"❌ Ошибка запроса CoinMarketCap: {e}")
            return {}
        except (KeyError, TypeError, ValueError) as e:
            print(f"⚠️ Ошибка обработки данных CoinMarketCap: {e}")
            return {}

        return {
            symbol: {
                "price": data[symbol]["quote"]["USDT"]["price"],
                "volume_24h": data[symbol]["quote"]["USDT"]["volume_24h"],
                "market_cap": data[symbol]["quote"]["USDT"]["market_cap"],
                "change_24h": data[symbol]["quote"]["USDT"]["percent_change_24h"]
            }
            for symbol in symbols if symbol in data
        }

    def fetch_top_100(self):
        """Получает список символов топ-100 монет по капитализации с CoinMarketCap"""
        headers = {"X-CMC_PRO_API_KEY": self.api_key}
        params = {"limit": 100, "convert": "USDT"}

        try:
            response = requests.get(self.listings_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
        except requests.RequestException as e:
            print(f"❌ Ошибка запроса топ-100 CoinMarketCap: {e}")
            return []
        except (KeyError, TypeError, ValueError) as e:
            print(f"⚠️ Ошибка обработки данных CoinMarketCap: {e}")
            return []

        return [item["symbol"] for item in data]
