import requests

class FearGreedIndexFetcher:
    API_URL = "https://api.alternative.me/fng/"

    @staticmethod
    def fetch_current_index():
        try:
            response = requests.get(FearGreedIndexFetcher.API_URL)
            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                value = int(data["data"][0]["value"])
                classification = data["data"][0]["value_classification"]
                return value, classification
            else:
                return None, None
        except Exception as e:
            print(f"⚠️ Ошибка получения Fear & Greed Index: {e}")
            return None, None
