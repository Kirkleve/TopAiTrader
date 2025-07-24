import requests
import time
from datetime import datetime

class FearGreedIndexFetcher:
    API_URL = "https://api.alternative.me/fng/"
    last_update = 0
    cached_value = 0.5
    cached_classification = "Neutral"

    @staticmethod
    def fetch_current_index():
        current_time = time.time()
        if current_time - FearGreedIndexFetcher.last_update < 600:
            return FearGreedIndexFetcher.cached_value, FearGreedIndexFetcher.cached_classification
        try:
            response = requests.get(FearGreedIndexFetcher.API_URL, timeout=5)
            response.raise_for_status()
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                value = int(data["data"][0]["value"]) / 100
                classification = data["data"][0]["value_classification"]
                FearGreedIndexFetcher.cached_value = value
                FearGreedIndexFetcher.cached_classification = classification
                FearGreedIndexFetcher.last_update = current_time
                return value, classification
            else:
                return 0.5, "Neutral"
        except requests.RequestException as e:
            print(f"⚠️ Ошибка получения текущего Fear & Greed Index: {e}")
            return 0.5, "Neutral"

    @staticmethod
    def fetch_historical_index(date):
        try:
            response = requests.get(f"{FearGreedIndexFetcher.API_URL}?limit=365",
                                    timeout=10)  # получаем сразу много данных за последний год
            response.raise_for_status()
            data = response.json()

            target_date = date.strftime("%d-%m-%Y")
            for entry in data["data"]:
                if entry["timestamp"]:
                    entry_date = datetime.fromtimestamp(int(entry["timestamp"])).strftime("%d-%m-%Y")
                    if entry_date == target_date:
                        return int(entry["value"])

            print(f"⚠️ Нет данных для даты {target_date}")
            return 50

        except requests.RequestException as e:
            print(f"⚠️ Ошибка получения исторического Fear & Greed Index: {e}")
            return 50

    @staticmethod
    def get_historical_fg_scores(dates):
        """
        dates: список дат (datetime)
        Возвращает список исторических значений индекса.
        """
        fg_scores = []
        for date in dates:
            fg_index = FearGreedIndexFetcher.fetch_historical_index(date)
            fg_scores.append(fg_index)
        return fg_scores
