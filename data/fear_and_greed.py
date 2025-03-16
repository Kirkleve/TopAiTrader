import requests
import time

class FearGreedIndexFetcher:
    API_URL = "https://api.alternative.me/fng/"
    last_update = 0
    cached_value = 0.5  # Начальное значение (нейтральное)
    cached_classification = "Neutral"

    @staticmethod
    def fetch_current_index():
        """ Загружает индекс страха и жадности с API, кэширует результат. """
        current_time = time.time()

        # **Если с последнего запроса прошло меньше 10 минут, используем кеш**
        if current_time - FearGreedIndexFetcher.last_update < 600:
            return FearGreedIndexFetcher.cached_value, FearGreedIndexFetcher.cached_classification

        try:
            response = requests.get(FearGreedIndexFetcher.API_URL, timeout=5)
            response.raise_for_status()  # Вызывает исключение, если ошибка запроса
            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                value = int(data["data"][0]["value"]) / 100  # **Нормализуем (0-1)**
                classification = data["data"][0]["value_classification"]

                # **Обновляем кеш**
                FearGreedIndexFetcher.cached_value = value
                FearGreedIndexFetcher.cached_classification = classification
                FearGreedIndexFetcher.last_update = current_time
                return value, classification
            else:
                return 0.5, "Neutral"  # **Возвращаем нейтральное значение, если API не ответил**
        except requests.RequestException as e:
            print(f"⚠️ Ошибка получения Fear & Greed Index: {e}")
            return 0.5, "Neutral"  # **Если ошибка, используем кешированное значение**
