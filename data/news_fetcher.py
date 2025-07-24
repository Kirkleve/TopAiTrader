import requests
import pandas as pd


class NewsFetcher:
    @staticmethod
    def fetch_news_for_date(date, symbol="BTC"):
        try:
            date = pd.to_datetime(date)

            # Проверка на даты в будущем
            today = pd.Timestamp.now(tz='UTC').normalize().date()

            if date.date() > today:
                print(f"⚠️ Запрошена дата из будущего ({date.date()}), новости отсутствуют.")
                return []

            api_url = f"https://cryptonews-api.com/api/v1?tickers={symbol}&date={date.strftime('%Y-%m-%d')}"

            response = requests.get(api_url, timeout=10)

            # Добавь здесь проверку на статус код 404
            if response.status_code == 404:
                # Без вывода ошибки, просто нет новостей на дату
                return []

            response.raise_for_status()

            news_data = response.json()

            if 'data' not in news_data or not isinstance(news_data['data'], list):
                print("⚠️ API вернул неожиданный формат данных:", news_data)
                return []

            news_list = [
                news_item['title'] + '. ' + news_item['text']
                for news_item in news_data['data']
                if 'title' in news_item and 'text' in news_item
            ]

            return news_list

        except requests.exceptions.RequestException as e:
            print(f"🚨 Ошибка при запросе новостей: {e}")
            return []

        except (ValueError, TypeError) as e:
            print(f"🚨 Ошибка обработки даты ({date}): {e}")
            return []
