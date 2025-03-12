import json
import os

class SimpleTradingStrategy:
    def __init__(self, params_file='strategy_params.json'):
        self.params_file = params_file
        self.load_params()

    def load_params(self):
        if os.path.exists(self.params_file):
            with open(self.params_file, 'r') as f:
                params = json.load(f)
        else:
            params = {
                "threshold_percent": 0.5,
                "sentiment_threshold": 0.6,
                "min_volume": 1e9
            }
            self.save_params()

        self.threshold_percent = params["threshold_percent"]
        self.sentiment_threshold = params["sentiment_threshold"]
        self.min_volume = params.get("min_volume", 1e9)

    def save_params(self):
        params = {
            "threshold_percent": self.threshold_percent,
            "sentiment_threshold": self.sentiment_threshold,
            "min_volume": self.min_volume
        }
        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=4)

    def update_params(self, threshold_percent=None, sentiment_threshold=None, min_volume=None):
        if threshold_percent is not None:
            self.threshold_percent = threshold_percent
        if sentiment_threshold is not None:
            self.sentiment_threshold = sentiment_threshold
        if min_volume is not None:
            self.min_volume = min_volume
        self.save_params()

    def decide(self, current_price, predicted_price, market_data, sentiment_score):
        diff_percent = ((predicted_price - current_price) / current_price) * 100

        if market_data["volume_24h"] < self.min_volume:
            return "hold", "Низкий объём торгов."

        if sentiment_score < self.sentiment_threshold:
            return "hold", "Негативный sentiment-анализ."

        if diff_percent > self.threshold_percent:
            return 'buy', f"Ожидаемый рост {diff_percent:.2f}% превышает порог {self.threshold_percent:.2f}%."
        elif diff_percent < -self.threshold_percent:
            return 'sell', f"Ожидаемое падение {diff_percent:.2f}% превышает порог {self.threshold_percent:.2f}%."

        return 'hold', "Движение цены недостаточно выражено."
