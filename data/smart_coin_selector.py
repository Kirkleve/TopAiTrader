from data.coin_selector import CoinSelector
from data.fetch_data import CryptoDataFetcher


class SmartCoinSelector:
    def __init__(self, sentiment_analyzer, trader):
        self.coin_selector = CoinSelector()
        self.sentiment_analyzer = sentiment_analyzer
        self.data_fetcher = CryptoDataFetcher()
        self.trader = trader

    def sentiment_filter(self, symbols, threshold=0.5):
        filtered_symbols = []
        for symbol in symbols:
            sentiment_result = self.sentiment_analyzer(symbol.split('/')[0])
            sentiment_score = (int(sentiment_result[0]['label'][0]) - 3) / 2
            if abs(sentiment_score) >= threshold:
                filtered_symbols.append(symbol)
        return filtered_symbols

    def trend_filter(self, symbols):
        filtered_symbols = []
        market_data = self.data_fetcher.fetch_historical_data(symbols)

        for symbol in symbols:
            df = market_data[symbol]
            ema_current = df['ema'].iloc[-1]
            price_current = df['close'].iloc[-1]

            if price_current > ema_current or price_current < ema_current:
                filtered_symbols.append(symbol)

        return filtered_symbols

    def historical_performance_filter(self, symbols, min_winrate=0.5):
        filtered_symbols = []
        for symbol in symbols:
            trades = self.trader.get_trade_history(symbol)
            if trades:
                wins = sum(1 for trade in trades if trade['profit'] > 0)
                winrate = wins / len(trades)
                if winrate >= min_winrate:
                    filtered_symbols.append(symbol)
        return filtered_symbols

    def select_best_coins(self, final_count=5):
        # Шаг 1: По объёму и волатильности (оставляем 30 монет)
        coins = self.coin_selector.select_coins_to_trade(top_n=30)

        # Шаг 2: sentiment-анализ (снижаем порог до 0.3)
        sentiment_coins = self.sentiment_filter(coins, threshold=0.3)
        if sentiment_coins:
            coins = sentiment_coins

        # Шаг 3: Тренд (опционально, можешь временно отключить)
        trend_coins = self.trend_filter(coins)
        if trend_coins:
            coins = trend_coins

        # Шаг 4: История успешных сделок
        performance_coins = self.historical_performance_filter(coins, min_winrate=0.4)
        if performance_coins:
            coins = performance_coins

        # Если после всех фильтров список пустой, вернем топ по волатильности
        if not coins:
            coins = self.coin_selector.top_by_volatility(self.coin_selector.top_by_volume(20), top_n=final_count)

        return coins[:final_count]
