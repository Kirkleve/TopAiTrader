from config import BYBIT_API_KEY, BYBIT_API_SECRET
from data.fetch_data import CryptoDataFetcherBybit
from data.cmc_data import MarketDataFetcher
import pandas as pd

class SmartCoinSelector:
    def __init__(self, sentiment_analyzer, trader):
        self.sentiment_analyzer = sentiment_analyzer
        self.data_fetcher = CryptoDataFetcherBybit(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
        self.market_fetcher = MarketDataFetcher()
        self.trader = trader

    def top_by_volume(self, top_n=20):
        market_data = self.market_fetcher.fetch_market_data()
        sorted_coins = sorted(
            market_data.items(),
            key=lambda item: item[1]['volume_24h'],
            reverse=True
        )
        return [f"{symbol}/USDT" for symbol, _ in sorted_coins[:top_n]]

    def top_by_volatility(self, symbols, top_n=10):
        atr_values = {}
        market_data = self.data_fetcher.fetch_historical_data_multi_timeframe(symbols)
        for symbol in symbols:
            df = market_data.get(symbol)
            if df is not None and 'atr' in df.columns and not df.empty:
                atr = df['atr'].iloc[-1]
                if not pd.isna(atr):
                    atr_values[symbol] = float(atr)

        sorted_by_atr = sorted(atr_values.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in sorted_by_atr[:top_n]]

    def sentiment_filter(self, symbols, threshold=0.3):
        filtered_symbols = []
        for symbol in symbols:
            sentiment_result = self.sentiment_analyzer(symbol.split('/')[0])
            sentiment_score = (int(sentiment_result[0]['label'][0]) - 3) / 2
            if abs(sentiment_score) >= threshold:
                filtered_symbols.append(symbol)
        return filtered_symbols

    def trend_filter(self, symbols):
        filtered_symbols = []
        market_data = self.data_fetcher.fetch_historical_data_multi_timeframe(symbols)
        for symbol in symbols:
            df = market_data.get(symbol)
            if df is not None and 'ema' in df.columns:
                ema_current = df['ema'].iloc[-1]
                price_current = df['close'].iloc[-1]
                if price_current > ema_current or price_current < ema_current:
                    filtered_symbols.append(symbol)
        return filtered_symbols

    def historical_performance_filter(self, symbols, min_winrate=0.4):
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
        coins = self.top_by_volatility(self.top_by_volume(30), top_n=20)

        sentiment_coins = self.sentiment_filter(coins)
        if sentiment_coins:
            coins = sentiment_coins

        trend_coins = self.trend_filter(coins)
        if trend_coins:
            coins = trend_coins

        performance_coins = self.historical_performance_filter(coins)
        if performance_coins:
            coins = performance_coins

        return coins[:final_count]
