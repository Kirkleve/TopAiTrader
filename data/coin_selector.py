from data.cmc_data import MarketDataFetcher
from data.fetch_data import CryptoDataFetcher
import pandas as pd

class CoinSelector:
    def __init__(self):
        self.market_fetcher = MarketDataFetcher()
        self.data_fetcher = CryptoDataFetcher()

    def top_by_volume(self, top_n=10):
        """Выбираем топ-монеты по объёму торгов за 24 часа"""
        market_data = self.market_fetcher.fetch_market_data()
        sorted_coins = sorted(
            market_data.items(),
            key=lambda item: item[1]['volume_24h'],
            reverse=True
        )
        return [f"{symbol}/USDT" for symbol, _ in sorted_coins[:top_n]]

    def top_by_volatility(self, symbols, top_n=10):
        """Выбираем топ монет по волатильности (ATR)"""
        atr_values = {}
        for symbol in symbols:
            data = self.data_fetcher.fetch_historical_data_multi_timeframe([symbol])[symbol]
            if data is not None and 'atr' in data.columns:
                atr = data['atr'].iloc[-1]
                if not pd.isna(atr):
                    atr_values[symbol] = float(atr)

        sorted_by_atr = sorted(atr_values.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in sorted_by_atr[:top_n]]

    def select_coins_to_trade(self, top_n=10):
        """Комбинируем оба подхода (объем и волатильность)"""
        top_volume_symbols = self.top_by_volume(top_n=50)  # топ-50 по объёму
        top_volatility_symbols = self.top_by_volatility(top_volume_symbols, top_n=top_n)

        return top_volatility_symbols
