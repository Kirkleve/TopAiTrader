from data.fetch_data import CryptoDataFetcher

fetcher = CryptoDataFetcher()
data = fetcher.fetch_historical_data_multitimeframe('BTC/USDT')

for timeframe, df in data.items():
    print(f"Таймфрейм: {timeframe}")
    print(df.tail())
