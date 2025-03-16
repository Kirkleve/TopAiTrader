import ccxt
import pandas as pd
from ta.momentum import RSIIndicator, WilliamsRIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator, CCIIndicator, MassIndex
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, MFIIndicator
from config import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET

class CryptoDataFetcher:
    def __init__(self):
        self.exchange = ccxt.binanceusdm({
            'apiKey': BINANCE_TESTNET_API_KEY,
            'secret': BINANCE_TESTNET_API_SECRET,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        self.exchange.set_sandbox_mode(True)
        self.exchange.options['adjustForTimeDifference'] = True

    def fetch_historical_data_multi_timeframe(self, symbol, timeframes=None, limit=200):
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']

        market_data = {}

        for timeframe in timeframes:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                # Индикаторы:
                df['rsi'] = RSIIndicator(df['close']).rsi()
                df['ema'] = EMAIndicator(df['close']).ema_indicator()
                macd = MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()

                bollinger = BollingerBands(df['close'])
                df['bollinger_high'] = bollinger.bollinger_hband()
                df['bollinger_low'] = bollinger.bollinger_lband()
                df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()

                df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
                df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

                # Добавляем новые индикаторы
                df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
                df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
                df['momentum'] = df['close'].diff()
                df['mfi'] = MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
                df['mass_index'] = MassIndex(df['high'], df['low']).mass_index()

                df.dropna(inplace=True)
                market_data[timeframe] = df

            except Exception as e:
                print(f"⚠️ Ошибка получения данных {symbol} [{timeframe}]: {e}")
                market_data[timeframe] = pd.DataFrame()

        return market_data


if __name__ == "__main__":
    fetcher = CryptoDataFetcher()
    symbol = "BTC/USDT"  # Можно поменять на другую пару
    timeframes = ["1h"]  # Можно проверить другие таймфреймы

    print(f"📊 Загружаем данные для {symbol}...")
    data = fetcher.fetch_historical_data_multi_timeframe(symbol, timeframes)

    if not data or "1h" not in data or data["1h"].empty:
        print("❌ Ошибка: данные не загружены!")
    else:
        print("✅ Данные загружены успешно!")
        print(data["1h"].head())  # Выведем первые строки данных
