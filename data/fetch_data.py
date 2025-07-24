import ccxt
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator, VolumeWeightedAveragePrice
from ta.momentum import WilliamsRIndicator
from ta.trend import CCIIndicator, MassIndex


class CryptoDataFetcherBybit:
    def __init__(self, api_key, api_secret):
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })

    def fetch_historical_data_multi_timeframe(self, symbol, timeframes=None, n_bars=1000):
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']

        market_data = {}

        for timeframe in timeframes:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=n_bars)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)

                # Убедимся, что индекс без NaT
                df = df[~df.index.isna()]

                # Проверка количества строк перед расчётом индикаторов
                if len(df) < 30:
                    print(f"⚠️ Недостаточно данных ({len(df)} строк) для расчёта индикаторов {symbol} [{timeframe}]")
                    market_data[timeframe] = pd.DataFrame()
                    continue

                # Технические индикаторы
                df['rsi'] = RSIIndicator(df['close']).rsi()
                df['ema'] = EMAIndicator(df['close']).ema_indicator()

                macd = MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()

                bollinger = BollingerBands(df['close'])
                df['bollinger_high'] = bollinger.bollinger_hband()
                df['bollinger_low'] = bollinger.bollinger_lband()

                df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'],
                                                        df['volume']).volume_weighted_average_price()

                df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
                df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

                df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
                df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
                df['momentum'] = df['close'].diff()
                df['mfi'] = MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
                df['mass_index'] = MassIndex(df['high'], df['low']).mass_index()

                df.dropna(inplace=True)

                if df.empty:
                    print(f"⚠️ После очистки нет данных для {symbol} [{timeframe}], пропускаем.")
                    market_data[timeframe] = pd.DataFrame()
                    continue

                market_data[timeframe] = df

            except Exception as e:
                print(f"⚠️ Ошибка получения данных {symbol} [{timeframe}]: {e}")
                market_data[timeframe] = pd.DataFrame()

        return market_data

