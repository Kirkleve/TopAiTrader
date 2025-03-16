import os
import matplotlib.pyplot as plt
import mplfinance as mpf
from data.fetch_data import CryptoDataFetcher


class CNNDataPreprocessor:
    def __init__(self, save_dir='cnn_images'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def generate_candlestick_images(self, data_multi, symbol, window_size=20):
        symbol_fmt = symbol.replace('/', '_')

        for timeframe, df in data_multi.items():
            df = df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]
            image_folder = os.path.join(self.save_dir, f"{symbol_fmt}_{timeframe}")
            os.makedirs(image_folder, exist_ok=True)

            existing_images = len(os.listdir(image_folder))
            required_images = len(df) - window_size

            if existing_images >= required_images:
                continue

            print(f"🚀 Генерация изображений для {symbol} [{timeframe}]...")

            for idx in range(window_size, len(df)):
                data_window = df.iloc[idx - window_size:idx]

                fig, ax = plt.subplots(figsize=(1.28, 1.28), dpi=100)
                mpf.plot(data_window, type='candle', ax=ax, style='charles', axisoff=True, tight_layout=True)

                image_path = os.path.join(image_folder, f"{idx}.png")
                fig.savefig(image_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            print(f"✅ Изображения свечей для {symbol} [{timeframe}] сохранены в {image_folder}")

    def check_and_prepare_images(self, symbol, timeframes, window_size=20):
        fetcher = CryptoDataFetcher()
        data_multi = fetcher.fetch_historical_data_multi_timeframe(symbol, timeframes)
        self.generate_candlestick_images(data_multi, symbol, window_size=window_size)


if __name__ == "__main__":
    symbol = "BTC/USDT"
    timeframes = ["15m", "1h", "4h", "1d"]
    window_size = 20

    preprocessor = CNNDataPreprocessor()
    preprocessor.check_and_prepare_images(symbol, timeframes, window_size)