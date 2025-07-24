import json
import os
import shutil

class CoinManager:
    def __init__(self, trader, market_data_fetcher, max_coins=5):
        self.trader = trader
        self.market_data_fetcher = market_data_fetcher
        self.max_coins = max_coins
        self.coins_file = 'coins.json'
        self.current_coins = self.load_coins()

    def load_coins(self):
        if os.path.exists(self.coins_file):
            with open(self.coins_file, 'r') as f:
                return json.load(f)
        return []

    def save_coins(self):
        with open(self.coins_file, 'w') as f:
            json.dump(self.current_coins, f, indent=4)

    def get_current_coins(self):
        return self.current_coins

    def add_coin(self, coin):
        coin = coin.upper() + '/USDT'
        if coin in self.current_coins:
            return f"⚠️ Монета {coin} уже добавлена."

        if self.coin_limit_reached():
            return f"⚠️ Достигнут максимум монет ({self.max_coins}), удали старую перед добавлением новой."

        self.current_coins.append(coin)
        self.save_coins()
        return f"✅ Монета {coin} добавлена."

    def replace_coin(self, old_coin, new_coin):
        old_coin = old_coin.upper() + '/USDT'
        new_coin = new_coin.upper() + '/USDT'

        if old_coin not in self.current_coins:
            return f"⚠️ Монета {old_coin} не найдена в списке."

        index = self.current_coins.index(old_coin)
        self.current_coins[index] = new_coin
        self.save_coins()

        # Удаление старых моделей
        self.remove_models(old_coin)

        return f"✅ Монета {old_coin} заменена на {new_coin}."

    def remove_models(self, symbol):
        symbol_dir = symbol.replace('/', '_')

        model_paths = [
            os.path.join('models', symbol_dir, 'lstm'),
            os.path.join('models', symbol_dir, 'neuralprophet'),
            os.path.join('models', symbol_dir, 'xgb'),
            os.path.join('models', symbol_dir, 'ppo')
        ]

        for path in model_paths:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                print(f"🗑️ Удалены модели в папке: {path}")

    def coin_limit_reached(self):
        return len(self.current_coins) >= self.max_coins
