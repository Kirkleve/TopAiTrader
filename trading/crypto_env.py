import numpy as np
import gymnasium as gym
from gymnasium import spaces
from data.fear_and_greed import FearGreedIndexFetcher
from trading.crypto_env_utils import precompute_model_predictions, calculate_reward, get_observation


class CryptoTradingEnv(gym.Env):
    def __init__(self, symbol, data, sentiment_scores, lstm_models, cnn_models,
                 xgb_model, xgb_scaler_X, xgb_scaler_y, initial_balance=1000,
                 trading_fee=0.1, sl_multiplier=2, tp_multiplier=3):
        super().__init__()

        self.symbol = symbol
        self.data = data
        self.sentiment_scores = sentiment_scores
        self.lstm_models = lstm_models
        self.cnn_models = cnn_models
        self.xgb_model = xgb_model
        self.scaler_X = xgb_scaler_X
        self.scaler_y = xgb_scaler_y
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trading_fee = trading_fee
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier

        self.position = None
        self.position_price = None
        self.position_open_step = None
        self.current_step = 20
        self.pnl = 0.0

        obs_len = 11 + len(self.lstm_models) + len(self.cnn_models) + 10
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_len,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        min_step = 20
        max_step = len(self.data) - 150
        self.current_step = np.random.randint(min_step, max_step) if max_step > min_step else 20
        self.balance = self.initial_balance
        self.position = None
        self.position_price = None
        self.position_open_step = None
        self.pnl = 0.0

        fear_greed_index, _ = FearGreedIndexFetcher.fetch_current_index()
        self.fear_greed_scaled = fear_greed_index / 100 if fear_greed_index else 0.5
        precompute_model_predictions(self)

        return self._get_observation(), {}

    def step(self, action):
        current_price, atr, adx = self.data[self.current_step][0], self.data[self.current_step][-1], \
                                  self.data[self.current_step][-2]
        reward = calculate_reward(self, action, current_price, atr, adx)

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        obs = self._get_observation()
        return obs, reward, terminated, False, {}

    def _get_observation(self):
        return get_observation(self)

    def check_observations(self):
        print("🔍 Запускаю предварительную проверку наблюдений...")

        for step in range(20, len(self.data) - 150):
            self.current_step = step
            obs = self._get_observation()
            if np.isnan(obs).any() or np.isinf(obs).any():
                print(f"🚨 Ошибка данных на шаге {step}: обнаружены NaN или Inf!")
                return False

        print("✅ Предварительная проверка пройдена успешно!")
        return True

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}, PNL: {self.pnl:.2f}")
