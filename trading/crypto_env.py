import numpy as np
import gymnasium as gym
from gymnasium import spaces
from data.fear_and_greed import FearGreedIndexFetcher
from trading.crypto_env_utils import precompute_model_predictions, calculate_reward, get_observation


class CryptoTradingEnv(gym.Env):
    def __init__(self, symbol, data, df_original_dict, sentiment_scores, lstm_models, np_models,
                 xgb_model, xgb_scaler_X, xgb_scaler_y, observation_scaler,
                 initial_balance=1000, trading_fee=0.1, sl_multiplier=2, tp_multiplier=3):

        super().__init__()

        self.symbol = symbol
        self.data = data
        self.df_original_dict = df_original_dict
        self.sentiment_scores = sentiment_scores
        self.lstm_models = lstm_models
        self.np_models = np_models
        self.xgb_model = xgb_model
        self.scaler_X = xgb_scaler_X
        self.scaler_y = xgb_scaler_y
        self.observation_scaler = observation_scaler
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trading_fee = trading_fee
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.consecutive_profitable_trades = 0
        self.pyramid_count = 0
        self.feature_names = [
            'close', 'rsi', 'ema', 'adx', 'atr', 'volume',
            'cci', 'williams_r', 'momentum', 'mfi', 'mass_index'
        ]

        self.position = None
        self.position_price = None
        self.position_open_step = None
        self.current_step = 20
        self.pnl = 0.0

        obs_len = data.shape[1]  # автоматически берём из shape нормализованного state_data
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_len,), dtype=np.float32)

    def adapt_strategy_parameters(self):
        # Получаем текущие значения метрик
        adx = self.data[self.current_step, self.feature_names.index('adx')]
        atr = self.data[self.current_step, self.feature_names.index('atr')]
        sentiment = self.sentiment_scores[self.current_step]

        # Индекс страха и жадности
        fg_index, _ = FearGreedIndexFetcher.fetch_current_index()
        fg_scaled = fg_index / 100 if fg_index else 0.5

        # Направление торговли на основе сантимента
        if sentiment > 0.2:
            self.trade_bias = 'long'
        elif sentiment < -0.2:
            self.trade_bias = 'short'
        else:
            self.trade_bias = 'neutral'

        # Адаптация риска на основе индекса страха и жадности
        self.risk_multiplier = 1 + (fg_scaled - 0.5)

        # Адаптация размера TP и SL на основе ATR и ADX
        if adx > 50:
            self.dynamic_take_profit = atr * 3.0
            self.dynamic_stop_loss = atr * 1.5
        elif adx > 30:
            self.dynamic_take_profit = atr * 2.5
            self.dynamic_stop_loss = atr * 1.25
        else:
            self.dynamic_take_profit = atr * 2.0
            self.dynamic_stop_loss = atr

        # Адаптация размера позиции
        base_risk = 0.01
        self.position_size = (base_risk * self.balance * self.risk_multiplier)

        # Ограничения по риску и размеру позиции
        self.position_size = np.clip(self.position_size, self.balance * 0.005, self.balance * 0.05)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        min_step = 20
        max_step = len(self.data) - 150

        # Безопасно проверяем и выбираем начальную точку
        if max_step > min_step:
            self.current_step = np.random.randint(min_step, max_step)
        else:
            # если данных мало, берём безопасную точку
            self.current_step = min_step if len(self.data) > min_step else 0

        self.balance = self.initial_balance
        self.position = None
        self.position_price = None
        self.position_open_step = None
        self.pnl = 0.0
        self.consecutive_profitable_trades = 0
        self.pyramid_count = 0
        precompute_model_predictions(self)
        self.adapt_strategy_parameters()

        return self._get_observation(), {}

    def step(self, action):
        # 👇 адаптация параметров стратегии
        self.adapt_strategy_parameters()

        current_price, atr, adx = (
            self.data[self.current_step][0],
            self.data[self.current_step][-1],
            self.data[self.current_step][-2]
        )

        reward = calculate_reward(self, action, current_price, atr, adx)

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        obs = self._get_observation()

        return obs, reward, terminated, False, {}

    def _get_observation(self):
        obs = get_observation(self)
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"🚨 NaN или Inf в наблюдениях на шаге {self.current_step}. Заменяю на нули.")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

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
