import gymnasium as gym
import numpy as np


class CryptoTradingEnv(gym.Env):
    def __init__(self, data, sentiment_scores, initial_balance=1000):
        super().__init__()
        self.data = data
        self.sentiment_scores = sentiment_scores
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.position_price = 0
        self.current_step = 0

        self.action_space = gym.spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        obs_shape = data.shape[1]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = None
        self.position_price = 0
        observation = self.data[self.current_step]
        return observation, {}

    def step(self, action):
        reward = 0
        current_price = self.data[self.current_step][0]

        if action == 1:  # BUY
            if self.position is None:
                self.position = 'long'
                self.position_price = current_price
                reward = 0  # открытие позиции, пока без награды
            else:
                reward = -0.1  # штраф за повторное открытие

        elif action == 2:  # sell
            if self.position == 'long' and self.position_price != 0:
                reward = ((current_price - self.position_price) / self.position_price) * 100
                self.balance += reward
                self.position = None
                self.position_price = 0
            else:
                reward = -0.1  # штраф за продажу без позиции

        elif action == 0:  # hold
            reward = -0.01  # штраф за бездействие (маленький)

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        observation = self.data[self.current_step]

        return observation, reward, terminated, truncated, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")

