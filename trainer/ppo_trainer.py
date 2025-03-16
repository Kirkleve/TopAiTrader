import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from trading.crypto_env import CryptoTradingEnv
from stable_baselines3.common.logger import configure

from trading.ppo_agent import PPOAgent


class PPOTrainer:
    def __init__(self, symbol, state_data, sentiment_scores, lstm_models,
                 cnn_model, cnn_scaler, xgb_model, xgb_scaler_X, xgb_scaler_y, model_manager, episodes=100000):
        self.symbol = symbol.replace('/', '_')
        self.state_data = state_data
        self.sentiment_scores = sentiment_scores
        self.lstm_models = lstm_models
        self.cnn_model = cnn_model
        self.cnn_scaler = cnn_scaler
        self.xgb_model = xgb_model
        self.xgb_scaler_X = xgb_scaler_X
        self.xgb_scaler_y = xgb_scaler_y
        self.model_manager = model_manager
        self.episodes = episodes

    def create_env(self):
        """Создаем среду для обучения и оценки."""
        return make_vec_env(lambda: CryptoTradingEnv(
            symbol=self.symbol,
            data=self.state_data,
            sentiment_scores=self.sentiment_scores,
            lstm_models=self.lstm_models,
            cnn_models=self.cnn_model,
            xgb_model=self.xgb_model,
            xgb_scaler_X=self.xgb_scaler_X,
            xgb_scaler_y=self.xgb_scaler_y,
        ), n_envs=1)

    def train(self, existing_model=None, eval_freq=10000, timesteps_per_epoch=5000, total_epochs=20):
        # Создание обучающей и оценочной среды
        env = self.create_env()
        eval_env = self.create_env()

        # Проверка наблюдений
        if not env.env_method('check_observations')[0]:
            raise ValueError("🚨 Ошибка в наблюдениях, обучение PPO остановлено!")
        else:
            print("✅ Наблюдения корректны. Запускаем PPO-обучение!")

        # Создание колбэков для оценки
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.model_manager.model_dir, 'best_model'),
            log_path=os.path.join(self.model_manager.model_dir, 'eval_logs'),
            eval_freq=eval_freq,
            deterministic=True,
            verbose=1
        )

        # Логирование (без tensorboard)
        log_path = os.path.join(self.model_manager.model_dir, 'ppo_logs')
        new_logger = configure(log_path, ["stdout", "csv"])

        # Создание PPO агента
        agent = PPOAgent(
            env=env,
            model_dir=self.model_manager.model_dir,
            symbol=self.symbol,
            total_epochs=total_epochs
        )

        agent.model.set_logger(new_logger)

        # Загрузка существующей модели, если есть
        if existing_model:
            agent.model = PPO.load(existing_model, env=env)

        # Обучение модели
        for epoch in range(agent.total_epochs):
            print(f"🚀 Эпоха обучения {epoch + 1}/{agent.total_epochs}")
            agent.model.learn(total_timesteps=timesteps_per_epoch * 2, callback=eval_callback)

            # Проверка и печать содержимого ep_info_buffer
            print("🔍 Содержимое ep_info_buffer:", agent.model.ep_info_buffer)

            # Получаем среднюю награду за эпизоды для использования в качестве метрики
            # Ожидаем, что структура данных отличается от предполагаемой, проверим и извлечем правильную награду
            try:
                avg_reward = np.mean(
                    [info['reward'] for info in agent.model.ep_info_buffer])  # Используем 'reward' вместо 'episode'
            except KeyError:
                avg_reward = 0  # В случае отсутствия ключа 'reward', используем 0

            # Подаем метрику для обновления learning rate
            agent.scheduler.step(avg_reward)  # Передаем метрику (средняя награда)

            print(f"📉 Learning rate после эпохи {epoch + 1}: {agent.scheduler.get_last_lr()[0]}")

        # Сохранение модели после обучения
        agent.save()

        return agent.model, env

    def evaluate_agent(self, agent, env, episodes=50):
        total_rewards = []
        win_count = 0

        for episode in range(episodes):
            obs = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                reward = rewards[0]
                done = dones[0]
                total_reward += reward

            total_rewards.append(total_reward)
            if total_reward > 0:
                win_count += 1

        avg_reward = np.mean(total_rewards)
        win_rate = (win_count / episodes) * 100

        print(f"\n📊 Оценка PPO-агента:")
        print(f"✅ Средняя прибыль: {avg_reward:.2f}")
        print(f"✅ Процент прибыльных сделок: {win_rate:.2f}%")

        return agent, env
