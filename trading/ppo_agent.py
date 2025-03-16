from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np

class PPOAgent:
    def __init__(self, env, model_dir='models', symbol='BTC_USDT', total_epochs=10):
        self.env = env
        self.model_dir = model_dir
        self.symbol = symbol
        self.total_epochs = total_epochs

        os.makedirs(self.model_dir, exist_ok=True)

        # Простая и мощная архитектура
        policy_kwargs = dict(
            net_arch=[512, 256, 128],  # Оптимальная архитектура для быстрого обучения
            activation_fn=nn.ReLU  # Простая активация для быстрых вычислений
        )

        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=1e-4,  # Стандартная начальная скорость обучения
            gamma=0.99,  # Умеренный коэффициент дисконтирования
            gae_lambda=0.95,  # Стабильное значение для GAE
            ent_coef=0.005,  # Умеренная энтропийная регуляризация для баланса
            clip_range=0.1,  # Стандартный диапазон для стабильного обучения
            normalize_advantage=True,
            n_steps=2048,  # Оптимальное количество шагов для стабильности
            batch_size=64,  # Достаточный размер пакета для быстрой сходимости
            policy_kwargs=policy_kwargs
        )

        # Умеренное снижение learning rate
        self.scheduler = ReduceLROnPlateau(self.model.policy.optimizer, 'min', patience=5, factor=0.5)

    def train(self, timesteps_per_epoch=10000, eval_freq=1000):
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=os.path.join(self.model_dir, 'best_model'),
            log_path=os.path.join(self.model_dir, 'eval_logs'),
            eval_freq=eval_freq,
            deterministic=True,
            verbose=1
        )

        checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=self.model_dir)

        for epoch in range(self.total_epochs):
            print(f"🚀 Запускаю эпоху {epoch + 1}/{self.total_epochs}")
            self.model.learn(total_timesteps=timesteps_per_epoch, callback=[eval_callback, checkpoint_callback])

            # Получаем среднюю награду за эпизоды для использования в качестве метрики
            avg_reward = np.mean([info['episode']['reward'] for info in self.model.ep_info_buffer])

            # Подаем метрику для обновления learning rate
            self.scheduler.step(avg_reward)  # Передаем среднюю награду как метрику

            print(f"📉 Learning rate после эпохи {epoch + 1}: {self.scheduler.get_last_lr()[0]}")

    def save(self, filename="ppo_final"):
        path = os.path.join(self.model_dir, f'{self.symbol}_{filename}.zip')
        self.model.save(path=path)
        print(f"✅ PPO-модель сохранена по пути: {path}")

    def load(self, filepath):
        if os.path.exists(filepath):
            self.model = PPO.load(filepath, env=self.env)
            print(f"✅ PPO-модель загружена: {filepath}")
        else:
            print(f"⚠️ PPO-модель не найдена по пути: {filepath}")
