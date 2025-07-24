import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from model.news_sentiment import SentimentAnalyzer
from trading.crypto_env import CryptoTradingEnv
from model.ppo_agent import PPOAgent
from data.data_preparation import DataPreparation
from data.fear_and_greed import FearGreedIndexFetcher


class PPOTrainer:
    def __init__(self, symbol, combined_data, dates, state_manager,
                 model_manager, sentiment_scores=None,
                 scaler_type='robust', sentiment_model="finbert"):

        self.symbol = symbol.replace('/', '_')
        self.combined_data = combined_data
        self.dates = dates
        self.state_manager = state_manager
        self.model_manager = model_manager
        self.features = state_manager.features
        self.observation_scaler = state_manager.scaler_dict.get('observation_scaler', DataPreparation(symbol, state_manager.features, scaler_type=scaler_type).get_scaler())

        if sentiment_scores is None:
            self.sentiment_analyzer = SentimentAnalyzer(model_type=sentiment_model)
            historical_sentiment = self.sentiment_analyzer.get_historical_sentiment_scores(
                dates, fetch_news_for_date=lambda date: ["Example news"]
            )
            historical_fg_scores = FearGreedIndexFetcher.get_historical_fg_scores(dates)

            self.sentiment_scores = {
                'historical': historical_sentiment,
                'fear_greed': historical_fg_scores
            }
        else:
            self.sentiment_scores = sentiment_scores

    def prepare_state_data(self):
        state_data = []

        for current_step in range(20, len(self.combined_data['1h'])):
            state_row = self.state_manager.create_state(self.combined_data, current_step)
            state_data.append(state_row)

        state_data_scaled = self.observation_scaler.fit_transform(state_data)
        if np.isnan(state_data_scaled).any() or np.isinf(state_data_scaled).any():
            raise ValueError("❌ В наблюдениях обнаружены NaN или Inf. Обучение остановлено.")

        return state_data_scaled

    def create_env(self, state_data_scaled):
        return make_vec_env(lambda: CryptoTradingEnv(
            symbol=self.symbol,
            data=state_data_scaled,
            df_original_dict=self.combined_data,
            sentiment_scores=self.sentiment_scores['historical'],
            lstm_models=self.state_manager.lstm_models,
            np_models=self.state_manager.np_models,
            xgb_model=self.state_manager.xgb_model,
            xgb_scaler_X=self.state_manager.xgb_scaler_X,
            xgb_scaler_y=self.state_manager.xgb_scaler_y,
            observation_scaler=self.observation_scaler
        ), n_envs=1)

    def train(self, existing_model=None, eval_freq=8192, timesteps_per_epoch=16384, total_epochs=20):
        state_data_scaled = self.prepare_state_data()
        env = self.create_env(state_data_scaled)
        eval_env = self.create_env(state_data_scaled)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.model_manager.model_dir, 'best_model'),
            eval_freq=eval_freq,
            deterministic=True,
            verbose=1
        )

        log_path = os.path.join(self.model_manager.model_dir, 'ppo_logs')
        new_logger = configure(log_path, ["stdout", "csv"])

        agent = PPOAgent(
            env=env,
            model_dir=self.model_manager.model_dir,
            symbol=self.symbol
        )

        agent.model.set_logger(new_logger)

        if existing_model:
            agent.model = PPO.load(existing_model, env=env)

        for epoch in range(total_epochs):
            print(f"🚀 Эпоха обучения {epoch + 1}/{total_epochs}")
            agent.model.learn(total_timesteps=timesteps_per_epoch, callback=eval_callback)

            try:
                avg_reward = np.mean([info['r'] for info in agent.model.ep_info_buffer])
            except KeyError:
                avg_reward = 0

            agent.scheduler.step(avg_reward)
            print(f"📉 Learning rate после эпохи {epoch + 1}: {agent.scheduler.optimizer.param_groups[0]['lr']}")

        # Сохранение модели через менеджер
        self.model_manager.save_model(agent.model)

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
                reward = rewards if isinstance(rewards, (int, float)) else rewards[0]
                done = dones if isinstance(dones, bool) else dones[0]
                total_reward += reward

            total_rewards.append(total_reward)
            if total_reward > 0:
                win_count += 1

        avg_reward = np.mean(total_rewards)
        win_rate = (win_count / episodes) * 100

        print(f"\n📊 Оценка PPO-агента:")
        print(f"✅ Средняя прибыль: {avg_reward:.2f}")
        print(f"✅ Процент прибыльных сделок: {win_rate:.2f}%")

        return avg_reward, win_rate
