import os
from stable_baselines3 import PPO

from logger_config import setup_logger

logger = setup_logger()


class PPOModelManager:
    def __init__(self, symbol):
        self.symbol = symbol.replace('/', '_')
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_dir = os.path.join(self.base_dir, "models", self.symbol, "ppo")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, f"{self.symbol}_ppo_agent.zip")

    def model_exists(self):
        return os.path.exists(self.model_path)

    def save_model(self, model):
        model.save(self.model_path)
        logger.info(f"✅ PPO-модель сохранена: {self.model_path}")

    def load_model(self, env=None):
        logger.info(f"🔎 Ищу PPO-модель по пути: {self.model_path}")
        if self.model_exists():
            model = PPO.load(self.model_path, env=env)
            logger.info(f"✅ PPO-модель загружена: {self.model_path}")
            return model
        else:
            logger.info(f"⚠️ PPO-модель не найдена по пути: {self.model_path}")
            return None
