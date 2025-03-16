import os

from stable_baselines3 import PPO

class PPOModelManager:
    def __init__(self, symbol):
        self.symbol = symbol.replace('/', '_')
        self.model_dir = os.path.join("models", self.symbol, "ppo")
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model):
        model_path = os.path.join(self.model_dir, f"{self.symbol}_ppo_agent.zip")
        model.save(model_path)
        print(f"✅ PPO-модель сохранена: {model_path}")

    def load_model(self, env=None):
        model_path = os.path.join(self.model_dir, f"{self.symbol}_ppo_agent.zip")
        print(f"🔎 Ищу PPO-модель по пути: {model_path}")  # <-- добавь для отладки
        if os.path.exists(model_path):
            model = PPO.load(model_path, env=env)
            print(f"✅ PPO-модель загружена: {model_path}")
            return model
        else:
            print(f"⚠️ PPO-модель не найдена по пути: {model_path}")
            return None

