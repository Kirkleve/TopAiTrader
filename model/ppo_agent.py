from stable_baselines3 import PPO
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PPOAgent:
    def __init__(self, env, model_dir='models', symbol='BTC_USDT'):
        self.env = env
        self.model_dir = model_dir
        self.symbol = symbol

        policy_kwargs = dict(
            net_arch=[512, 256, 128],
            activation_fn=nn.ReLU
        )

        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=0.0002,
            gamma=0.95,
            gae_lambda=0.95,
            ent_coef=0.001,
            clip_range=0.1,
            normalize_advantage=True,
            n_steps=4096,
            batch_size=512,
            policy_kwargs=policy_kwargs
        )

        self.scheduler = ReduceLROnPlateau(self.model.policy.optimizer, mode='max', patience=5, factor=0.5)
