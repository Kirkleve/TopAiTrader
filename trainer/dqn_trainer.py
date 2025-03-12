import torch
from trading.agent import DQNAgent
from trading.crypto_env import CryptoTradingEnv

class DQNTrainer:
    def __init__(self, state_data, sentiment_scores, epochs=50):
        self.state_data = state_data
        self.sentiment_scores = sentiment_scores
        self.epochs = epochs

    def train_and_save(self, symbol):
        env = CryptoTradingEnv(self.state_data, self.sentiment_scores)

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n if hasattr(env.action_space, 'n') else 3

        agent = DQNAgent(state_size=state_size, action_size=action_size)

        overall_profit = 0  # Общая прибыль по всем эпизодам

        for episode in range(self.epochs):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done, total_reward = False, 0

            while not done:
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                agent.remember(state, action, reward, next_state_tensor, done)
                state = next_state_tensor

                total_reward += reward

                if len(agent.memory) > 32:
                    agent.replay(32)

            overall_profit += total_reward

            print(
                f"🚀 Эпизод [{episode + 1}/{self.epochs}] - Прибыль эпизода: {total_reward:.2f}%, PNL эпизода: {overall_profit:.2f}%, Общая прибыль: {overall_profit:.2f}%")

        # Итоговая общая прибыль по всем эпизодам
        print(f"\n✅ Итоговая общая прибыль за {self.epochs} эпизодов: {overall_profit:.2f}%")

        torch.save(agent.model.state_dict(), f'trading/trained_agent_{symbol.replace("/", "_")}.pth')

