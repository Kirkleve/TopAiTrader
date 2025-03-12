import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random  # добавлен импорт random!
from collections import deque  # добавлен импорт deque!

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  # ← это важно!
        self.memory = deque(maxlen=2000)  # память для опыта
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 0.001

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)

        state = state.clone().detach().unsqueeze(0).float()
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = state.clone().detach().float()
            next_state = next_state.clone().detach().float()

            current_q = self.model(state.unsqueeze(0))
            target_q = current_q.clone().detach()

            if done:
                target_q[0][action] = reward
            else:
                next_max_q = torch.max(self.model(next_state.unsqueeze(0)))
                target_q[0][action] = reward + self.gamma * next_max_q

            self.optimizer.zero_grad()
            loss = self.loss_fn(current_q, target_q)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
