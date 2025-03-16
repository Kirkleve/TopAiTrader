import torch
import torch.nn as nn


class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout_rate=0.2):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Берём последний временной шаг

        if out.shape[0] > 1:  # BatchNorm работает только если батч > 1
            out = self.batch_norm(out)

        out = self.dropout(out)  # Dropout
        out = torch.nn.functional.leaky_relu(self.fc1(out))  # LeakyReLU
        out = self.fc2(out)
        return out
