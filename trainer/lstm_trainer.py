import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from model.lstm_price_predictor import LSTMPricePredictor

class LSTMTrainer:
    def __init__(self, features, epochs=50, seq_length=20):
        self.scaler = None
        self.features = features
        self.epochs = epochs
        self.seq_length = seq_length

    def prepare_data(self, df):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[self.features])

        X, y = [], []
        for i in range(self.seq_length, len(scaled_data)):
            X.append(scaled_data[i - self.seq_length:i])
            y.append(scaled_data[i, 0])

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)  # Исправлено и быстро!
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

        return X_tensor, y_tensor, scaler

    def train_and_save(self, df, symbol, timeframe):
        X_tensor, y_tensor, scaler = self.prepare_data(df)
        model = LSTMPricePredictor(input_size=len(self.features))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(self.epochs):
            model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()

        model_dir = f'trainer/models/{symbol.replace("/", "_")}'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{model_dir}/{symbol.replace('/', '_')}_{timeframe}_lstm.pth")
        self.scaler = scaler

        return model, scaler

    def load_model_and_scaler(self, symbol, timeframe='1h'):
        symbol_dir = symbol.replace('/', '_')
        model_path = f'trainer/models/{symbol_dir}/{symbol_dir}_{timeframe}_lstm.pth'

        model = LSTMPricePredictor(input_size=len(self.features))
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # scaler пока заново создаётся, либо загружай сохранённый scaler (если сохранял)
        scaler = MinMaxScaler()
        return model, scaler

