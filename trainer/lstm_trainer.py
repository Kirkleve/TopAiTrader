import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from model.lstm_price_predictor import LSTMPricePredictor
from trainer.model_manager.lstm_model import LSTMModelManager

class LSTMTrainer:
    def __init__(self, symbol, features, timeframe, epochs=50, seq_length=20, device='cpu'):
        self.symbol = symbol.replace('/', '_')
        self.features = features
        self.timeframe = timeframe
        self.epochs = epochs
        self.seq_length = seq_length
        self.device = device

        self.model_manager = LSTMModelManager(self.symbol, self.features)
        self.model = LSTMPricePredictor(input_size=len(features)).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )

    def prepare_data(self, df):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[self.features])

        X, y = [], []
        for i in range(self.seq_length, len(scaled_data)):
            X.append(scaled_data[i - self.seq_length:i])
            y.append(scaled_data[i, 0])

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        return X_tensor, y_tensor, scaler

    def train(self, df, batch_size=64):
        X_tensor, y_tensor, scaler = self.prepare_data(df)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = LSTMPricePredictor(input_size=len(self.features)).to(self.device)

        best_loss = float('inf')
        total_loss = 0.0
        total_lr = 0.0

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                output = model(batch_X)
                loss = self.loss_fn(output, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            total_loss += avg_loss

            self.scheduler.step(avg_loss)
            total_lr += self.scheduler.optimizer.param_groups[0]['lr']

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.model_manager.save_model(model, self.timeframe)
                self.model_manager.save_scaler(scaler, self.timeframe)

        avg_total_loss = total_loss / self.epochs
        avg_lr = total_lr / self.epochs

        print(f"🏅 Средний Loss за обучение: {avg_total_loss:.4f}")
        print(f"📉 Средний Learning Rate за обучение: {avg_lr:.6f}")

        model = self.model_manager.load_model(self.timeframe)
        scaler = self.model_manager.load_scaler(self.timeframe)

        return model, scaler