import os
import pickle

import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from data.fetch_data import CryptoDataFetcher
from model.CNN.cnn_dataloader import get_dataloader
from model.CNN.cnn_price_predictor import CNNPricePredictor
from trainer.model_manager.cnn_model import CNNModelManager


class CNNTrainer:
    def __init__(self, symbol, timeframe, epochs=10, device='cpu'):
        self.symbol = symbol.replace('/', '_')
        self.timeframe = timeframe
        self.epochs = epochs
        self.device = device

        self.model_manager = CNNModelManager(self.symbol)
        self.model = CNNPricePredictor().to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

    def prepare_labels(self):
        fetcher = CryptoDataFetcher()
        df = fetcher.fetch_historical_data_multi_timeframe(
            self.symbol.replace('_', '/'), [self.timeframe]
        )[self.timeframe]

        labels = df['close'].shift(-1).iloc[20:].dropna().values.reshape(-1, 1)
        scaler = MinMaxScaler()
        labels_scaled = scaler.fit_transform(labels).flatten()

        scaler_path = os.path.join(self.model_manager.model_dir, f"scaler_{self.timeframe}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        self.scaler = scaler

        return labels_scaled

    def train(self, image_dir, batch_size=32):
        global epoch
        print(f"🚀 CNN обучение [{self.timeframe}]...")

        labels = self.prepare_labels()
        dataloader = get_dataloader(image_dir=image_dir, labels=labels, batch_size=batch_size)

        best_loss = float('inf')

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for images, labels_batch in dataloader:
                images, labels_batch = images.to(self.device), labels_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images).squeeze()
                loss = self.loss_fn(outputs, labels_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.model_manager.save_model(self.model, self.timeframe)

        print(f"✅ CNN-модель сохранена: models\\{self.symbol}\\cnn\\{self.symbol}_{self.timeframe}_cnn.pth")
        print(f"🏅 Лучший Loss за обучение: {best_loss:.4f}")

        self.model = self.model_manager.load_model(self.timeframe)

        return self.model
