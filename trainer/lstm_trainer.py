import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.lstm_price_predictor import LSTMPricePredictor
from trainer.model_manager.lstm_manager import LSTMModelManager
from data.data_preparation import DataPreparation

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

    def train(self, prepared_df, batch_size=64):
        # Сначала пытаемся загрузить модель и скейлер
        lstm_model = self.model_manager.load_model(self.timeframe)
        scaler = self.model_manager.load_scaler(self.timeframe)

        if lstm_model and scaler:
            print(f"📦 LSTM модель [{self.timeframe}] уже загружена, обучение пропущено.")
            return lstm_model, scaler

        # Если модель или скейлер не загружены — обучаем заново
        prep = DataPreparation(self.symbol, self.features)
        X_tensor, y_tensor, scaler = prep.get_lstm_data(prepared_df, self.seq_length)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.loss_fn(output, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

        # Сохраняем модель и скейлер только здесь!
        self.model_manager.save_model(self.model, self.timeframe)
        self.model_manager.save_scaler(scaler, self.timeframe)

        print(f"✅ LSTM [{self.timeframe}] обучен и сохранён с loss={best_loss:.4f}")

        return self.model, scaler

def train_lstm_models_for_timeframes(symbol: str, features: list[str], combined_data: dict, device='cpu'):
    lstm_models, lstm_scalers = {}, {}

    for tf, df in combined_data.items():
        print(f"🚀 Запуск LSTM тренера для таймфрейма [{tf}]...")
        trainer = LSTMTrainer(symbol, features, timeframe=tf, epochs=50, device=device)
        lstm_model, scaler = trainer.train(df)

        lstm_models[tf] = lstm_model
        lstm_scalers[tf] = scaler

    return lstm_models, lstm_scalers

