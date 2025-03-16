import os
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler
from model.CNN.cnn_price_predictor import CNNPricePredictor
from model.lstm_price_predictor import LSTMPricePredictor

class CNNLSTMModelManager:
    def __init__(self, symbol, features):
        self.symbol = symbol.replace('/', '_')
        self.features = features
        self.model_dir = os.path.join("models", self.symbol)
        self.cnn_dir = os.path.join(self.model_dir, "cnn")
        self.lstm_dir = os.path.join(self.model_dir, "lstm")
        self.scaler_dir = os.path.join(self.model_dir, "scalers")

        for path in [self.cnn_dir, self.lstm_dir, self.scaler_dir]:
            os.makedirs(path, exist_ok=True)

    def save_cnn_model(self, model, timeframe):
        model_path = os.path.join(self.cnn_dir, f"{self.symbol}_{timeframe}_cnn.pth")
        torch.save(model.state_dict(), model_path)

    def load_cnn_model(self, timeframe):
        model_path = os.path.join(self.cnn_dir, f"{self.symbol}_{timeframe}_cnn.pth")
        if os.path.exists(model_path):
            model = CNNPricePredictor()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"✅ CNN-модель [{timeframe}] загружена: {model_path}")
            return model
        print(f"⚠️ CNN-модель не найдена: {model_path}")
        return None

    def save_lstm_model(self, model, timeframe):
        model_path = os.path.join(self.lstm_dir, f"{self.symbol}_{timeframe}_lstm.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✅ LSTM-модель сохранена: {model_path}")

    def load_lstm_model(self, timeframe):
        model_path = os.path.join(self.lstm_dir, f"{self.symbol}_{timeframe}_lstm.pth")
        if os.path.exists(model_path):
            model = LSTMPricePredictor(input_size=len(self.features))
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"✅ LSTM-модель загружена: {model_path}")
            return model
        else:
            print(f"⚠️ LSTM-модель не найдена: {model_path}")
            return None

    def save_scaler(self, scaler, timeframe):
        scaler_path = os.path.join(self.scaler_dir, f"scaler_{self.symbol}_{timeframe}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler сохранён: models\\{self.symbol}\\scalers\\...")

    def load_scaler(self, timeframe):
        scaler_path = os.path.join(self.scaler_dir, f"scaler_{self.symbol}_{timeframe}.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Scaler загружен: models\\{self.symbol}\\scalers\\...")
        else:
            scaler = MinMaxScaler()
            print(f"⚠️ Scaler не найден, создан новый MinMaxScaler для {self.symbol} [{timeframe}]")
        return scaler

