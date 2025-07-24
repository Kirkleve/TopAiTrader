import os
import torch
from model.CNN.cnn_price_predictor import CNNPricePredictor

class CNNModelManager:
    def __init__(self, symbol):
        self.symbol = symbol.replace('/', '_')
        self.model_dir = os.path.join("models", self.symbol, "cnn")
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model, timeframe):
        model_path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}_cnn.pth")
        torch.save(model.state_dict(), model_path)


    def load_model(self, timeframe):
        model_path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}_cnn.pth")
        if os.path.exists(model_path):
            model = CNNPricePredictor()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"✅ CNN-модель [{timeframe}] загружена: {model_path}")
            return model
        else:
            print(f"⚠️ CNN-модель для [{timeframe}] не найдена по пути {model_path}")
            return None

