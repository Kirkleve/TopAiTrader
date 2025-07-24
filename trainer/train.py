import warnings
import logging
from trainer.universal_trainer import UniversalTrainer
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings('ignore')  # игнорируем все warnings
logging.getLogger("NP").setLevel(logging.ERROR)  # только ошибки NeuralProphet


def train(symbol):
    print(f"🚀 Запуск полного цикла обучения для {symbol}...")
    universal_trainer = UniversalTrainer(symbol, device='cpu')
    universal_trainer.run_training()
    print(f"\n✅ Обучение всех моделей для {symbol} успешно завершено!")


if __name__ == "__main__":
    train('BTC/USDT')
