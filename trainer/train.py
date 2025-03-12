from trainer.universal_trainer import UniversalTrainer

def train(symbol='BTC/USDT'):
    trainer = UniversalTrainer(symbol)
    trainer.run_training()

if __name__ == "__main__":
    train()
