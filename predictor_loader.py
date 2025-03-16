from trainer.model_manager import ModelManager
from trainer.universal_trainer import UniversalTrainer


def load_models_for_symbols(symbols):
    models = {}

    for symbol in symbols:
        lstm_models = load_lstm_models()
        ppo_agent = manager.load_trained_model()
        scalers = manager.load_scalers()
        xgb_predictor = manager.load_xgb_model()

        if not lstm_models or not ppo_agent or not scalers or not xgb_predictor:
            print(f"⚠️ Модели для {symbol} не найдены или неполные, запускаем обучение...")
            trainer = UniversalTrainer(symbol)
            trainer.run_training()

            lstm_models = manager.load_lstm_models()
            ppo_agent = manager.load_trained_model()
            scalers = manager.load_scalers()
            xgb_predictor = manager.load_xgb_model()

        models[symbol] = {
            'lstm': lstm_models,
            'ppo': ppo_agent,
            'scalers': scalers,
            'xgb': xgb_predictor
        }

    return models
