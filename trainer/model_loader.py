import torch
import os
from model.lstm_price_predictor import LSTMPricePredictor
from trading.agent import DQNAgent

def load_models(symbol='BTC/USDT', input_size=6, state_size=9, action_size=3, timeframe='1h'):
    symbol_dir = symbol.replace('/', '_')

    lstm_model_path = f'trainer/models/{symbol_dir}/{symbol_dir}_{timeframe}_lstm.pth'
    dqn_model_path = f'trading/trained_agent_{symbol_dir}.pth'

    if not os.path.exists(lstm_model_path) or not os.path.exists(dqn_model_path):
        raise FileNotFoundError(f"Модели для {symbol} не найдены.")

    lstm_model = LSTMPricePredictor(input_size=input_size)
    lstm_model.load_state_dict(torch.load(lstm_model_path))
    lstm_model.eval()

    dqn_agent = DQNAgent(state_size=state_size, action_size=action_size)
    dqn_agent.model.load_state_dict(torch.load(dqn_model_path))
    dqn_agent.model.eval()

    return lstm_model, dqn_agent
