import json
import os
import pandas as pd

PARAMS_FILE = 'params.json'
METRICS_FILE = 'bot_metrics.csv'

def load_strategy_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as file:
            return json.load(file)
    return {"threshold_percent": 0.05, "sentiment_threshold": 0.6, "sl_multiplier": 1.5, "tp_multiplier": 3}

def save_strategy_params(params):
    with open(PARAMS_FILE, 'w') as file:
        json.dump(params, file, indent=4)

def adapt_strategy():
    if not os.path.exists(METRICS_FILE):
        print("⚠️ Нет файла с метриками сделок, используются стандартные параметры.")
        return load_strategy_params()

    df = pd.read_csv(METRICS_FILE)
    total_trades = len(df)

    if total_trades == 0:
        print("⚠️ Нет сделок для анализа, параметры не изменены.")
        return load_strategy_params()

    winning_trades = df[df['pnl_percent'] > 0]
    losing_trades = df[df['pnl_percent'] <= 0]

    win_rate = len(winning_trades) / total_trades
    avg_profit = winning_trades['pnl_percent'].mean() if not winning_trades.empty else 0
    avg_loss = abs(losing_trades['pnl_percent'].mean()) if not losing_trades.empty else 0

    params = load_strategy_params()

    # Адаптация threshold_percent
    if win_rate < 0.5 and params['threshold_percent'] <= 0.2:
        params['threshold_percent'] += 0.01
    elif win_rate > 0.7 and params['threshold_percent'] > 0.02:
        params['threshold_percent'] -= 0.01

    # Адаптация sentiment_threshold
    if avg_profit < avg_loss:
        params['sentiment_threshold'] += 0.05
    else:
        params['sentiment_threshold'] = max(0.1, params['sentiment_threshold'] - 0.05)

    # Дополнительная адаптация TP и SL
    if avg_profit < avg_loss:
        params['sl_multiplier'] = min(params['sl_multiplier'] + 0.1, 3)
        params['tp_multiplier'] = max(params['tp_multiplier'] - 0.1, 1.5)
    else:
        params['tp_multiplier'] = min(params['tp_multiplier'] + 0.1, 5)
        params['sl_multiplier'] = max(params['sl_multiplier'] - 0.1, 1)

    save_strategy_params(params)

    print(f"🔄 Параметры адаптированы: {params}")
    return params
