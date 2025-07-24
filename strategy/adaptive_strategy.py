import json
import os
import pandas as pd

PARAMS_FILE = 'risk_params.json'
METRICS_FILE = 'bot_metrics.csv'

def load_risk_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as file:
            return json.load(file)
    return {
        "risk_percent": 0.02,
        "sentiment_threshold": 0.5,
        "sl_multiplier": 1.5,
        "tp_multiplier": 3
    }

def save_risk_params(params):
    with open(PARAMS_FILE, 'w') as file:
        json.dump(params, file, indent=4)

def adapt_risk_params():
    params = load_risk_params()

    if not os.path.exists(METRICS_FILE):
        print("⚠️ Нет файла с метриками сделок, используются текущие параметры риска.")
        return params

    df = pd.read_csv(METRICS_FILE)
    total_trades = len(df)

    if total_trades < 10:
        print("⚠️ Недостаточно данных для адаптации параметров риска (нужно минимум 10 сделок).")
        return params

    winning_trades = df[df['pnl_percent'] > 0]
    losing_trades = df[df['pnl_percent'] <= 0]

    win_rate = len(winning_trades) / total_trades
    avg_profit = winning_trades['pnl_percent'].mean() if not winning_trades.empty else 0
    avg_loss = abs(losing_trades['pnl_percent'].mean()) if not losing_trades.empty else 0

    # Адаптация риск-процента на сделку
    if win_rate > 0.6:
        params['risk_percent'] = min(params['risk_percent'] + 0.005, 0.05)
    elif win_rate < 0.4:
        params['risk_percent'] = max(params['risk_percent'] - 0.005, 0.01)

    # Адаптация sentiment_threshold
    if avg_profit < avg_loss:
        params['sentiment_threshold'] = min(params['sentiment_threshold'] + 0.05, 0.8)
    else:
        params['sentiment_threshold'] = max(params['sentiment_threshold'] - 0.05, 0.2)

    # Адаптация TP и SL (только тонкая настройка)
    if avg_profit < avg_loss:
        params['sl_multiplier'] = min(params['sl_multiplier'] + 0.1, 2.0)
        params['tp_multiplier'] = max(params['tp_multiplier'] - 0.1, 2.0)
    else:
        params['tp_multiplier'] = min(params['tp_multiplier'] + 0.1, 4.0)
        params['sl_multiplier'] = max(params['sl_multiplier'] - 0.1, 1.0)

    save_risk_params(params)
    print(f"🔄 Параметры риска успешно адаптированы: {params}")

    return params
