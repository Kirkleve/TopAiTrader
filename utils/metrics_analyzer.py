import pandas as pd
from datetime import datetime, timedelta
import os


def calculate_metrics(period='day'):
    metrics_file = 'bot_metrics.csv'

    if not os.path.exists(metrics_file):
        print("⚠️ Файл bot_metrics.csv не найден, метрики пока недоступны.")
        return {
            'total_pnl': 0.0,
            'win_rate': 0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'risk_reward': 0.0,
            'trade_frequency': 0.0
        }

    df = pd.read_csv(metrics_file, parse_dates=['timestamp'])
    now = datetime.now()

    if period == 'day':
        start_date = now - timedelta(days=1)
    elif period == 'week':
        start_date = now - timedelta(weeks=1)
    elif period == 'month':
        start_date = now - timedelta(days=30)
    elif period == 'year':
        start_date = now - timedelta(days=365)
    else:
        start_date = df['timestamp'].min()

    df_period = df[df['timestamp'] >= start_date]

    trade_count = len(df_period)
    df_profit = df_period[df_period['pnl_percent'] > 0]
    df_loss = df_period[df_period['pnl_percent'] < 0]

    metrics = {
        'total_pnl': round(df_period['pnl_percent'].sum(), 2),
        'win_rate': round(len(df_profit) / trade_count * 100, 2) if trade_count else 0,
        'avg_profit': round(df_profit['pnl_percent'].mean(), 2) if not df_profit.empty else 0.0,
        'avg_loss': round(df_loss['pnl_percent'].mean(), 2) if not df_loss.empty else 0.0,
        'risk_reward': round(abs(df_profit['pnl_percent'].mean() / df_loss['pnl_percent'].mean()), 2) if not df_loss.empty else 0.0,
        'trade_frequency': round(trade_count / ((now - start_date).days or 1), 2)
    }

    return metrics
