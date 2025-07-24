import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def calculate_fibonacci_levels(min_price, max_price):
    diff = max_price - min_price
    return {
        "fib_23_6": max_price - 0.236 * diff,
        "fib_38_2": max_price - 0.382 * diff,
        "fib_50": max_price - 0.5 * diff,
        "fib_61_8": max_price - 0.618 * diff,
        "fib_78_6": max_price - 0.786 * diff,
    }


def prepare_neuralprophet_data(df, features, fib_window=100):
    np_input_df = pd.DataFrame({
        'ds': df.index,
        'y': df['close'].values
    })

    price_scaler = MinMaxScaler()
    np_input_df['y'] = price_scaler.fit_transform(df[['close']])

    feature_scalers = {}
    for feature in features:
        if feature != 'close':
            scaler = MinMaxScaler()
            np_input_df[feature] = scaler.fit_transform(df[[feature]])
            feature_scalers[feature] = scaler

    # Фибоначчи (без изменений)
    fib_cols = ['fib_23_6', 'fib_38_2', 'fib_50', 'fib_61_8', 'fib_78_6']
    for col in fib_cols:
        np_input_df[col] = np.nan

    if len(np_input_df) >= fib_window:
        for i in range(fib_window, len(np_input_df)):
            min_price = df['close'].iloc[i - fib_window:i].min()
            max_price = df['close'].iloc[i - fib_window:i].max()
            fib_levels = calculate_fibonacci_levels(min_price, max_price)
            for col in fib_cols:
                np_input_df.at[i, col] = fib_levels[col]
    else:
        min_price = df['close'].min()
        max_price = df['close'].max()
        fib_levels = calculate_fibonacci_levels(min_price, max_price)
        for col in fib_cols:
            np_input_df[col] = fib_levels[col]

    np_input_df.fillna(method='bfill', inplace=True)
    np_input_df.fillna(method='ffill', inplace=True)
    np_input_df.reset_index(drop=True, inplace=True)

    return np_input_df, price_scaler, feature_scalers


