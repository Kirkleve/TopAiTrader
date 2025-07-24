import ta.volatility as vol


def calculate_atr(df, window=14):
    """
    Рассчитывает ATR для одного таймфрейма.
    """
    return vol.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window).average_true_range()


def get_combined_atr(data_multi, window=14):
    """
    Рассчитывает усреднённый ATR по всем таймфреймам.
    """
    atr_values = []
    for tf, df in data_multi.items():
        if df is not None and not df.empty:
            atr_values.append(calculate_atr(df, window).mean())

    return sum(atr_values) / len(atr_values) if atr_values else None
