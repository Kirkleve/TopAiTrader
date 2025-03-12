from ta.volatility import AverageTrueRange


def calculate_atr(df, period=14):
    atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=period)
    return atr_indicator.average_true_range().iloc[-1]

