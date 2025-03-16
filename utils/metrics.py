import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(true_values, predicted_values):
    """
    Рассчитывает метрики точности прогнозирования модели.

    :param true_values: numpy-массив фактических значений цен
    :param predicted_values: numpy-массив предсказанных моделью цен
    :return: словарь с метриками RMSE, MAE, MAPE, R²
    """
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)  # Среднеквадратичная ошибка
    mae = mean_absolute_error(true_values, predicted_values)  # Средняя абсолютная ошибка
    r2 = r2_score(true_values, predicted_values)  # Коэффициент детерминации

    # Вычисляем MAPE (избегаем деления на 0)
    mape = np.mean(np.abs((true_values - predicted_values) / np.clip(true_values, 1e-6, None))) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R²": r2
    }
