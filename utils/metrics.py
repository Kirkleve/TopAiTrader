import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_rmse(true_values, predicted_values):
    """
    Рассчитывает среднеквадратичную ошибку прогнозирования (RMSE).

    :param true_values: numpy-массив фактических значений цен
    :param predicted_values: numpy-массив предсказанных моделью цен
    :return: RMSE (ошибка модели)
    """
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    return rmse
