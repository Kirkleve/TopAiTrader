from neuralprophet import NeuralProphet, set_log_level


def create_neuralprophet_model():
    set_log_level("ERROR")  # отключаем подробные логи

    model = NeuralProphet(
        yearly_seasonality=False,  # У крипты нет годовой сезонности
        weekly_seasonality=True,  # Недельная полезна
        daily_seasonality=True,  # Дневная полезна
        learning_rate=0.003,  # Более низкий шаг обучения
        epochs=150,  # Увеличим количество эпох
        n_forecasts=1,
        n_lags=48,  # Увеличим окна, чтобы модель видела больше данных прошлого
        batch_size=64  # Увеличим батч, чтобы стабилизировать градиенты
    )

    # Отключаем внутреннее логгирование Lightning
    if hasattr(model, "trainer"):
        model.trainer.logger = False
        model.trainer.enable_checkpointing = False
        model.trainer.enable_model_summary = False

    return model
