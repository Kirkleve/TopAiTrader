import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from model.xgboost_predictor import XGBoostPredictor
from trainer.model_manager.xgb_manager import XGBModelManager
from data.data_preparation import DataPreparation


def train_xgboost_model(symbol: str, features: list[str], df_1h, optimize=True, trials=50):
    """
    Обучает XGBoost на 1h-данных, сохраняет модель и скейлеры.
    Возвращает XGBoostPredictor с обученной моделью.
    """
    manager = XGBModelManager(symbol)
    predictor = XGBoostPredictor(symbol)

    model, scaler_X, scaler_y = manager.load_model_and_scalers()
    if model and scaler_X and scaler_y:
        predictor.model = model
        predictor.scaler_X = scaler_X
        predictor.scaler_y = scaler_y
        print("✅ XGBoost-модель загружена из файла.")
        return predictor

    print("🚀 Обучение XGBoost [1h]...")

    # Используем общий модуль подготовки данных
    prep = DataPreparation(symbol, features)
    X_scaled, y_scaled, scaler_X, scaler_y = prep.get_xgboost_data(df_1h)

    X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

    if optimize:
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }

            model = xgb.XGBRegressor(**params, objective='reg:squarederror', eval_metric='rmse')
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            preds = model.predict(X_valid)
            return mean_squared_error(y_valid, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=trials)
        best_params = study.best_params
    else:
        best_params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }

    # Обучение модели с найденными параметрами
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds) ** 0.5
    print(f"✅ XGBoost обучен. RMSE на валидации: {rmse:.4f}")

    predictor.model = model
    predictor.scaler_X = scaler_X
    predictor.scaler_y = scaler_y

    manager.save_model_and_scalers(model, scaler_X, scaler_y)

    return predictor
