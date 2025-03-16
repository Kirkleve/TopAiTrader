import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna

class XGBoostPredictor:
    def __init__(self, symbol):
        self.symbol = symbol.replace("/", "_")
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.best_params = None

    def optimize_hyperparameters(self, X, y, trials=50):
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = xgb.XGBRegressor(**params, objective='reg:squarederror', eval_metric='rmse')
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
            )
            preds = model.predict(X_valid)
            return mean_squared_error(y_valid, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        self.best_params = study.best_params
        print(f"🚀 Лучшие параметры: {self.best_params}")
        return self.best_params

    def train(self, X_scaled, y_scaled, optimize=False):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_scaled, y_scaled, test_size=0.2, shuffle=False
        )

        if optimize:
            self.optimize_hyperparameters(X_train, y_train)

        self.model = xgb.XGBRegressor(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            objective='reg:squarederror',
            eval_metric='rmse'
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

        preds = self.model.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds) ** 0.5
        print(f"✅ XGBoost-модель обучена. RMSE на валидации: {rmse:.4f}")

    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        preds_scaled = self.model.predict(X_scaled)
        preds_real = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return preds_real
