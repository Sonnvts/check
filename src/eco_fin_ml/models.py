from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from arch import arch_model
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


@dataclass
class TorchSequenceConfig:
    seq_length: int = 12
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, task: str):
        super().__init__()
        self.task = task
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class GRUModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, task: str):
        super().__init__()
        self.task = task
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class TransformerModel(torch.nn.Module):
    def __init__(self, input_size: int, d_model: int, output_size: int, task: str):
        super().__init__()
        self.task = task
        self.embedding = torch.nn.Linear(input_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = torch.nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.fc(x)


class TorchSequenceTrainer:
    def __init__(self, model: torch.nn.Module, task: str, config: TorchSequenceConfig):
        self.model = model
        self.task = task
        self.config = config
        self.loss_fn = (
            torch.nn.BCEWithLogitsLoss() if task == "classification" else torch.nn.MSELoss()
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.config.epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_x).squeeze(-1)
                loss = self.loss_fn(preds, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X, dtype=torch.float32)).squeeze(-1)
        if self.task == "classification":
            return torch.sigmoid(preds).numpy()
        return preds.numpy()


def make_sequence_data(X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    for idx in range(seq_length, len(X)):
        sequences.append(X[idx - seq_length : idx])
        targets.append(y[idx])
    return np.array(sequences), np.array(targets)


def build_models(task: str) -> Dict[str, object]:
    if task == "classification":
        return {
            "logistic": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "xgboost": XGBClassifier(eval_metric="logloss"),
            "lightgbm": LGBMClassifier(),
            "catboost": CatBoostClassifier(verbose=False),
            "svm": SVC(probability=True),
            "mlp": MLPClassifier(max_iter=500),
        }
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(),
        "elasticnet": ElasticNet(),
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "xgboost": XGBRegressor(),
        "lightgbm": LGBMRegressor(),
        "catboost": CatBoostRegressor(verbose=False),
        "svm": SVR(),
        "mlp": MLPRegressor(max_iter=500),
    }


def build_stacking(task: str, estimators: List[Tuple[str, object]]) -> object:
    if task == "classification":
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
    return StackingRegressor(estimators=estimators, final_estimator=LinearRegression())


def fit_arima(y: pd.Series, order: Tuple[int, int, int]) -> object:
    model = ARIMA(y, order=order)
    return model.fit()


def fit_sarima(y: pd.Series, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int]) -> object:
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False)


def fit_garch(y: pd.Series) -> object:
    model = arch_model(y, vol="Garch", p=1, q=1)
    return model.fit(disp="off")


class HybridArimaResidual:
    def __init__(self, arima_order: Tuple[int, int, int], residual_model: object):
        self.arima_order = arima_order
        self.residual_model = residual_model
        self.arima_result = None

    def fit(self, y: pd.Series, X: np.ndarray) -> None:
        self.arima_result = fit_arima(y, self.arima_order)
        residuals = y - self.arima_result.fittedvalues
        self.residual_model.fit(X, residuals)

    def predict(self, X: np.ndarray, steps: int) -> np.ndarray:
        if self.arima_result is None:
            raise ValueError("ARIMA model not fitted")
        arima_forecast = self.arima_result.forecast(steps=steps)
        residual_preds = self.residual_model.predict(X)
        return arima_forecast.values + residual_preds


def evaluate_arima(y_true: pd.Series, forecast: np.ndarray) -> float:
    return mean_squared_error(y_true, forecast, squared=False)
