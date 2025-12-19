import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Optional

from recruit_restaurant_visitor_forecasting.config import (
    AIR_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
)
from recruit_restaurant_visitor_forecasting.dataset import DataDir, read_csv


def load_data() -> tuple:
    features = read_csv("features.csv", DataDir.PROCESSED)
    labels = read_csv("labels.csv", DataDir.PROCESSED)

    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]

    return features, labels


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features[VISIT_DATE_COL] = pd.to_datetime(features[VISIT_DATE_COL])
    features = features.sort_values([VISIT_DATE_COL, AIR_RESTAURANT_ID_COL])

    return features


def create_model_gridsearch(
    param_grid: Optional[dict[str, list]] = None,
    n_splits: int = 3,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 1,
) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectFromModel(Lasso(max_iter=5000, random_state=42))),
            ("ridge", Ridge()),
        ]
    )

    if param_grid is None:
        param_grid = [
            {
                "selector__estimator__alpha": [1e-3, 1e-2, 1e-1],
                "selector__threshold": ["median", "mean", 1e-4],
                "ridge__alpha": np.logspace(-3, 3, 7),
            }
        ]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=False,
    )

    return grid_search


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> dict:
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    X_array = X.values
    y_array = y.values

    rmse_scores = -cross_val_score(
        model,
        X_array,
        y_array,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    mae_scores = -cross_val_score(
        model, X_array, y_array, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    r2_scores = cross_val_score(
        model, X_array, y_array, cv=tscv, scoring="r2", n_jobs=-1
    )

    results = {
        "rmse_mean": rmse_scores.mean(),
        "rmse_std": rmse_scores.std(),
        "mae_mean": mae_scores.mean(),
        "mae_std": mae_scores.std(),
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
        "rmse_scores": rmse_scores.tolist(),
        "mae_scores": mae_scores.tolist(),
        "r2_scores": r2_scores.tolist(),
    }

    return results
