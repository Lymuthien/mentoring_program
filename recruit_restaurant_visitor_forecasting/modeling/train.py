import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
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


class ExpandingWindowSplit:
    def __init__(self, test_size, date_col, n_splits=5, max_train_size=60):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.date_col = date_col

    def split(self, X, y=None, groups=None):
        dates = pd.Series(X[self.date_col].unique()).sort_values().reset_index(drop=True)
        n_dates = len(dates)
        n_splits = self.n_splits
        test_size = self.test_size

        if n_dates - n_splits * test_size <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of dates"
                f"={n_dates} with test_size={test_size}."
            )

        indices = pd.Series(np.arange(len(X)), index=X.index)
        date_to_indices = {}
        for date, group_indices in X.groupby(self.date_col).groups.items():
            date_to_indices[date] = indices.loc[group_indices].values
        test_starts = range(n_dates - n_splits * test_size, n_dates, test_size)

        for test_start in test_starts:
            test_end = test_start + test_size
            test_dates = dates.iloc[test_start:test_end]

            train_end = test_start
            if self.max_train_size and self.max_train_size < train_end:
                train_start = train_end - self.max_train_size
            else:
                train_start = 0
            train_dates = dates.iloc[train_start:train_end]
            
            train_indices = np.concatenate([date_to_indices[date] for date in train_dates])
            test_indices = np.concatenate([date_to_indices[date] for date in test_dates])
            
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class FeatureDropper(TransformerMixin, BaseEstimator):
    def __init__(self, features: list[str] = None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=[f for f in self.features if f in X.columns])
        return X


def create_model_gridsearch(
    param_grid: Optional[dict[str, list]] = None,
    n_splits: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 1,
    drop_features: list[str] = None,
) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("feature_dropper", FeatureDropper(drop_features)),
            ("scaler", StandardScaler()),
            ("selector", SelectFromModel(Lasso(max_iter=5000, random_state=42))),
            ("ridge", Ridge()),
        ]
    )

    if param_grid is None:
        param_grid = [
            {
                "selector__estimator__alpha": np.logspace(-3, 1, 5),
                "selector__threshold": ["median", "mean", 1e-1, 1e-2, 1e-3, 1e-4],
                "ridge__alpha": np.logspace(-2, 5, 8),
            }
        ]

    tscv = ExpandingWindowSplit(n_splits=n_splits, max_train_size=175, test_size=39, date_col=VISIT_DATE_COL)
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
