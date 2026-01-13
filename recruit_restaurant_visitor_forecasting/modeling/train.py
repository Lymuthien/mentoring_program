import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union

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


def train_test_split_by_date(
    X: pd.DataFrame,
    y: pd.Series,
    date_col: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    unique_dates = X[date_col].drop_duplicates().sort_values().reset_index(drop=True)
    n_dates = len(unique_dates)

    n_test_dates = int(np.ceil(n_dates * test_size))
    n_train_dates = n_dates - n_test_dates

    train_dates = set(unique_dates.iloc[:n_train_dates])
    test_dates = set(unique_dates.iloc[n_train_dates:])

    train_mask = X[date_col].isin(train_dates)
    test_mask = X[date_col].isin(test_dates)

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


class ExpandingWindowSplit:
    def __init__(self, test_size, date_col, n_splits=5, max_train_size=None):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.date_col = date_col

    def split(self, X, y=None, groups=None):
        dates = (
            pd.Series(X[self.date_col].unique()).sort_values().reset_index(drop=True)
        )
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

            train_indices = np.concatenate(
                [date_to_indices[date] for date in train_dates]
            )
            test_indices = np.concatenate(
                [date_to_indices[date] for date in test_dates]
            )

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class FeatureDropper(TransformerMixin, BaseEstimator):
    def __init__(self, features: list[str] = None):
        self.features = features or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.features:
            return X

        X = X.drop(columns=[f for f in self.features if f in X.columns])
        return X


def shap_fs(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    drop_features: list[str] = None,
    test_size: float = 0.2,
    top_k: int = 35,
) -> tuple[list[str], pd.DataFrame]:
    drop_features = set(drop_features or [])
    X_train, X_test, y_train, y_test = train_test_split_by_date(
        X, y, date_col=VISIT_DATE_COL, test_size=test_size
    )
    X_train = X_train.drop(columns=drop_features, errors="ignore")
    X_test = X_test.drop(columns=drop_features, errors="ignore")

    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    shap_abs_std = np.abs(shap_values).std(axis=0)

    imp_df = (
        pd.DataFrame(
            {
                "feature": X_train.columns,
                "shap_mean_abs": shap_abs_mean,
                "shap_std_abs": shap_abs_std,
            }
        )
        .sort_values(["shap_mean_abs"], ascending=False)
        .reset_index(drop=True)
    )

    selected_features = imp_df.head(top_k)["feature"].tolist()

    return selected_features, imp_df


def build_feature_drop_list(
    all_columns: list[str],
    selected_features: list[str],
) -> list[str]:
    drop_set = set(all_columns) - set(selected_features)
    return list(drop_set)


def rmsle(y_true, y_pred) -> float:
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


RMSLE_SCORER = make_scorer(rmsle, greater_is_better=False)


def _build_lgbm_pipeline(drop_features: list[str], random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("feature_dropper", FeatureDropper(drop_features)),
            ("model", LGBMRegressor(random_state=random_state, verbose=-1)),
        ]
    )


def lgbm_gridsearch(
    param_grid: dict[str, list],
    n_splits: int = 5,
    scoring: Union[str, callable] = RMSLE_SCORER,
    n_jobs: int = -1,
    verbose: int = 1,
    drop_features: list[str] = None,
    random_state: int = 42,
) -> GridSearchCV:
    pipeline = _build_lgbm_pipeline(drop_features, random_state)

    tscv = ExpandingWindowSplit(n_splits=n_splits, test_size=1, date_col=VISIT_DATE_COL)
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


def lgbm_optuna_search(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    n_splits: int = 5,
    scoring: Union[str, callable] = RMSLE_SCORER,
    n_jobs: int = -1,
    timeout: Optional[int] = None,
    drop_features: list[str] = None,
    random_state: int = 42,
) -> tuple[optuna.Study, Pipeline]:
    drop_features = drop_features or []

    tscv = ExpandingWindowSplit(
        n_splits=n_splits,
        test_size=1,
        date_col=VISIT_DATE_COL,
    )

    def _score_mean(cv_scores: np.ndarray) -> float:
        if (hasattr(scoring, "_sign") and getattr(scoring, "_sign") == -1) or (
            isinstance(scoring, str) and scoring.startswith("neg_")
        ):
            return float(-cv_scores.mean())
        return float(cv_scores.mean())

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 32),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 250),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0),
        }

        pipeline = _build_lgbm_pipeline(drop_features, random_state)
        pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=tscv,
            scoring=scoring,
            n_jobs=1,
        )
        trial.set_user_attr("cv_scores", cv_scores.tolist())
        return _score_mean(cv_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    best_pipeline = _build_lgbm_pipeline(drop_features, random_state)
    best_pipeline.set_params(**{f"model__{k}": v for k, v in study.best_params.items()})
    best_pipeline.fit(X, y)

    return study, best_pipeline


def ridge_gridsearch(
    param_grid: Optional[dict[str, list]] = None,
    n_splits: int = 5,
    scoring: str = "neg_root_mean_squared_log_error",
    n_jobs: int = -1,
    verbose: int = 1,
    drop_features: list[str] = None,
    random_state: int = 42,
) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("feature_dropper", FeatureDropper(drop_features)),
            ("scaler", StandardScaler()),
            ("selector", SelectFromModel(Lasso(random_state=random_state))),
            ("ridge", Ridge()),
        ]
    )

    if param_grid is None:
        param_grid = [
            {
                "selector__estimator__max_iter": [500, 1000, 2000],
                "selector__estimator__alpha": np.logspace(-3, 1, 5),
                "selector__threshold": ["median", "mean", 1e-1, 1e-2, 1e-3, 1e-4],
                "ridge__alpha": np.logspace(-2, 5, 8),
            }
        ]

    tscv = ExpandingWindowSplit(
        n_splits=n_splits, max_train_size=90, test_size=1, date_col=VISIT_DATE_COL
    )
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
