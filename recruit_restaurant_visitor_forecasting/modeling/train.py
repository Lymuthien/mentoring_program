import numpy as np
import optuna
import pandas as pd
import shap
import mlflow
from optuna.samplers import TPESampler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from typing import Optional, Union

from recruit_restaurant_visitor_forecasting.config import (
    AIR_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
)
from recruit_restaurant_visitor_forecasting.dataset import DataDir, read_csv
from recruit_restaurant_visitor_forecasting.modeling.cv import (
    ExpandingWindowSplit,
    cv_recursive_score,
)
from recruit_restaurant_visitor_forecasting.modeling.pipeline import (
    build_lgbm_pipeline,
    build_ridge_pipeline,
)
from recruit_restaurant_visitor_forecasting.utils import rmsle


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


def shap_fs(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    drop_features: list[str] = None,
    test_size: float = 0.2,
    top_k: int = None,
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

    if top_k:
        selected_features = imp_df.head(top_k)["feature"].tolist()
    else:
        selected_features = imp_df["feature"].tolist()

    return selected_features, imp_df


def lgbm_gridsearch(
    param_grid: dict[str, list],
    scoring: Union[str, callable],
    n_splits: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
    drop_features: list[str] = None,
    random_state: int = 42,
) -> GridSearchCV:
    pipeline = build_lgbm_pipeline(drop_features, random_state)

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


def _score_mean(cv_scores: np.ndarray, scoring) -> float:
    if (hasattr(scoring, "_sign") and getattr(scoring, "_sign") == -1) or (
        isinstance(scoring, str) and scoring.startswith("neg_")
    ):
        return float(-cv_scores.mean())
    return float(cv_scores.mean())


def lgbm_optuna_search(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    n_splits: int = 5,
    scoring: Union[str, callable] = rmsle,
    n_jobs: int = -1,
    timeout: Optional[int] = None,
    drop_features: list[str] = None,
    features_top: list[str] = None,
    random_state: int = 42,
    window_test_size: int = 1,
) -> tuple[optuna.Study, Pipeline]:
    drop_features = drop_features or []

    tscv = ExpandingWindowSplit(
        n_splits=n_splits,
        test_size=window_test_size,
        date_col=VISIT_DATE_COL,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        with mlflow.start_run(
            nested=True, run_name=f"trial_{trial.number}"
        ) as child_run:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 5),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-2, 0.3, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 35),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            max_depth = params["max_depth"]
            params["num_leaves"] = trial.suggest_int(
                "num_leaves", 2**max_depth // 2, 2**max_depth
            )

            pipeline = build_lgbm_pipeline(drop_features, random_state, features_top)
            pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})

            if features_top:
                fd_params = {
                    "feature_dropper__keep_count": trial.suggest_int(
                        "keep_count", 10, 49
                    )
                }
                pipeline.set_params(**fd_params)

            cv_scores = cv_recursive_score(
                pipeline,
                X,
                y,
                cv=tscv,
                scoring=scoring,
                n_jobs=8,
            )
            mean_rmsle = _score_mean(cv_scores, scoring)
            mlflow.log_metrics({"mean_rmsle": mean_rmsle})
            mlflow.sklearn.log_model(pipeline, name="model")
            trial.set_user_attr("cv_scores", cv_scores.tolist())
            trial.set_user_attr("run_id", child_run.info.run_id)

            return _score_mean(cv_scores)

    with mlflow.start_run(run_name="study"):
        mlflow.log_param("n_trials", n_trials)

        sampler = TPESampler(seed=random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

        best_params = study.best_params.copy()
        mlflow.log_params(best_params)
        mlflow.log_metrics({"best_mean_rmsle": study.best_value})
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)

        best_pipeline = build_lgbm_pipeline(drop_features, random_state, features_top)
        if "keep_count" in best_params:
            best_pipeline.set_params(
                **{"feature_dropper__keep_count": best_params.pop("keep_count")}
            )

        best_pipeline.set_params(**{f"model__{k}": v for k, v in best_params.items()})
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
    pipeline = build_ridge_pipeline(drop_features, random_state)

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
