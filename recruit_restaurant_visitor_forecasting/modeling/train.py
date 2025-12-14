from typing import Optional

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import typer

from recruit_restaurant_visitor_forecasting.config import (
    AIR_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
)
from recruit_restaurant_visitor_forecasting.dataset import DataDir, read_csv

app = typer.Typer()


def load_data() -> tuple:
    features = read_csv("features.csv", DataDir.PROCESSED)
    labels = read_csv("labels.csv", DataDir.PROCESSED)

    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]

    if len(features) != len(labels):
        raise ValueError(
            f"Features and labels have different lengths: {len(features)} vs {len(labels)}"
        )

    return features, labels


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features[VISIT_DATE_COL] = pd.to_datetime(features[VISIT_DATE_COL])
    features = features.sort_values([VISIT_DATE_COL, AIR_RESTAURANT_ID_COL])

    return features


def create_model_pipeline(
    alpha: float = 1.0,
    use_pca: bool = False,
    n_components: Optional[int] = None,
) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components)))

    model = Ridge(alpha=alpha, random_state=42)
    steps.append(("model", model))

    return Pipeline(steps)


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
