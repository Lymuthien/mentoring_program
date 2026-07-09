from lightgbm import LGBMRegressor
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FeatureDropper(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        drop_features: list[str] = None,
        features_top: list[str] = None,
        keep_count: int = None,
    ):
        self.drop_features = drop_features
        self.features_top = features_top
        self.keep_count = keep_count

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.drop_features:
            X = X.drop(columns=[f for f in self.drop_features if f in X.columns])

        if self.features_top:
            if not self.keep_count:
                return X

            drop_features = self.features_top[self.keep_count :]
            X = X.drop(columns=[f for f in drop_features if f in X.columns])

        return X


class MeanBaseline(BaseEstimator):
    def __init__(self, id_col: str, date_col: str, dow_col: str, mean_col: str):
        self.id_col = id_col
        self.date_col = date_col
        self.dow_col = dow_col
        self.mean_col = mean_col
        self._last_values = None
        self._global_mean = None

    def fit(self, X, y):
        X = X.sort_values([self.id_col, self.dow_col, self.date_col])
        last_values = X.groupby([self.id_col, self.dow_col], as_index=False).tail(1)
        last_values = last_values[[self.id_col, self.dow_col, self.mean_col]]
        self._last_values = last_values

        self._global_mean = X.groupby([self.dow_col], as_index=False)[self.mean_col].mean()

        return self

    def predict(self, X):
        if self.mean_col in X:
            X = X.drop(columns=[self.mean_col])

        X = X.merge(self._last_values, on=[self.id_col, self.dow_col], how="left")
        X = X.merge(
            self._global_mean,
            on=self.dow_col,
            how="left",
            suffixes=("", "_global")
        )

        X[self.mean_col] = X[self.mean_col].fillna(
            X[f"{self.mean_col}_global"]
        )

        return X[self.mean_col].values

def build_lgbm_pipeline(
    drop_features: list[str],
    random_state: int,
    top_features: list[str] = None,
    f_count: int = None,
    objective: str = "regression",
) -> Pipeline:
    return Pipeline(
        [
            ("feature_dropper", FeatureDropper(drop_features, top_features, f_count)),
            (
                "model",
                LGBMRegressor(
                    random_state=random_state, verbose=-1, objective=objective
                ),
            ),
        ]
    )


def build_ridge_pipeline(
    drop_features: list[str],
    random_state: int,
):
    return Pipeline(
        [
            ("feature_dropper", FeatureDropper(drop_features)),
            ("scaler", StandardScaler()),
            ("selector", SelectFromModel(Lasso(random_state=random_state))),
            ("ridge", Ridge()),
        ]
    )
