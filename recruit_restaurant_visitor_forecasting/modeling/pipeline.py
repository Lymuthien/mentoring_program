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


def build_lgbm_pipeline(
    drop_features: list[str],
    random_state: int,
    top_features: list[str] = None,
    f_count: int = None,
) -> Pipeline:
    return Pipeline(
        [
            ("feature_dropper", FeatureDropper(drop_features, top_features, f_count)),
            ("model", LGBMRegressor(random_state=random_state, verbose=-1)),
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
