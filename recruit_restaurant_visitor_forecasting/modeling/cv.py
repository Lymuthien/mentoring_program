import numpy as np
import pandas as pd

from sklearn.base import clone
from joblib import Parallel, delayed
from recruit_restaurant_visitor_forecasting.config.preprocessing import DROP_COLUMNS
from recruit_restaurant_visitor_forecasting.modeling.predict import recursive_predict


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


def _run_one_fold(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    scoring,
):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    model_ = clone(model)
    model_.fit(X_train, y_train)

    y_pred, _ = recursive_predict(
        model=model_,
        train_features=X_train,
        train_labels=y_train,
        test_features=X_val,
        drop_cols=DROP_COLUMNS,
    )

    return scoring(y_val, y_pred)


def cv_recursive_score(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring,
    n_jobs: int = 1,
    verbose: int = 0,
):
    splits = list(cv.split(X))

    scores = Parallel(
        n_jobs=n_jobs,
        verbose=verbose,
        backend="loky",
    )(
        delayed(_run_one_fold)(
            model,
            X,
            y,
            train_idx,
            val_idx,
            scoring,
        )
        for train_idx, val_idx in splits
    )

    return np.asarray(scores)