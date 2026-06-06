import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif as _mutual_info_classifier

from config.defaults import RANDOM_SEED

_TOP_VALUES_LIMIT = 5
_CATEGORICAL_MISSING_TOKEN = "__MISSING__"


def _is_numeric_feature(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _numeric_distribution(series: pd.Series) -> dict[str, float]:
    s = series.dropna().astype(float)
    if len(s) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "median": float(s.median()),
    }


def _categorical_distribution(series: pd.Series) -> dict[str, dict[str, int]]:
    top = series.value_counts(dropna=True).head(_TOP_VALUES_LIMIT)
    return {"top_values": {str(k): int(v) for k, v in top.items()}}


def _prepare_features_for_mutual_info_classifier(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Numeric values with NaN imputed by median, categoricals label-coded with a sentinel for NaN. discrete_mask
    flags the categoricals so sklearn treats them as discrete."""
    n = len(df)
    X = np.zeros((n, len(feature_cols)), dtype=float)
    discrete_mask = np.zeros(len(feature_cols), dtype=bool)
    for j, col in enumerate(feature_cols):
        s = df[col]
        if _is_numeric_feature(s):
            filled = s.astype(float)
            if filled.isna().any():
                median = filled.median()
                if pd.isna(median):
                    median = 0.0
                filled = filled.fillna(median)
            X[:, j] = filled.to_numpy()
            discrete_mask[j] = False
        else:
            filled = s.astype("object").where(~s.isna(), _CATEGORICAL_MISSING_TOKEN)
            codes = pd.Categorical(filled).codes
            X[:, j] = codes.astype(float)
            discrete_mask[j] = True
    return X, discrete_mask


def profile_dataframe(
    df: pd.DataFrame,
    label_column: str = "isFraud",
    random_state: int = RANDOM_SEED,
) -> dict[str, dict[str, Any]]:
    """Per-column profile: dtype, detected_type, cardinality, missing_rate,
    distribution, and mutual information against label_column.

    MI uses the fraud label for characterization only. To rank which features
    carry signal; LOF and the geometry MLPs never train on the label. MI is
    None for every column when label_column is absent. random_state seeds
    _mutual_info_classifier.
    """
    feature_cols = [c for c in df.columns if c != label_column]

    if label_column in df.columns and feature_cols:
        y = df[label_column].to_numpy()
        label_mask = ~pd.isna(y)
        X, discrete_mask = _prepare_features_for_mutual_info_classifier(
            df.loc[label_mask, :], feature_cols
        )
        mi_scores = _mutual_info_classifier(
            X,
            y[label_mask],
            discrete_features=discrete_mask,
            random_state=random_state,
        )
        mi_by_column = {col: float(mi) for col, mi in zip(feature_cols, mi_scores)}
    else:
        mi_by_column = {}

    profile: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        series = df[col]
        if _is_numeric_feature(series):
            detected_type = "numeric"
            distribution: dict[str, Any] = _numeric_distribution(series)
        else:
            detected_type = "categorical"
            distribution = _categorical_distribution(series)

        if col == label_column or col not in mi_by_column:
            mutual_information: float | None = None
        else:
            mutual_information = mi_by_column[col]

        profile[col] = {
            "dtype": str(series.dtype),
            "detected_type": detected_type,
            "cardinality": int(series.nunique(dropna=True)),
            "missing_rate": float(series.isna().mean()),
            "distribution": distribution,
            "mutual_information": mutual_information,
        }

    return profile


def save_profile_as_json(profile: dict[str, dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, sort_keys=True)


def load_profile_from_json(path: Path | str) -> dict[str, dict[str, Any]]:
    """Inverse of save_profile_as_json."""
    with open(path) as f:
        return json.load(f)
