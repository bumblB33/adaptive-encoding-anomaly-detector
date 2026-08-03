from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif as _mutual_info_classifier

from config.defaults import RANDOM_SEED
from config.feature_profile import _TOP_VALUES_LIMIT, _CATEGORICAL_MISSING_TOKEN


def _is_numeric_feature(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _numeric_distribution(series: pd.Series) -> dict[str, float]:
    numeric_series_col = series.dropna().astype(float)
    if len(numeric_series_col) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "min": float(numeric_series_col.min()),
        "max": float(numeric_series_col.max()),
        "mean": float(numeric_series_col.mean()),
        "std": float(numeric_series_col.std(ddof=0)),
        "median": float(numeric_series_col.median()),
    }


def _categorical_distribution(series: pd.Series) -> dict[str, dict[str, int]]:
    top = series.value_counts(dropna=True).head(_TOP_VALUES_LIMIT)
    return {"top_values": {str(k): int(v) for k, v in top.items()}}


def _prepare_features_for_mutual_info_classifier(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    - Numeric values with NaN inferred by median.
    - Categoricals label-coded with a sentinel for NaN.
    - discrete_mask flags categoricals so sklearn treats them as discrete.
    """
    numerical_feature_length = len(df)
    feature_matrix_X = np.zeros(
        (numerical_feature_length, len(feature_cols)), dtype=float
    )
    discrete_mask = np.zeros(len(feature_cols), dtype=bool)
    for row_index, col in enumerate(feature_cols):
        numeric_series_col = df[col]
        if _is_numeric_feature(numeric_series_col):
            filled = numeric_series_col.astype(float)
            if filled.isna().any():
                median = filled.median()
                if pd.isna(median):
                    median = 0.0
                filled = filled.fillna(median)
            feature_matrix_X[:, row_index] = filled.to_numpy()
            discrete_mask[row_index] = False
        else:
            filled = numeric_series_col.astype("object").where(
                ~numeric_series_col.isna(), _CATEGORICAL_MISSING_TOKEN
            )
            codes = pd.Categorical(filled).codes
            feature_matrix_X[:, row_index] = codes.astype(float)
            discrete_mask[row_index] = True
    return feature_matrix_X, discrete_mask


def profile_dataframe(
    df: pd.DataFrame,
    label_column: str = "isFraud",
    random_state: int = RANDOM_SEED,
) -> dict[str, dict[str, Any]]:
    """Per-column profile: dtype, detected_type, cardinality, missing_rate, distribution,
    and mutual information (MI) against label_column. MI is None for every column when label_column is absent.

    MI uses the fraud label for characterization only. Not used for segmentation or training.
    """
    feature_cols = [col for col in df.columns if col != label_column]

    if label_column in df.columns and feature_cols:
        target_vector_y = df[label_column].to_numpy()
        label_mask = ~pd.isna(target_vector_y)
        feature_matrix_X, discrete_mask = _prepare_features_for_mutual_info_classifier(
            df.loc[label_mask, :], feature_cols
        )
        mutual_info_scores = _mutual_info_classifier(
            feature_matrix_X,
            target_vector_y[label_mask],
            discrete_features=discrete_mask,
            random_state=random_state,
        )
        mutual_info_by_column = {
            col: float(mutual_info)
            for col, mutual_info in zip(feature_cols, mutual_info_scores)
        }
    else:
        mutual_info_by_column = {}

    profile: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        series = df[col]
        if _is_numeric_feature(series):
            detected_type = "numeric"
            distribution: dict[str, Any] = _numeric_distribution(series)
        else:
            detected_type = "categorical"
            distribution = _categorical_distribution(series)

        if col == label_column or col not in mutual_info_by_column:
            mutual_info: float | None = None
        else:
            mutual_info = mutual_info_by_column[col]

        profile[col] = {
            "dtype": str(series.dtype),
            "detected_type": detected_type,
            "cardinality": int(series.nunique(dropna=True)),
            "missing_rate": float(series.isna().mean()),
            "distribution": distribution,
            "mutual_info": mutual_info,
        }

    return profile
