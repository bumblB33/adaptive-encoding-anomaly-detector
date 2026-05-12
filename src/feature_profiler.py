"""Module 1: per-column profile of a DataFrame for encoding decisions.

For each column, emit:
- ``dtype`` — pandas dtype string
- ``detected_type`` — ``"numeric"`` or ``"categorical"``
- ``cardinality`` — ``nunique(dropna=True)``
- ``missing_rate`` — ``isna().mean()``
- ``distribution`` — summary stats (numeric: min/max/mean/std/median;
  categorical: ``top_values`` mapping)
- ``mutual_information`` — MI against the label column via
  ``sklearn.feature_selection.mutual_info_classif``

The fraud label is used here for *feature characterization only*.
Mutual information helps prioritize which features carry signal
for Modules 2 and 3. LOF (Module 5) and the Module 7 MLPs never
see the label as a training target. The profile is descriptive
metadata, not training input.

See docs/portfolio_project_plan.md "Module 1" for the spec and
CLAUDE.md "Anomaly signal proxy (Module 1)" for the invariant.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

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


def _prepare_features_for_mi(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Encode features for ``mutual_info_classif``.

    Numeric features keep their values (NaN imputed with the column
    median, or 0.0 if every value is NaN). Categorical features are
    label-encoded after a ``__MISSING__`` sentinel fills NaN. The
    returned ``discrete_mask`` tells sklearn which features to treat
    as discrete (the integer-coded categoricals).
    """
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
            filled = s.astype("object").where(
                ~s.isna(), _CATEGORICAL_MISSING_TOKEN
            )
            codes = pd.Categorical(filled).codes
            X[:, j] = codes.astype(float)
            discrete_mask[j] = True
    return X, discrete_mask


def profile_dataframe(
    df: pd.DataFrame,
    label_column: str = "isFraud",
    random_state: int = RANDOM_SEED,
) -> dict[str, dict[str, Any]]:
    """Build the per-column profile.

    Parameters
    ----------
    df:
        Wide frame of features. May or may not contain ``label_column``.
    label_column:
        Column name to use as the target for MI computation. If the
        column is absent, MI is skipped for every feature (all entries
        get ``mutual_information=None``).
    random_state:
        Seed for ``mutual_info_classif``'s kNN-based estimator,
        defaulting to ``config.defaults.RANDOM_SEED``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Keyed by column name. Each entry has the fields documented in
        the module docstring.
    """
    feature_cols = [c for c in df.columns if c != label_column]

    if label_column in df.columns and feature_cols:
        y = df[label_column].to_numpy()
        label_mask = ~pd.isna(y)
        X, discrete_mask = _prepare_features_for_mi(
            df.loc[label_mask, :], feature_cols
        )
        mi_scores = mutual_info_classif(
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


def save_profile(profile: dict[str, dict[str, Any]], path: Path | str) -> None:
    """Persist profile as JSON. Parent directories are created as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, sort_keys=True)


def load_profile(path: Path | str) -> dict[str, dict[str, Any]]:
    """Inverse of ``save_profile``."""
    with open(path) as f:
        return json.load(f)
