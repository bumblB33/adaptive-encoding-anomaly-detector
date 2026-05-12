"""Tests for src/feature_profiler.py (Module 1).

These tests pin the contract of the feature profiler:

- Every input column appears in the output profile.
- Each entry carries detected_type, cardinality, missing_rate,
  distribution summary, and mutual_information.
- The label column is excluded from MI computation (its own
  mutual_information field is None) — the label is used for feature
  *characterization* only, never as a training target. This is the
  invariant in CLAUDE.md that prevents accidental leakage of the
  fraud label into LOF or the Module 7 MLPs.
- The MI computation is reproducible: same data + same seed => same
  scores.
- The profile dict is JSON-serializable round-trip via save/load.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import defaults
from src import feature_profiler


@pytest.fixture
def toy_frame():
    """Tiny mixed-type frame, large enough for stable MI estimates.

    `numeric_signal` is correlated with the label (perfectly above 0.5).
    `numeric_noise` is independent of the label.
    `category_signal` is correlated with the label.
    `category_noise` is independent.
    `numeric_with_nan` exercises missingness handling.
    """
    rng = np.random.default_rng(defaults.RANDOM_SEED)
    n = 500
    label = rng.integers(0, 2, size=n)
    numeric_signal = label + rng.normal(0, 0.1, size=n)
    numeric_noise = rng.normal(0, 1, size=n)
    category_signal = np.where(label == 1, "high", "low")
    category_noise = rng.choice(["a", "b", "c"], size=n)
    numeric_with_nan = rng.normal(0, 1, size=n).astype(float)
    numeric_with_nan[:50] = np.nan
    return pd.DataFrame(
        {
            "label": label,
            "numeric_signal": numeric_signal,
            "numeric_noise": numeric_noise,
            "category_signal": category_signal,
            "category_noise": category_noise,
            "numeric_with_nan": numeric_with_nan,
        }
    )


def test_profile_covers_every_column(toy_frame):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    assert set(profile.keys()) == set(toy_frame.columns)


def test_label_column_mutual_information_is_none(toy_frame):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    assert profile["label"]["mutual_information"] is None


def test_feature_columns_have_finite_mutual_information(toy_frame):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    for col in toy_frame.columns:
        if col == "label":
            continue
        mi = profile[col]["mutual_information"]
        assert mi is not None
        assert mi >= 0.0
        assert np.isfinite(mi)


def test_signal_columns_have_higher_mi_than_noise(toy_frame):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    assert (
        profile["numeric_signal"]["mutual_information"]
        > profile["numeric_noise"]["mutual_information"]
    )
    assert (
        profile["category_signal"]["mutual_information"]
        > profile["category_noise"]["mutual_information"]
    )


def test_detected_type_numeric_vs_categorical(toy_frame):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    assert profile["numeric_signal"]["detected_type"] == "numeric"
    assert profile["numeric_noise"]["detected_type"] == "numeric"
    assert profile["numeric_with_nan"]["detected_type"] == "numeric"
    assert profile["category_signal"]["detected_type"] == "categorical"
    assert profile["category_noise"]["detected_type"] == "categorical"


def test_cardinality_matches_nunique(toy_frame):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    for col in toy_frame.columns:
        assert profile[col]["cardinality"] == toy_frame[col].nunique(dropna=True)


def test_missing_rate_matches_isna_mean(toy_frame):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    for col in toy_frame.columns:
        expected = toy_frame[col].isna().mean()
        assert profile[col]["missing_rate"] == pytest.approx(expected)


def test_numeric_distribution_has_expected_keys(toy_frame):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    expected = {"min", "max", "mean", "std", "median"}
    assert set(profile["numeric_signal"]["distribution"].keys()) == expected


def test_categorical_distribution_has_top_values(toy_frame):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    dist = profile["category_signal"]["distribution"]
    assert "top_values" in dist
    assert isinstance(dist["top_values"], dict)
    assert len(dist["top_values"]) > 0
    # top_values is name -> count, counts must be positive ints
    for value, count in dist["top_values"].items():
        assert isinstance(value, str)
        assert isinstance(count, int)
        assert count > 0


def test_random_state_defaults_to_pinned_seed(toy_frame):
    profile_default = feature_profiler.profile_dataframe(
        toy_frame, label_column="label"
    )
    profile_explicit = feature_profiler.profile_dataframe(
        toy_frame, label_column="label", random_state=defaults.RANDOM_SEED
    )
    for col in toy_frame.columns:
        if col == "label":
            continue
        assert (
            profile_default[col]["mutual_information"]
            == profile_explicit[col]["mutual_information"]
        )


def test_random_state_reproducible(toy_frame):
    a = feature_profiler.profile_dataframe(
        toy_frame, label_column="label", random_state=123
    )
    b = feature_profiler.profile_dataframe(
        toy_frame, label_column="label", random_state=123
    )
    for col in toy_frame.columns:
        if col == "label":
            continue
        assert a[col]["mutual_information"] == b[col]["mutual_information"]


def test_handles_missing_values_without_error(toy_frame):
    # numeric_with_nan has 10% NaN; profiler must not raise.
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    assert profile["numeric_with_nan"]["missing_rate"] == pytest.approx(50 / 500)
    assert profile["numeric_with_nan"]["mutual_information"] is not None


def test_save_load_roundtrip(toy_frame, tmp_path):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    out = tmp_path / "feature_profile.json"
    feature_profiler.save_profile(profile, out)
    assert out.exists()
    reloaded = feature_profiler.load_profile(out)
    assert reloaded == profile


def test_save_writes_valid_json(toy_frame, tmp_path):
    profile = feature_profiler.profile_dataframe(toy_frame, label_column="label")
    out = tmp_path / "feature_profile.json"
    feature_profiler.save_profile(profile, out)
    # Must be readable by plain json.load — no numpy types leaking through.
    with open(out) as f:
        data = json.load(f)
    assert set(data.keys()) == set(toy_frame.columns)


def test_label_column_can_be_absent(toy_frame):
    """If label_column is missing, profiler still runs and skips MI."""
    no_label = toy_frame.drop(columns=["label"])
    profile = feature_profiler.profile_dataframe(no_label, label_column="label")
    assert set(profile.keys()) == set(no_label.columns)
    for col in no_label.columns:
        assert profile[col]["mutual_information"] is None
