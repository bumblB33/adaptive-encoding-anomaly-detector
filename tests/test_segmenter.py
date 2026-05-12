"""Tests for src/feature_segmenter.py (Module 2).

These tests pin the contract of the feature segmenter:

- Every IEEE-CIS column from the input profile receives one of the
  five domain labels in ``SEGMENT_LABELS``. The label vocabulary is
  fixed because Module 7's per-segment metrics bind to it.
- Known IEEE-CIS column-name patterns are mapped by rules
  (``TransactionAmt`` → ``"transaction amount"``, ``id_*`` →
  ``"identity/device"``, ``C*`` → ``"behavioral frequency"``,
  ``D*``/``TransactionDT`` → ``"temporal/timing"``,
  ``card*``/``addr*``/``M*``/``ProductCD`` → ``"card/account"``).
- Residual columns (e.g. the anonymized ``V*`` engineered features)
  are clustered with k-means on the feature-profile vector and each
  cluster is mapped to the rule-segment whose centroid is closest in
  profile-vector space, so every column always lands inside the
  fixed vocabulary.
- The label column (``isFraud``) is excluded from segment assignments
  — it is a target, not a feature.
- Output is reproducible under a fixed ``random_state`` (defaulting
  to ``config.defaults.RANDOM_SEED``) and JSON-serializable round-trip.
"""

import json

import pytest

from config import defaults
from src import feature_segmenter


_RULE_FIXTURE_COLUMNS = [
    # transaction amount
    ("TransactionAmt", "numeric", 20000, 0.0, 1.0),
    ("dist1", "numeric", 1500, 0.6, 0.05),
    # identity/device
    ("id_01", "numeric", 90, 0.7, 0.04),
    ("id_31", "categorical", 130, 0.7, 0.03),
    ("DeviceType", "categorical", 2, 0.7, 0.02),
    ("DeviceInfo", "categorical", 1700, 0.8, 0.04),
    ("P_emaildomain", "categorical", 60, 0.15, 0.02),
    ("R_emaildomain", "categorical", 60, 0.75, 0.01),
    # behavioral frequency
    ("C1", "numeric", 1500, 0.0, 0.03),
    ("C13", "numeric", 1700, 0.0, 0.04),
    # temporal/timing
    ("TransactionDT", "numeric", 580000, 0.0, 0.01),
    ("D1", "numeric", 640, 0.2, 0.02),
    ("D15", "numeric", 860, 0.15, 0.01),
    # card/account
    ("ProductCD", "categorical", 5, 0.0, 0.05),
    ("card1", "numeric", 13000, 0.0, 0.05),
    ("card4", "categorical", 4, 0.0, 0.04),
    ("addr1", "numeric", 330, 0.1, 0.02),
    ("M4", "categorical", 3, 0.5, 0.02),
]

_RESIDUAL_FIXTURE_COLUMNS = [
    ("V1", "numeric", 2, 0.5, 0.01),
    ("V100", "numeric", 30, 0.4, 0.02),
    ("V200", "numeric", 70, 0.4, 0.03),
    ("V300", "numeric", 50, 0.5, 0.02),
]


def _entry(detected_type: str, cardinality: int, missing: float, mi: float):
    if detected_type == "numeric":
        distribution = {
            "min": 0.0,
            "max": float(cardinality),
            "mean": 0.0,
            "std": 1.0,
            "median": 0.0,
        }
    else:
        distribution = {"top_values": {"a": 1}}
    return {
        "dtype": "float64" if detected_type == "numeric" else "object",
        "detected_type": detected_type,
        "cardinality": cardinality,
        "missing_rate": missing,
        "distribution": distribution,
        "mutual_information": mi,
    }


@pytest.fixture
def ieee_profile():
    """Profile fixture mirroring the IEEE-CIS column landscape."""
    profile = {
        "isFraud": {
            "dtype": "int64",
            "detected_type": "numeric",
            "cardinality": 2,
            "missing_rate": 0.0,
            "distribution": {
                "min": 0.0,
                "max": 1.0,
                "mean": 0.035,
                "std": 0.18,
                "median": 0.0,
            },
            "mutual_information": None,
        },
    }
    for name, dtype, card, miss, mi in _RULE_FIXTURE_COLUMNS:
        profile[name] = _entry(dtype, card, miss, mi)
    for name, dtype, card, miss, mi in _RESIDUAL_FIXTURE_COLUMNS:
        profile[name] = _entry(dtype, card, miss, mi)
    return profile


def test_segment_labels_vocabulary_is_fixed_five():
    """Module 7 binds to this exact vocabulary; freeze the strings."""
    assert feature_segmenter.SEGMENT_LABELS == (
        "transaction amount",
        "identity/device",
        "behavioral frequency",
        "temporal/timing",
        "card/account",
    )


def test_every_feature_column_is_assigned(ieee_profile):
    assignments = feature_segmenter.segment_features(ieee_profile)
    expected = set(ieee_profile.keys()) - {"isFraud"}
    assert set(assignments.keys()) == expected


def test_label_column_is_excluded(ieee_profile):
    assignments = feature_segmenter.segment_features(ieee_profile)
    assert "isFraud" not in assignments


def test_every_assignment_is_in_the_fixed_vocabulary(ieee_profile):
    assignments = feature_segmenter.segment_features(ieee_profile)
    valid = set(feature_segmenter.SEGMENT_LABELS)
    for col, label in assignments.items():
        assert label in valid, f"{col} mapped to unknown label {label!r}"


@pytest.mark.parametrize(
    "column,expected_label",
    [
        ("TransactionAmt", "transaction amount"),
        ("dist1", "transaction amount"),
        ("id_01", "identity/device"),
        ("id_31", "identity/device"),
        ("DeviceType", "identity/device"),
        ("DeviceInfo", "identity/device"),
        ("P_emaildomain", "identity/device"),
        ("R_emaildomain", "identity/device"),
        ("C1", "behavioral frequency"),
        ("C13", "behavioral frequency"),
        ("TransactionDT", "temporal/timing"),
        ("D1", "temporal/timing"),
        ("D15", "temporal/timing"),
        ("ProductCD", "card/account"),
        ("card1", "card/account"),
        ("card4", "card/account"),
        ("addr1", "card/account"),
        ("M4", "card/account"),
    ],
)
def test_rule_based_assignments(ieee_profile, column, expected_label):
    assignments = feature_segmenter.segment_features(ieee_profile)
    assert assignments[column] == expected_label


def test_residual_v_features_are_assigned_into_the_vocabulary(ieee_profile):
    assignments = feature_segmenter.segment_features(ieee_profile)
    valid = set(feature_segmenter.SEGMENT_LABELS)
    for col in ("V1", "V100", "V200", "V300"):
        assert col in assignments
        assert assignments[col] in valid


def test_random_state_defaults_to_pinned_seed(ieee_profile):
    default = feature_segmenter.segment_features(ieee_profile)
    explicit = feature_segmenter.segment_features(
        ieee_profile, random_state=defaults.RANDOM_SEED
    )
    assert default == explicit


def test_segment_features_is_reproducible(ieee_profile):
    a = feature_segmenter.segment_features(ieee_profile, random_state=123)
    b = feature_segmenter.segment_features(ieee_profile, random_state=123)
    assert a == b


def test_save_load_roundtrip(ieee_profile, tmp_path):
    assignments = feature_segmenter.segment_features(ieee_profile)
    out = tmp_path / "segment_assignments.json"
    feature_segmenter.save_segments(assignments, out)
    assert out.exists()
    reloaded = feature_segmenter.load_segments(out)
    assert reloaded == assignments


def test_save_writes_valid_json(ieee_profile, tmp_path):
    assignments = feature_segmenter.segment_features(ieee_profile)
    out = tmp_path / "segment_assignments.json"
    feature_segmenter.save_segments(assignments, out)
    with open(out) as f:
        data = json.load(f)
    assert data == assignments


def test_label_column_can_be_absent(ieee_profile):
    """If the profile has no label column, segmenter still works."""
    no_label = {k: v for k, v in ieee_profile.items() if k != "isFraud"}
    assignments = feature_segmenter.segment_features(no_label)
    assert set(assignments.keys()) == set(no_label.keys())


def test_unknown_column_with_no_residuals_still_ok():
    """Smallest valid input: just rule-mapped columns, no V-features."""
    profile = {
        "TransactionAmt": _entry("numeric", 20000, 0.0, 1.0),
        "C1": _entry("numeric", 1500, 0.0, 0.03),
        "card1": _entry("numeric", 13000, 0.0, 0.05),
    }
    assignments = feature_segmenter.segment_features(profile)
    assert assignments == {
        "TransactionAmt": "transaction amount",
        "C1": "behavioral frequency",
        "card1": "card/account",
    }
