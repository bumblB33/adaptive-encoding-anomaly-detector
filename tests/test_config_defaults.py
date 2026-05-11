"""Pin every named constant in config/defaults.py.

These constants are project invariants — they drive seeding,
thresholding, and compute-scope decisions across every downstream
module. A silent change to any value would break reproducibility
or shift the headline metric, so tests catch drift immediately.
"""

from config import defaults


def test_random_seed_is_42():
    assert defaults.RANDOM_SEED == 42


def test_lof_threshold_percentile_matches_prevalence():
    # IEEE-CIS positive rate ~3.5% => flag top 3.5% by LOF score
    # => 96.5th percentile cutoff.
    assert defaults.LOF_THRESHOLD_PERCENTILE == 96.5


def test_lofo_max_anomalies():
    assert defaults.LOFO_MAX_ANOMALIES == 100


def test_lofo_max_features():
    assert defaults.LOFO_MAX_FEATURES == 50


def test_lofo_mode_default_is_feature():
    assert defaults.LOFO_MODE == "feature"


def test_lofo_mode_in_supported_set():
    assert defaults.LOFO_MODE in {"feature", "segment"}
