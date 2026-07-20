from config import defaults


def test_random_seed_is_42():
    assert defaults.RANDOM_SEED == 42


def test_lof_threshold_percentile_matches_prevalence():
    assert defaults.LOF_THRESHOLD_PERCENTILE == 96.5


def test_lofo_max_anomalies():
    assert defaults.LOFO_MAX_ANOMALIES == 100


def test_lofo_max_features():
    assert defaults.LOFO_MAX_FEATURES == 50


def test_lofo_mode_default_is_feature():
    assert defaults.LOFO_MODE == "feature"


def test_lofo_mode_in_supported_set():
    assert defaults.LOFO_MODE in {"feature", "segment"}
