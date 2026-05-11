"""Project-wide named constants.

Every module imports from here so behavior is centrally configurable
and reproducible across runs. Values are pinned by tests/test_config_defaults.py.

- RANDOM_SEED: seeds k-means (Module 2), stratified splits and LOF
  subsampling (Module 3), and MLP training (Module 7). Every module
  accepts a random_state kwarg defaulting to this constant.
- LOF_THRESHOLD_PERCENTILE: flag the top (100 - LOF_THRESHOLD_PERCENTILE)
  percent of transactions by LOF score for P/R/F1 reporting. 96.5 matches
  the IEEE-CIS ~3.5% positive rate, so the prevalence-percentile threshold
  aligns with the true class balance without resampling.
- LOFO_*: bound the Module 6 leave-one-feature-out explainer's compute
  cost. LOFO re-runs reuse the 10% stratified sample from Module 3 rather
  than the full ~590K rows.
"""

RANDOM_SEED = 42

LOF_THRESHOLD_PERCENTILE = 96.5

LOFO_MAX_ANOMALIES = 100
LOFO_MAX_FEATURES = 50
LOFO_MODE = "feature"
