# Adaptive Encoding Anomaly Detector

This project asks whether the way features are encoded before a model sees them changes what an anomaly detector finds and what a neural network learns. Encoding selection is framed as an optimization problem, so the choice for each feature can be audited.

The design is in [docs/design.md](docs/design.md). Terms are in [docs/glossary.md](docs/glossary.md). A six-week learning plan is in [docs/student-project.md](docs/student-project.md). This file describes what exists in the repository now.

## Status

| # | Stage | Status | Location |
|---|-------|--------|----------|
| 1 | Feature Profiler | Built | `src/feature_profiler.py` |
| 2 | Feature Segmenter | Built | `src/feature_segmenter.py` |
| 3 | Encoding Candidate Evaluator | Planned | - |
| 4 | MILP Encoding Selector | Planned | - |
| 5 | Anomaly Detector (LOF vs one-hot baseline) | Planned | - |
| 6 | Structured Explainer (LOFO) | Planned | - |
| 7 | Representation Geometry Analysis | Planned | - |
| 8 | Streamlit App | Planned | - |

## What is built

**Feature Profiler.** `profile_dataframe` returns, for each column, the dtype, detected type (numeric or categorical), cardinality, missing rate, a distribution summary, and mutual information (MI) with the label column. MI uses the label for characterization only. It does not feed segmentation or training.

**Feature Segmenter.** `segment_features` assigns each column to one of five segments: transaction amount, identity/device, behavioral frequency, temporal/timing, card/account. Column-name rules assign known IEEE-CIS columns. Remaining columns are clustered with k-means on their profile and mapped to the nearest rule-based segment.

**Configuration.**
- `config/defaults.py` holds `RANDOM_SEED = 42`, `LOF_THRESHOLD_PERCENTILE = 96.5`, and the LOFO limits.
- `config/ordinal_features.yaml` lists categoricals with a confirmed order.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) from Kaggle. The training set has 590,540 transactions, of which 20,663 are fraud (3.499%). These counts come from secondary sources, not from the Kaggle page. Check them with `pandas` after download.

- The data is split into a transaction table and an identity table, joined on `TransactionID`. Not every transaction has an identity row.
- The label is `isFraud`. Kaggle's host says labels derive from reported chargebacks, and transactions not reported within 120 days are labelled legitimate. Some fraud is therefore labelled legitimate.
- The merged tables have more than 400 provided columns. The `V` columns are engineered by Vesta, not raw.
- `TransactionDT` is a time offset, not a timestamp.

Raw data is not committed to the repository.

## Setup

1. Clone and install:

   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,milp,geometry,demo]"
   pytest
   ```

2. Download the data.
   - Accept the competition rules in a browser at https://www.kaggle.com/competitions/ieee-fraud-detection/rules.
   - Authenticate the Kaggle CLI as described in the [Kaggle CLI docs](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md).
   - Then run:

   ```
   pip install kaggle
   kaggle competitions download ieee-fraud-detection -p data --unzip
   ```

   These commands follow the current Kaggle CLI docs. They have not been run for this project. `--unzip` deletes the zip file.

The pipeline uses `train_transaction.csv` and `train_identity.csv` only.

## Stack

- Python 3.11 or later
- numpy, scipy, scikit-learn, pandas, joblib, pyyaml
- pytest (extra `dev`)
- PuLP and HiGHS (extra `milp`)
- PyTorch (extra `geometry`)
- Streamlit (extra `demo`)
