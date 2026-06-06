# Adaptive Encoding Anomaly Detector

Production machine learning pipelines may treat feature encoding as a preprocessing afterthought: pick one-hot, get on with the model. This project asks a different question. If encoding choices upstream of a neural network change what the network can represent, then choosing those encodings deserves the same care we give to model selection. Here, encoding selection is posed as a Mixed-Integer Linear Program, and the resulting decision matrix doubles as an auditable explanation of why each feature was encoded the way it was.


## Two claims, separable on purpose

**Feature Engineering Selection.** Categorical encoding (one-hot, ordinal, target, binary, binning, passthrough) is treated as a constraint-aware optimization rather than a default. The MILP minimizes a three-term objective: detection loss, encoded dimensionality, and an interpretability rubric score. Each term is normalized onto the same scale so the weights are directly comparable. Local Outlier Factor serves as the primary anomaly detector, and the MILP-selected encoding regime is benchmarked against a uniform one-hot baseline on Precision-Recall Area Under Curve.

**Interpretability Impact.** Two multilayer perceptrons, identical in architecture, random seed, and hyperparameters, are trained on the two encoding regimes. From the first hidden layer of each, three measurements of superposition are taken: the interference matrix and its off-diagonal Frobenius norm, per-segment linear probe accuracy, and per-segment capacity. The MLP is not the anomaly detector. It's a controlled instrument for asking whether upstream encoding choices leave a measurable fingerprint on representation geometry.

The MILP work holds even if the geometry experiment returns a null result, and both are worth reporting.

## Dataset

IEEE-CIS Fraud Detection from Kaggle. Roughly 590,000 transactions with more than 400 raw features (email domains, device types, card networks, product codes, transaction type codes) and ground-truth fraud labels. Raw data is not committed to the repository.

## Stack

Python 3.11. Core dependencies: numpy, scipy, scikit-learn, pandas, joblib, pyyaml. Optional extras installed per phase: pytest for tests, PuLP and HiGHS for the MILP, PyTorch for the geometry MLPs, Streamlit for the demo.

## Reproducing the pipeline

The reproduction target is under thirty minutes from a clean clone, given Kaggle credentials. All randomness is seeded by `RANDOM_SEED = 42` in `config/defaults.py`.

1. Clone and install:

   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,milp,geometry,demo]"
   ```

2. Download the dataset. Create a Kaggle account, accept the competition rules at https://www.kaggle.com/competitions/ieee-fraud-detection/rules, place your API token at `~/.kaggle/kaggle.json` (`chmod 600`), then:

   ```
   pip install kaggle
   kaggle competitions download -c ieee-fraud-detection -p data/
   unzip -d data/ data/ieee-fraud-detection.zip
   ```

   The pipeline uses only `train_transaction.csv` and `train_identity.csv`.

3. Run the modules in order. The pipeline is under active build; see `docs/implementation-plan.md` for current status.

## Repository layout

- `src/` holds the analysis modules: feature profiler, segmenter, encoding evaluator, MILP selector, anomaly detector, explainer, representation geometry.
- `config/` pins reproducibility constants (`RANDOM_SEED`, LOF thresholds, compute caps).
- `tests/` holds per-module unit tests, added as each module is built.
- `demo/` holds the Streamlit application.
- `docs/` holds the canonical spec (`project-plan.md`), the living implementation plan (`implementation-plan.md`), and short design-rationale notes (`rationale.md`).

## References

- Elhage et al. (2022). Toy Models of Superposition. Transformer Circuits Thread.
- Scherlis et al. (2022). Polysemanticity and Capacity in Neural Networks. arXiv:2210.01892.
- Li and Fan (2026). Explainable Heterogeneous Anomaly Detection in Financial Networks via Adaptive Expert Routing. arXiv:2510.17088.
- Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers. ACM SIGMOD.
- Potdar et al. (2017). A Comparative Study of Categorical Variable Encoding Techniques for Neural Network Classifiers. International Journal of Computer Applications, 175(4), 7–9.
