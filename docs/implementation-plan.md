# Implementation Plan: Adaptive Encoding Anomaly Detector



---

## 1. Feature summary

A project at the intersection of constraint-based optimization, anomaly
detection, and neural-network interpretability.

1. **Encoding Selection Pipeline.** 
  
  Feature encoding is treated as an optimization problem using a Mixed-Integer Linear Program (MILP). MILP is an optimization algorithm that chooses among discrete options, one encoding per feature, while respecting linear constraints. In the context of this project, the MILP's objective is to minimize `α·loss + β·(dim/D_max) + γ·(1−xplain)`, a weighted sum of detection
   error, encoded size, and un-interpretability. The solver is **PuLP + HiGHS** (a Python modeling library + open-source solver). The primary detector on the IEEE-CIS Fraud Detection dataset is Local Outlier Factor (LOF), which flags outliers by identifying data points sitting in unusually sparse neighborhoods.
   The MILP-selected set of encodings is then compared against a plain
   one-hot encoded (uniform-OHE) baseline by measuring area under the precision–recall curve (PR-AUC). This was chosen because in the labeled dataset, fraud is rare (roughly 3.5% of transactions). PR-AUC was chosen because it ignores the large number of true-negative samples that another metric might reward. The pipeline also produces structured explanations for individual anomalies by running leave-one-feature-out (LOFO) on the top anomalies and features by mutual information (MI).

2. **Interpretability (representation geometry).** Two identical multilayer perceptrons (MLPs) are used. These are basic feedforward neural networks.
   `MLP-MILP` and `MLP-OHE` are trained on the two encoding regimes.
   They share the same architecture, the same random seed, and the same
   hyperparameters. Only the inputs differ. Activations are frozen
   for both, and superposition for each is measured three ways. Superposition is a state where a neural network is representing more features than it has neurons. It does so by letting them share overlapping directions at the cost of interference. Interference, broadly speaking, is when no clean separation of concept representations per-neuron can be identified via probe. 
   The three measurements used to evaluate superposition are: 
   (1) The interference matrix `Wᵀ·W` (summarized by the off-diagonal
   Frobenius norm)
   (2) Per-segment linear-probe accuracy, measuring for interference. 
   (3) Per-segment capacity - the fraction of each neuron's activation variance attributable to each segment. MLPs are controlled instruments for studying representation
   geometry. They are distinct from the anomaly detection.


**Processing Pipeline** 
1. Profile every feature
2. Sort features into five human-meaningful segments 
3. Score every (feature, encoding) pair on error, size, and interpretability 
4. Let a Multi-Integer Linear Program (MILP) pick the best
encoding per feature under a size budget 
5. Run anomaly detection on that encoding vs. a one-hot baseline 
6. Explain individual anomalies 
7. Use a probe to investigate whether and how the encoding reshapes a neural net's internal geometry.
---

## 2. Glossary


**Encoding**

- **Encoding**. How a raw data column is formatted into model input.
- **One-hot (OHE)**. A singular binary 0/1 column per category. Assumes no inherent order, but increases the number of columns.
- **Ordinal**. Map categories to ordered integers (0, 1, 2, …). This can introduce implicit ordering, and should only be used when the categories have a genuine and meaningful order.
- **Passthrough**. Leave a numeric column as-is, no transform.
- **Cardinality**. How many distinct values a column has.
- **Missingness / missing rate**. Fraction of rows where the value is
  absent.
- **Dimensionality / dimension**. How many columns the encoded data has.
More categories become more dimensions.

**Optimization**

- **MILP**. Mixed-Integer Linear Program. An optimization algorithm that chooses discrete options to minimize a linear objective under linear constraints. In this context, the linear objective is to pick one
encoding per feature.
- **Objective `α·loss + β·(dim/D_max) + γ·(1−xplain)`**. A weighted sum
of three costs, each scaled into [0, 1] so they are comparable.
  - `loss`. How poorly anomalies are detected under that encoding.
  - `dim/D_max`. Encoded size, normalized.
  - `1 − xplain`. Un-interpretability (`xplain` is the rubric score
    below. High interpretability is on one end, with low interpretability (with consequentially lower cost) on the other.).
  - `α, β, γ`. Weights that can be tuned to trade off between the three costs.
- **D_max**. The largest total dimensionality possible (sum, over
  features, of each feature's most-expensive encoding). It's a fixed
  normalization anchor that puts the size term on a 0–1 scale. It is
  not a constraint the solver must satisfy.
- **Interpretability rubric (`xplain`)**. A score in [0, 1] for assessing how human-readable each encoding is.
- **Dimension budget**. A hard cap on total encoded columns.
- **Infeasibility**. When no assignment satisfies every constraint
  (e.g., the budget is too tight to also meet the interpretability
  floor); the solver should report accordingly.
- **LP relaxation**. Allows for solving the easier continuous version of a problem by allowing fractional choices and rounding them. Faster, but potentially less accurate. Only trusted in this context if it matches the exact MILP on the MILP-relaxation benchmark.

**Anomaly Detection**

- **LOF (Local Outlier Factor)**. Unsupervised outlier detector. Scores a point by how much sparser its local neighborhood is than its neighbors'. A higher score means a data point is more of an outlier compared to similar neighborhoods of data points. The fraud label is never incorporated as input in its scoring assessment.
- **PR-AUC**. Area under the precision–recall curve; the preferred
  single number when positives (fraud) are rare, because it ignores the
  large, easy true-negative mass that ROC-AUC rewards.
- **Precision / Recall / F1**. At a chosen cutoff: precision = of the
  flagged, how many are fraud; recall = of the fraud, how many were
  flagged; F1 = their harmonic mean.
- **Stratified sample**. A subsample that preserves the class balance
  (here ~3.5% fraud), so a 10% slice still "looks like" the whole.
- **Mutual information (MI)**. A measure (≥ 0; 0 means independent) of
  how much knowing a feature reduces uncertainty about the label. Unlike
  correlation, it catches nonlinear association. Computed with
  `_mutual_info_classifier`. Used for *characterization only*. See the
  feature-profiler invariant.
- **k-means**. Clusters points into *k* groups, each point assigned to
  the nearest group average (centroid).
- **LOFO (leave-one-feature-out)**. Re-run the detector with one feature
  removed and watch how an anomaly's score moves; the change attributes
  "blame" to that feature. This is the explainer's engine.

**Interpretability & Neural Network Geometry**

- **MLP**. Multilayer perceptron. A model where input is passed to hidden layers, processed, and returned as output. Training is handled via backpropagation. This project uses a 2-hidden-layer ReLU net with a fraud head.
- **Superposition**. A network representing more features than it has
  neurons by letting features share overlapping directions; the price is
  *interference* (features bleed into each other). The "filing cabinet
  with more folders than drawers" framing in `docs/rationale.md`.
- **Interference matrix `Wᵀ·W`**. `W` is the weight matrix from inputs
  to the first hidden layer; `Wᵀ·W` ("W-transpose times W") is a square
  matrix whose off-diagonal entries measure how much two input features
  get written into the same neurons.
- **Frobenius norm**. The overall "size" of a matrix: the square root of
  the sum of all its squared entries. Taking it over just the
  off-diagonal entries collapses interference to a single scalar.
- **Linear probe**. A simple linear classifier trained to read a
  property (here: which of the five segments a feature belongs to) off
  the frozen hidden activations. Its accuracy describes how cleanly that
  property is encoded.
- **Capacity**. The fraction of a neuron's activation variance
  attributable to each segment. A practical operationalization inspired
  by but not equivalent to Scherlis et al. (2022), whose formal definition uses fractional embedding dimension.

**Reproducibility**

- **Random seed**. A fixed number (`RANDOM_SEED = 42`) that makes
  randomized steps (k-means init, sampling, MLP weight init) repeatable
  run-to-run.
---

## 3. Source inventory

| Source | Type | Identifier |
|---|---|---|
| Canonical spec | Markdown (tracked) | [docs/project-plan.md](project-plan.md) |
| Design-rationale notes | Markdown (tracked) | [docs/rationale.md](rationale.md) |
| Top-level README (reproducibility entry point) | Markdown (tracked) | [README.md](../README.md) |

---

## 4. Acceptance criteria

Status key: `[x]` met • `[~]` partially met • `[ ]` not started.

### Encoding Selection Pipeline

- [x] **Feature profile.** Produces a per-feature profile with
  detected type, cardinality (count of distinct values), distribution
  shape, missingness rate, and **mutual information** (how much the
  feature tells you about the fraud label; 0 = nothing) against the
  binary fraud label via `_mutual_info_classifier`. The MI value is used for
  *characterization only*: it helps rank which features carry signal, but
  LOF and the representation-analysis MLPs never see the label as a
  training target.
  **Evidence:** [src/feature_profiler.py](../src/feature_profiler.py)
  exposes `profile_dataframe`, `save_profile_as_json`, `load_profile_from_json`; the
  "characterization only" invariant is documented in the
  `profile_dataframe` docstring. Behavior pinned by 15 tests in
  [tests/test_profiler.py](../tests/test_profiler.py): coverage of every
  column, label MI is `None`, signal columns have higher MI than noise,
  detected-type/cardinality/missing-rate accuracy, reproducible MI under
  a fixed `random_state`, save/load round-trip. Smoke-run on a 50K
  stratified IEEE-CIS sample produced sensible MI rankings (V-features
  dominate); the output `outputs/feature_profile.json` is gitignored
  but recomputable from `data/raw/`.
- [x] **Feature segmentation.** Sorts features into the 5
  domain-labeled groups ("transaction amount", "identity/device",
  "behavioral frequency", "temporal/timing", "card/account") using a
  hybrid of rule-based logic (column-name patterns informed by
  cardinality and dtype) and **k-means** (clustering by nearest
  centroid) on the feature-profile vector for residuals. The leftover
  columns no rule matched. The representation analysis's per-segment
  metrics (see *Three superposition metrics* below) bind to this fixed
  5-segment vocabulary, which is why cluster IDs are never user-facing.
  **Evidence:** [src/feature_segmenter.py](../src/feature_segmenter.py)
  exposes `segment_features`, `save_segments`, `load_segments`, and the
  fixed `SEGMENT_LABELS` tuple. The rule pre-pass maps the IEEE-CIS
  conventions (`TransactionAmt`/`dist*`, `id_*`/`Device*`/`*emaildomain`,
  `C*`, `TransactionDT`/`D*`, `card*`/`addr*`/`M*`/`ProductCD`); residual
  columns (e.g. `V*`) are clustered with k-means
  (`n_clusters = min(5, n_residuals)`) on a profile vector
  (log-cardinality, missing rate, MI, `is_categorical`) standardized
  against the rule-mapped basis, and each cluster is mapped to the
  rule-segment whose centroid is nearest. Behavior pinned by 29 tests in
  [tests/test_segmenter.py](../tests/test_segmenter.py): vocabulary
  freeze, full coverage, label-column exclusion, every parametrized
  rule mapping, residual landing in the fixed vocabulary, default-seed
  reproducibility, JSON round-trip, and the no-residuals edge case.
  Smoke-run on a 50K stratified IEEE-CIS sample assigns all 433 feature
  columns into the five labels.
- [ ] **Encoding evaluation matrix.** Produces
  `outputs/evaluation_matrix.csv` for the top-N features by variance
  and/or MI, where N is pinned in `config/defaults.py` as
  `ENCODING_EVAL_MAX_FEATURES`. Features outside the top-N are recorded
  with the passthrough encoding only (left as-is, no transform). Each row
  in the matrix represents one (evaluated feature, encoding) pair and
  contains detection loss (LOF on a 10% stratified sample), encoded
  dimensionality (column count), and a rubric-based interpretability
  score. The rubric reads `config/ordinal_features.yaml`: features listed
  there receive 0.70 for the ordinal encoding row, and features not
  listed receive 0.30 (uncertain-ordering penalty, because imposing a
  fake order injects spurious structure); non-ordinal rubric rows are
  unaffected. Detection loss and rubric score are in [0, 1];
  dimensionality is a positive integer. **Evidence:** no
  `src/encoding_evaluator.py` exists yet; `ENCODING_EVAL_MAX_FEATURES`
  not yet defined in [config/defaults.py](../config/defaults.py).
- [ ] **MILP solver.** Solves the MILP
  `minimize α·loss + β·(dim/D_max) + γ·(1−xplain)` subject to three
  constraints (one encoding per feature; total dimension ≤ budget; an
  interpretability floor per segment) using PuLP + HiGHS. `D_max` is
  precomputed analytically from `evaluation_matrix.csv` before the solver
  runs. It scales the size term onto [0, 1] and is an anchor, not a
  constraint. Infeasibility (no assignment can satisfy every constraint)
  surfaces a clear error rather than a degenerate result. **Evidence:**
  no `src/milp_selector.py` exists yet; `pulp` and `highspy` declared in
  the `milp` optional extra in [pyproject.toml](../pyproject.toml).
- [ ] **LOF and baseline comparison.** Runs LOF (the anomaly
  detector) on the MILP-selected encoded space and on the uniform-OHE
  (one-hot-everything) baseline, reports **PR-AUC** as the headline metric
  (area under the precision–recall curve. The right summary when fraud is
  rare), and P/R/F1 at the `LOF_THRESHOLD_PERCENTILE` (= 96.5)
  prevalence-percentile threshold (i.e., flag the top ≈ 3.5% of
  transactions by LOF score, matching the true fraud rate). No SMOTE
  (synthetic minority oversampling), because LOF is an unsupervised
  density estimator and resampling would distort the densities it reads.
  **Evidence:** no `src/anomaly_detector.py` exists yet.
  `LOF_THRESHOLD_PERCENTILE` defined in
  [config/defaults.py](../config/defaults.py).
- [ ] **Structured explanations.** Produces a three-layer
  explanation for the top `LOFO_MAX_ANOMALIES` (= 100) anomalies by LOF
  score, running **leave-one-out** (LOFO. Remove one feature, re-score,
  see how much the anomaly score moves) over the top `LOFO_MAX_FEATURES`
  (= 50) by MI, on the 10% stratified sample from the encoding evaluator,
  in either `'feature'` or `'segment'` mode (`LOFO_MODE`). **Evidence:** no
  `src/explainer.py` exists yet. `LOFO_MAX_ANOMALIES`,
  `LOFO_MAX_FEATURES`, and `LOFO_MODE` defined in
  [config/defaults.py](../config/defaults.py).

### Interpretability claim (representation geometry)

- [ ] **Identical MLP architectures.** `MLP-MILP` and
  `MLP-OHE` (two multilayer perceptrons. Plain feedforward neural nets)
  train with identical architecture (2 hidden layers, ReLU activation,
  a binary fraud classification head), the same `RANDOM_SEED`, and the
  same hyperparameters; only the input encoding differs. Holding
  everything but the encoding fixed is what makes any geometry difference
  attributable to the encoding. **Evidence:** no
  `src/representation_analysis.py` exists yet; `torch` declared in the
  `geometry` optional extra in [pyproject.toml](../pyproject.toml).
- [ ] **Three superposition metrics.** Computed on frozen
  first-hidden-layer activations (the net's weights are held fixed while
  we measure): (1) interference matrix `Wᵀ·W` with the Frobenius norm of
  its off-diagonal terms as a single scalar summary of how much features
  share neurons; (2) per-segment linear-probe accuracy (how cleanly a
  simple linear readout recovers each of the 5 segments from the hidden
  activations); (3) per-segment capacity, the fraction of each neuron's
  activation variance attributable to each segment. Documentation
  acknowledges the capacity metric as a practical operationalization
  inspired by Scherlis et al. (2022), whose formal definition uses
  fractional embedding dimension. **Evidence:** no code yet.
- [ ] **Side-by-side comparison.** Reports all three metrics
  across `MLP-MILP` and `MLP-OHE`, published in
  `notebooks/03_representation_geometry.ipynb` and the Streamlit demo.
  **Evidence:** no notebook yet.

### Demo and write-up

- [ ] **Streamlit demo. Panels.** `demo/app.py` exposes: encoding
  decision table, per-anomaly explanation cards, baseline comparison
  panel, representation-geometry panel (interference matrix heatmaps +
  probe accuracy table), and α/β/γ + dimensionality-budget sliders (the
  three objective weights plus the size cap). Deployed to Streamlit
  Community Cloud (free). **Evidence:** only
  [demo/.gitkeep](../demo/.gitkeep); `streamlit` declared in the `demo`
  optional extra in [pyproject.toml](../pyproject.toml).
- [ ] **Streamlit demo. Encoding-decision source.** Encoding decisions
  load from a pre-computed grid of ~60 MILP solutions keyed by (α, β,
  γ, budget) by default, committed under
  `outputs/sample/encoding_decisions/` so the Streamlit Cloud demo
  loads them from a clean clone. Upgrade to **live LP relaxation**
  (solving the rounded continuous version in the browser) is gated on the
  MILP-relaxation benchmark: ≥95% of (α, β, γ, budget) grid points produce
  LP-rounded decision matrices identical to the MILP solution, no point
  violates the dimensionality budget after rounding, and every solve runs
  in <500ms. **Evidence:** no precomputed grid or LP benchmark yet.
- [ ] **Analysis write-up.** `docs/analysis.md` covers (a) LOF PR-AUC
  comparison and tradeoff curves, (b) representation-geometry findings
  with honest null-result reporting connected to Elhage et al. (2022)
  and Scherlis et al. (2022), and (c) rubric sensitivity analysis (each
  interpretability score perturbed ±10%, fraction of decisions that
  flip recorded). **Evidence:** `docs/analysis.md` not yet written. The
  `docs/` folder now holds this plan, the canonical spec, design
  rationale, and data-download instructions.

### Reproducibility

- [x] **Random seed pinned.** `RANDOM_SEED = 42` defined in
  `config/defaults.py`; every module is required to accept a
  `random_state` kwarg defaulting to this constant, so randomized steps
  repeat run-to-run. **Evidence:** `RANDOM_SEED` in
  [config/defaults.py](../config/defaults.py), pinned by
  `test_random_seed_is_42` in
  [tests/test_config_defaults.py](../tests/test_config_defaults.py).
  Per-module enforcement happens as each module is built.
- [ ] **30-minute reproduction.** A new contributor can reproduce all
  pipeline outputs in under 30 minutes by setting `RANDOM_SEED` in
  `config/defaults.py` and running the pipeline end-to-end.
  **Evidence:** the pipeline does not exist yet; reproducibility README
  finalization is an analysis-and-polish deliverable.

---

## 5. Definition of done

- [~] **Per-module tests pass.** Tests are added when the module is
  built (staggered, not retroactively). **Evidence:** the
  full suite runs green: `pytest -q` reports **53 passed** (profiler 15,
  segmenter 29, config 6, ordinal-yaml 3). `test_config_defaults.py` and
  `test_ordinal_features_yaml.py` cover the config artifacts;
  `[tool.pytest.ini_options]` configured in
  [pyproject.toml](../pyproject.toml). Tagged `[~]` because tests for
  the remaining modules (encoding evaluator onward) do not exist yet.
- [ ] **Expensive computations cache to `outputs/`.** The encoding
  evaluation grid, MILP solutions, and MLP training each cache their
  result so re-runs don't recompute. Sample artifacts needed by the demo
  live in `outputs/sample/`, the only carve-out from the `outputs/`
  gitignore rule. **Evidence:** no expensive computations exist yet; the
  carve-out (`outputs/*` with negations for `outputs/.gitkeep` and
  `outputs/sample/**`) is in [.gitignore](../.gitignore).
- [x] **Raw IEEE-CIS data is never committed.** **Evidence:** `*.zip`
  rule in [.gitignore](../.gitignore); the local
  `data/ieee-fraud-detection.zip` is gitignored;
  [README.md](../README.md) documents the Kaggle download.
- [~] **README.md finalized for reproducibility.** **Evidence:**
  [README.md](../README.md) exists at the repo root with installation,
  Kaggle download steps, and a repo-layout tour. Full reproducibility
  (≤30-min end-to-end run from a clean clone) requires the full pipeline
  to be built; this remains an analysis-and-polish deliverable.
- [ ] **Live Streamlit Community Cloud URL recorded in README.**
  **Evidence:** no demo yet.
- [ ] **Null-result reporting honored in `docs/analysis.md`.** If
  `MLP-MILP` and `MLP-OHE` exhibit equivalent superposition, that
  finding is reported and interpreted as evidence of network
  compensation, not omitted. (An honest null result is a legitimate
  finding here, not a failure.) **Evidence:** analysis-and-polish deliverable.

---

## 6. Current state

Single repo. Two phases are complete. The foundation phase delivered the
scaffold, the pinned config, and the dependency manifest, along with the
seeded ordinal annotation, an EDA notebook documenting those ordinal
choices, and the feature profiler; the segmentation phase delivered the
feature segmenter. Both modules are implemented and tested, and the full
suite is green (53 tests). The encoding evaluator is the next module to
build.

### Repo layout (what exists)

- [pyproject.toml](../pyproject.toml): package `adaptive-encoding-anomaly-detector` v0.1.0,
  Python ≥ 3.11, core deps `numpy/scipy/scikit-learn/joblib/pandas/pyyaml`,
  optional extras `dev = [pytest]`, `milp = [pulp, highspy]`,
  `geometry = [torch]`, `demo = [streamlit]`, pytest configured against
  `tests/`.
- [config/defaults.py](../config/defaults.py): `RANDOM_SEED = 42`,
  `LOF_THRESHOLD_PERCENTILE = 96.5`, `LOFO_MAX_ANOMALIES = 100`,
  `LOFO_MAX_FEATURES = 50`, `LOFO_MODE = "feature"`. The constant names
  are self-documenting; `LOF_THRESHOLD_PERCENTILE` carries a one-line
  note on the 96.5 / top-3.5% choice.
- [config/ordinal_features.yaml](../config/ordinal_features.yaml):
  two entries: `id_34` (4-value `match_status:N` sequence) and `M4`
  (3-value `M0`/`M1`/`M2` sequence). These are the only IEEE-CIS
  categoricals whose value strings literally encode an order. The defense
  for these (and the deliberate exclusion of `id_23` and `id_15`) lives
  in the "Ordinal candidate investigation" section of
  `notebooks/01_eda.ipynb`. Every other categorical falls back to the
  0.30 unlisted score in the encoding-evaluator rubric.
- [tests/test_config_defaults.py](../tests/test_config_defaults.py):
  6 tests pinning all five constants; one validates `LOFO_MODE ∈ {"feature", "segment"}`.
- [tests/test_ordinal_features_yaml.py](../tests/test_ordinal_features_yaml.py):
  3 tests validating schema. A top-level mapping from string feature
  name to a non-empty list of ordered values; an empty mapping is
  explicitly accepted as the early-stage state.
- [.gitignore](../.gitignore): gitignores `.venv/`, `*.zip`,
  `__pycache__/`, IDE files, build artifacts; `outputs/*` ignored
  except `outputs/.gitkeep` and `outputs/sample/**`; `CLAUDE.md` listed
  individually.
- [src/__init__.py](../src/__init__.py), [config/__init__.py](../config/__init__.py),
  [tests/__init__.py](../tests/__init__.py): empty package markers.
- [src/feature_profiler.py](../src/feature_profiler.py): feature-profiler
  implementation exposing `profile_dataframe`, `save_profile_as_json`, `load_profile_from_json`.
  The fraud label is used for MI characterization only; the module
  docstring states this invariant explicitly.
- [tests/test_profiler.py](../tests/test_profiler.py): 15 tests covering
  every column field plus the MI/label-isolation contract and JSON
  round-trip.
- [src/feature_segmenter.py](../src/feature_segmenter.py):
  feature-segmenter implementation exposing `segment_features`, `save_segments`,
  `load_segments`, and the fixed `SEGMENT_LABELS` tuple. Rule pre-pass on
  IEEE-CIS column-name conventions, k-means residual fallback in
  standardized profile-vector space; cluster centroids mapped back to the
  rule-segment whose centroid is nearest, so every column lands inside
  the fixed five-label vocabulary.
- [tests/test_segmenter.py](../tests/test_segmenter.py): 29 tests
  covering vocabulary freeze, full coverage, label-column exclusion,
  every parametrized rule mapping, residual-vocab containment,
  default-seed reproducibility, JSON round-trip, and the no-residuals
  edge case.
- [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb): foundation-phase EDA
  (exploratory data analysis) pass. Type distribution, categorical
  landscape, ordinal candidate investigation, narrative conclusions
  feeding the profiler and segmenter.
- [demo/.gitkeep](../demo/.gitkeep), [outputs/.gitkeep](../outputs/.gitkeep):
  directories exist as placeholders.
- [docs/](.): holds this plan plus the canonical spec
  [project-plan.md](project-plan.md) and
  design-rationale notes [rationale.md](rationale.md). All three
  are tracked. Data-download and reproduction instructions live in
  [README.md](../README.md) at the repo root.

### Local-only (gitignored) artifacts present

- `data/ieee-fraud-detection.zip`: raw dataset, present locally only.
- `outputs/feature_profile.json`: a cached feature-profiler profile from
  a smoke run; gitignored, recomputable from `data/raw/`.

### Module status

- **Feature Profiler:** complete.
  [src/feature_profiler.py](../src/feature_profiler.py) +
  [tests/test_profiler.py](../tests/test_profiler.py) (15 tests).
- **Feature Segmenter:** complete.
  [src/feature_segmenter.py](../src/feature_segmenter.py) +
  [tests/test_segmenter.py](../tests/test_segmenter.py) (29 tests).
- **Encoding Candidate Evaluator:** not started.
- **MILP Encoding Selector:** not started; PuLP + HiGHS
  declared in optional extra `milp`.
- **Anomaly Detector:** not started; PR-AUC threshold config
  pinned.
- **Structured Explainer:** not started; LOFO config pinned.
- **Representation Geometry:** not started; `torch` declared
  in optional extra `geometry`.
- **Streamlit demo:** not started; `streamlit` declared in optional
  extra `demo`.
- **Analysis write-up:** not started.

---

## 7. Remaining work

Ordered from the foundation phase through analysis and polish. Each
item's bold lead phrase pairs with the corresponding acceptance
criterion above by topic; where one item spans multiple criteria or
another remaining-work item, the body spells out the linkage by lead
phrase.

### Foundation

- [x] **Ordinal annotation.** Two entries seeded in
  [config/ordinal_features.yaml](../config/ordinal_features.yaml):
  `id_34` (`match_status:-1/0/1/2`) and `M4` (`M0/M1/M2`). Every other
  IEEE-CIS categorical falls back to the 0.30 unlisted-feature penalty
  in the encoding-evaluator rubric. Defense for the included pair and
  explicit exclusion of close calls (`id_23`, `id_15`) lives in the
  "Ordinal candidate investigation" section of
  [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb). Schema continues
  to pass
  [tests/test_ordinal_features_yaml.py](../tests/test_ordinal_features_yaml.py).
- [x] **Foundation EDA notebook.**
  [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) covers type
  distribution across the merged 590,540 × 434 frame, the categorical
  landscape (31 string columns, top-3 values and missingness), and an
  ordinal candidate investigation that justifies the two entries above.
  Final markdown cell orients the profiler and segmenter (feature
  profiler invariants, segment vocabulary mapping). The notebook is
  committed unexecuted; the raw CSVs it reads from `data/raw/` are
  gitignored.
- [x] **Feature-profiler implementation.**
  [src/feature_profiler.py](../src/feature_profiler.py) exposes
  `profile_dataframe(df, label_column='isFraud', random_state=RANDOM_SEED)`
  returning per-column dtype / detected_type / cardinality / missing
  rate / distribution / mutual information, plus `save_profile_as_json` and
  `load_profile_from_json`. The "label used for characterization only" invariant
  is stated in the `profile_dataframe` docstring. Behavior is pinned by 15 tests in
  [tests/test_profiler.py](../tests/test_profiler.py); MI reproducibility
  uses `RANDOM_SEED` from
  [config/defaults.py](../config/defaults.py). Smoke-run on a 50K
  stratified IEEE-CIS sample produces sensible MI rankings.

### Segmentation

- [x] **Feature-segmenter implementation.**
  [src/feature_segmenter.py](../src/feature_segmenter.py) exposes
  `segment_features(profile, label_column='isFraud', random_state=RANDOM_SEED)`
  returning `column` as `segment-label`, plus `save_segments` /
  `load_segments` for the `outputs/segment_assignments.json` artifact
  and the fixed `SEGMENT_LABELS = ("transaction amount",
  "identity/device", "behavioral frequency", "temporal/timing",
  "card/account")` tuple. Hybrid pipeline: a rule pre-pass mapping
  IEEE-CIS column-name conventions (informed by cardinality and dtype)
  followed by k-means on the feature-profile vector
  (log-cardinality, missing rate, MI, `is_categorical`) for residual
  columns, with `n_clusters = min(5, n_residuals)` standardized against
  the rule-mapped basis and each cluster mapped to the nearest
  rule-segment centroid. Behavior is pinned by 29 tests in
  [tests/test_segmenter.py](../tests/test_segmenter.py); reproducibility
  uses `RANDOM_SEED` from
  [config/defaults.py](../config/defaults.py). Smoke-run on a 50K
  stratified IEEE-CIS sample assigns all 433 feature columns into the
  five labels.

### Encoding evaluation grid

- [ ] **Encoding-evaluator implementation.** Implement
  `src/encoding_evaluator.py`. Prerequisite: add
  `ENCODING_EVAL_MAX_FEATURES` to
  [config/defaults.py](../config/defaults.py) and pin it in
  [tests/test_config_defaults.py](../tests/test_config_defaults.py).
  Then select the top-`ENCODING_EVAL_MAX_FEATURES` features by variance
  and/or MI for full encoding evaluation; remaining features are
  recorded with the passthrough encoding only. For each (evaluated
  feature, encoding) pair, run a small cross-validated LOF on a 10%
  stratified sample, and record loss, encoded dimensionality, and a
  rubric-based interpretability score. The rubric scorer reads
  [config/ordinal_features.yaml](../config/ordinal_features.yaml):
  features present receive 0.70 for the ordinal row, features absent
  receive 0.30. Cache all results to disk under `outputs/`. Add
  `tests/test_encoding_evaluator.py` verifying: scores ∈ [0, 1], dim
  values are positive integers, matrix shape is (evaluated-features ×
  encodings), top-N selection respects `ENCODING_EVAL_MAX_FEATURES`,
  the rubric scorer applies 0.70 to a listed feature fixture and 0.30
  to an unlisted feature fixture, and non-evaluated features carry the
  passthrough encoding only. Run against a small synthetic set so the
  test is fast.

### MILP

- [ ] **Install `milp` extra.** `pip install -e ".[milp]"` so `pulp`
  and `highspy` are available.
- [ ] **MILP-selector implementation.** Implement `src/milp_selector.py`.
  First, precompute `D_max` analytically from `evaluation_matrix.csv`
  (sum of max-dimensionality encoding per feature). Comment it as a
  normalization anchor, not a constraint. Then build the objective
  `α·loss + β·(dim/D_max) + γ·(1−xplain)` with all three terms in
  [0, 1]; add docstrings on each term. Constraints: one encoding per
  feature, total dim ≤ budget, interpretability floor per segment.
  Handle infeasibility with a clear error. Add
  `tests/test_milp_selector.py`.
- [ ] **Run MILP across the grid.** Run the solver across ~60 (α, β,
  γ, budget) grid points; persist all decision matrices to
  `outputs/sample/encoding_decisions/`. This is the precomputed grid
  the demo loads at runtime; the `outputs/sample/**` gitignore
  carve-out keeps the grid tracked. Supports the *Streamlit demo.
  Encoding-decision source* criterion above.
- [ ] **LP-relaxation benchmark gate.** For the same ~60 grid points,
  compare LP-rounded decisions (the rounded continuous solve) to the
  exact MILP decisions. Record: (a) the fraction of grid points whose
  full decision matrices are identical between LP-rounded and MILP,
  which is the gate criterion; (b) any post-rounding budget violations;
  (c) per-solve latency. If ≥95% of grid points produce identical
  decision matrices, no point violates the budget after rounding, and
  every solve runs in <500ms, mark the demo eligible for the live-LP
  upgrade. Otherwise the demo stays on the precomputed grid. Document
  the result. Satisfies the *Streamlit demo. Encoding-decision source*
  criterion by either route.

### Anomaly detection and baseline comparison

- [ ] **Pipeline deps for full dataset.** Install pipeline deps as
  needed (kaggle for dataset download is already documented in
  [README.md](../README.md)).
- [ ] **Anomaly-detector implementation.** Implement
  `src/anomaly_detector.py`. Apply MILP-selected encodings to the full
  dataset and run LOF; do the same for the uniform-OHE baseline. For
  P/R/F1, flag transactions with LOF scores above the
  `LOF_THRESHOLD_PERCENTILE`-th percentile (i.e., the top
  `100 − LOF_THRESHOLD_PERCENTILE` ≈ 3.5% of transactions); report
  PR-AUC as the headline metric across all thresholds. Produce the
  PR-curve comparison. Add `tests/test_anomaly_detector.py`: synthetic
  2-class smoke (10 features × 200 samples), verify output format and
  that the threshold produces ~3.5% positive rate. Output
  `notebooks/02_results.ipynb` + `outputs/metrics/`.

### Explanation layer

- [ ] **Explainer implementation.** Implement `src/explainer.py`. Build
  a leave-one-feature-out wrapper around LOF. Apply compute-scope caps
  from `config.defaults` (`LOFO_MAX_ANOMALIES = 100`,
  `LOFO_MAX_FEATURES = 50`); reuse the encoding-evaluator 10% stratified
  sample for the LOF re-computation; respect `LOFO_MODE` (`"feature"` or
  `"segment"`). Emit the three-layer explanation object as JSON +
  human-readable text. Add `tests/test_explainer.py`: verify
  three-layer output structure, valid floats, and that the
  near-zero-contribution edge case is handled. Output:
  `outputs/explanations/sample_explanations.json`.

### Representation geometry and demo

- [ ] **Install `geometry` extra.** `pip install -e ".[geometry]"`.
- [ ] **Representation-analysis implementation.** Implement
  `src/representation_analysis.py`. Train `MLP-MILP` and `MLP-OHE`:
  identical 2-hidden-layer ReLU architecture, binary fraud
  classification head, the same `RANDOM_SEED`, the same hyperparameters;
  only the encoded inputs differ. The MLP is not the anomaly detector;
  note this in the docstring. Compute the three metrics on frozen
  first-hidden-layer activations:
  (1) `Wᵀ·W` and its off-diagonal Frobenius norm (total interference),
  (2) per-segment linear probe accuracy across the 5 segments,
  (3) per-segment capacity as the fraction of each neuron's activation
  variance attributable to each segment. Document the
  Scherlis-et-al.-2022 operationalization caveat in the module
  docstring. Output `outputs/geometry/` and
  `notebooks/03_representation_geometry.ipynb`. Add
  `tests/test_representation_analysis.py`: verify `Wᵀ·W` is square and
  PSD (positive semidefinite, so off-diagonal Frobenius ≥ 0), per-segment
  probe accuracy ∈ [0,1], and per-neuron capacity sums ≈ 1.0 within
  floating-point tolerance, against a tiny synthetic MLP. Satisfies the
  three representation-geometry criteria above (identical architectures,
  three metrics, side-by-side comparison).
- [ ] **Install `demo` extra.** `pip install -e ".[demo]"`.
- [ ] **Build Streamlit app.** Build `demo/app.py`: encoding decision
  table, anomaly explanation cards, baseline comparison, and a
  representation-geometry panel (interference heatmaps side-by-side +
  probe accuracy table). Wire sliders to load from the precomputed
  grid produced by *Run MILP across the grid* above (default). If the
  *LP-relaxation benchmark gate* passed, add the live-LP-relaxation
  path; otherwise leave sliders snapping to grid points. Deploy to
  Streamlit Community Cloud and record the URL in the README.
  Satisfies the *Streamlit demo. Panels* and *Streamlit demo.
  Encoding-decision source* criteria above.

### Analysis and polish

- [ ] **Write analysis document.** Write `docs/analysis.md`: LOF PR-AUC
  comparison and tradeoff curves; representation-geometry findings with
  explicit honest null-result reporting; references to Elhage et al.
  (2022) *Toy Models of Superposition* and Scherlis et al. (2022)
  *Polysemanticity and Capacity*. Satisfies the *Analysis write-up*
  criterion partially and the *Null-result reporting honored in
  `docs/analysis.md`* definition-of-done item.
- [ ] **Rubric sensitivity analysis.** Perturb each interpretability
  score ±10%, re-run the full MILP on `evaluation_matrix.csv` per
  perturbation, and record the fraction of encoding decisions that
  flip. For every rubric row whose perturbation flips at least one
  decision, also record which encoding pairs are involved in the
  flips. That surfaces which γ-term thresholds are load-bearing rather
  than neutral (i.e., which interpretability scores actually change the
  optimizer's choice), and is itself an interpretability finding about
  the MILP objective structure. Publish both the flip-fraction stability
  table and the threshold-attribution detail in `docs/analysis.md`.
  Completes the *Analysis write-up* criterion.
- [ ] **Finalize README.** Finalize `README.md` for reproducibility
  (≤30-min reproduction from a clean clone given Kaggle credentials
  and `RANDOM_SEED = 42`). Satisfies the *30-minute reproduction*
  criterion and the *README.md finalized for reproducibility*
  definition-of-done item.

### Cross-cutting

- [ ] **Optional: blog-post draft.** Blog-post draft framing the
  geometry findings against the superposition literature. The
  canonical spec calls this out as optional in the analysis-and-polish phase.

---

## 8. Open questions and unresolved conflicts

None. Every row of the canonical spec's Open Design Decisions table is
resolved and reflected in this plan. The MILP LP-relaxation benchmark
gate is not an open question; It's an explicit, gated decision that
will resolve once the MILP selector exists.

---

## 9. External references

### Superposition and polysemanticity

- Elhage, N. et al. (2022). *Toy Models of Superposition.* Transformer
  Circuits Thread. https://transformer-circuits.pub/2022/toy_model/index.html
- Scherlis, A. et al. (2022). *Polysemanticity and Capacity in Neural
  Networks.* arXiv:2210.01892. https://ar5iv.labs.arxiv.org/html/2210.01892
- *200 COP in MI: Exploring Polysemanticity and Superposition*
  (LessWrong). https://www.lesswrong.com/posts/o6ptPu7arZrqRCxyz/200-cop-in-mi-exploring-polysemanticity-and-superposition
- *Superposition is Not Just Neuron Polysemanticity* (Alignment Forum).
  https://www.alignmentforum.org/posts/8EyCQKuWo6swZpagS/superposition-is-not-just-neuron-polysemanticity

### Multi-objective optimization

- Weighted Sum Method (ScienceDirect topic page).
  https://www.sciencedirect.com/topics/computer-science/weighted-sum-method

The canonical spec [docs/project-plan.md](project-plan.md)
also carries a *Reference Material* section with the academic citations
behind specific design decisions (Li & Fan 2026, arXiv:2510.17088, for
the heterogeneous-anomaly framing; Potdar et al. 2017, IJCA 175(4), as
informing — not dictating — the encoding interpretability ordering;
Breunig et al. 2000 for LOF, plus PuLP and HiGHS solver docs).
