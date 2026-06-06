# Adaptive Encoding Anomaly Detector

## Summary
 
## Thesis: Encoding selection is a first-class, optimized, and auditable part of the anomaly detection pipeline, not a preprocessing assumption. The encoding decision matrix is itself an explanation layer.

1. **Encoding Selection pipeline.** How each raw feature becomes model input. Framed as a Mixed-Integer Linear Program (MILP) with an auditable decision matrix, rather than with a default-dtype preprocessing assumption. LOF is the primary anomaly detector on the IEEE-CIS Fraud Detection dataset. The MILP-selected encoding is compared against a uniform one-hot (OHE) baseline on PR-AUC.
2. **Representation Geometry & Interpretability.** Two identical MLPs (`MLP-MILP`, `MLP-OHE`), differing only in input encoding, measure superposition three ways: the interference matrix `W^T·W`, per-segment linear-probe accuracy, and per-segment capacity. MLP does not operate as an anomaly detector. It's a controlled instrument for studying representation geometry.

**Deliverables.** Full pipeline build and visualization incorporating a uniform-OHE baseline comparison, a structured per-anomaly explainer, a representation-geometry analysis, and a Streamlit demo.


## Motivation

This project synthesizes research inspired by
**Explainable Heterogeneous Anomaly Detection** (Li & Fan, 2026, arXiv:2510.17088v2). This paper posits that financial anomalies arise from heterogeneous mechanisms (price shocks, liquidity freezes, contagion cascades, momentum reversals). The paper investigates this theory with a mixture-of-experts architecture in which routing weights serve as interpretable proxies for mechanism attribution. The authors call this "architectural interpretability" to distinguish it from post-hoc explanation. Combined with the neuron superposition hypothesis (Elhage et al., 2022), this motivates the question: 

If different anomaly mechanisms have different feature signatures, and encoding choice materially affects how well those signatures are preserved, then encoding selection should be a first-class, optimized, and auditable part of the anomaly detection pipeline, not a preprocessing assumption. The encoding decision matrix is itself an explanation layer.

**Feature Encoding Tradeoffs in ML:** Feature encoding choice (one-hot, ordinal, target, binary, binning, passthrough) significantly affects model performance and coefficient interpretability, particularly in linear and neural architectures. Production ML pipelines may choose to apply a single encoding scheme to all categorical features based on dtype, rather than evaluating each feature's cardinality, ordinality, or relationship to the target. This project is designed to investigate whether that convenience leaves meaningful performance and interpretability gains on the table.

**Project Proposal:** Encoding selection as a Mixed-Integer Linear Program. Designed to convert feature engineering and preprocessing steps into a principled, constraint-aware optimization. 

---

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle, free with account)

- 400+ raw features: email domains, device types, card networks, product codes, transaction type codes.
- A mix of high-cardinality nominal, low-cardinality ordinal, binary flag, and continuous features. Enough variety that encoding choice matters.
- Ground-truth fraud labels enable concrete precision/recall evaluation, not just unsupervised scoring.
- A well-documented benchmark with community reference material.

Rather than trying to generalize results across additional datasets, this project is focused on evaluating the resultant representation geometry. A small MLP is trained on both encoding methods, and its hidden layer representations are analyzed for superposition.

**Data Note:** `README.md` contains download instructions.

---

## System Architecture

```
Raw tabular financial dataset (mixed feature types)
  - Feature Profiler
  - Feature Segmenter
  - Encoding Candidate Evaluator
  - MILP Encoding Selector
  - Anomaly Detector
  - Structured Explainer
  - Representation Geometry Analysis
  - Streamlit App (demo)
```

### Feature Profiler

- Detects data type (continuous, nominal, ordinal, datetime); computes cardinality, distribution shape, and missingness rate.
- Computes mutual information with the binary fraud label via `_mutual_info_classifier`. **Invariant:** the label is used for feature characterization only. LOF and the representation-analysis MLPs never see it as a training target.
- **Output:** `feature_profile.json`

### Feature Segmenter

- Clusters features by profile into 5 domain-labeled segments: "transaction amount", "identity/device", "behavioral frequency", "temporal/timing", "card/account".
- **Output:** `segment_assignments.json`

### Encoding Candidate Evaluator

- For each feature × encoding pair, computes encoded dimensionality (known analytically), detection loss on a held-out fold (LOF, subsampled), and a rubric-based interpretability score.
- Results are cached to disk.
- **Output:** `evaluation_matrix.csv`

### MILP Encoding Selector

- Decision variables: `x[feature, encoding] ∈ {0, 1}`.
- Pre-computation: `D_max` = sum of the maximum encoding dimensionality per feature, computed analytically from `evaluation_matrix.csv`.
- Objective: minimize `α·loss + β·(dim/D_max) + γ·(1−xplain)`, all three terms in [0, 1] (`loss` ∈ [0, 1] naturally; `dim/D_max` ∈ [0, 1] by construction; `1−xplain` ∈ [0, 1] by rubric definition).
- Constraints: exactly one encoding per feature; total post-encoding dimensionality ≤ budget; a minimum explainability floor per segment.
- Solver: PuLP + HiGHS.
- **Output:** `encoding_decisions.json` (auditable decision matrix)

### Anomaly Detector

- Applies the selected encodings to the full dataset and runs LOF on the MILP-selected encoded feature space; runs the identical pipeline with uniform OHE as the baseline.
- Class imbalance handling (IEEE-CIS: ~3.5% positive rate):
  - Primary metric: PR-AUC (Average Precision).
  - Threshold for P/R/F1: flag transactions with LOF scores above the `LOF_THRESHOLD_PERCENTILE`-th percentile (the top ~3.5%, matching the known fraud rate); the constant is stored in config.
  - No SMOTE or resampling.
- Baseline comparison: compares PR curves directly and reports PR-AUC as the primary scalar summary.
- **Output:** anomaly scores, metrics comparison, PR curves

### Structured Explainer

- Per flagged observation, produces a three-layer explanation:
  - Layer 1. Which features drove the LOF score (leave-one-feature-out neighborhood decomposition).
  - Layer 2. Why those features were encoded the way they were (from the MILP decision matrix + rubric).
  - Layer 3. Which segment they belong to, and the anomaly implication.
- Compute scope (from `config/defaults.py`): top `LOFO_MAX_ANOMALIES` (default 100) anomalies by LOF score; top `LOFO_MAX_FEATURES` (default 50) features by MI; LOF re-computation reuses the 10% stratified sample from the encoding evaluator (not the full ~590K rows); `LOFO_MODE = 'feature'` or `'segment'`.
- **Output:** explanation objects (JSON + human-readable text)

### Representation Geometry Analysis

- Trains two identical small MLPs (same architecture, seed, hyperparameters): `MLP-MILP` on MILP-selected encoded features, `MLP-OHE` on OHE-everywhere baseline features.
- **Invariant:** the MLP is not the anomaly detector (LOF remains primary); It's a controlled instrument for studying how encoding choice affects learned representation geometry.
- Measures superposition via three metrics on frozen first-hidden-layer activations:
  1. Interference matrix `W^T·W`; reports the Frobenius norm of the off-diagonal mass.
  2. Per-segment linear probe accuracy: a linear classifier trained on frozen activations to predict each of the 5 segment labels.
  3. Per-segment capacity: for each neuron, the fraction of activation variance attributable to each segment. (Practical operationalization inspired by Scherlis et al. (2022), whose formal definition uses fractional embedding dimension rather than activation variance.)
- **Testable hypothesis:** `MLP-MILP` exhibits lower off-diagonal interference, higher probe accuracy, and higher per-segment capacity than `MLP-OHE`.
- **Output:** `geometry_metrics.json`, `interference_matrix_{milp,ohe}`

### Demo: Streamlit App

- Upload a dataset or use the bundled sample.
- Tune α/β/γ weights and the dimensionality budget interactively; encoding decisions update from the pre-computed grid (not live).
- Inspect per-anomaly explanation cards.
- Baseline comparison panel (OHE-everywhere vs. MILP-selected).
- Representation geometry panel: interference matrix heatmaps side-by-side (`MLP-MILP` vs `MLP-OHE`) + probe accuracy table.
- Deployed to Streamlit Community Cloud (free).

---

## Interpretability Scoring Rubric

This rubric drives the γ term in the MILP objective. It's a parameter of the system, not hardcoded truth. One of the project's core empirical contributions is showing how changing γ shifts the encoding decisions along the performance and interpretability tradeoff curve.

| Encoding | Score | Rationale |
|---|---|---|
| Passthrough (continuous) | 0.90 | Direct numeric value; domain-meaningful as-is |
| Binning | 0.85 | Explicit threshold ranges; easily described in prose |
| Binary flag | 0.85 | Single bit; trivially interpretable |
| One-hot (low cardinality ≤ 10) | 0.80 | Each column is a named, explicit presence indicator |
| Ordinal (ordering confirmed) | 0.70 | Valid when ordering assumption holds; only applied when feature is listed in `config/ordinal_features.yaml` with an explicit value sequence |
| Ordinal (ordering unconfirmed) | 0.30 | Applied to features not listed in ordinal_features.yaml; penalized to reflect the uncertain ordering assumption as a liability |
| Frequency encoding | 0.50 | Indirect (rarity signal) but somewhat interpretable |
| One-hot (high cardinality > 10) | 0.40 | Correct but produces a wide sparse space; hard to reason about holistically |
| Target encoding | 0.35 | Compact but meaning is indirect (conditional probability) |

**Source and justification:** The ordering is a project-defined design judgment, informed by the qualitative observation in Potdar et al. (2017) that categorical encodings differ in structure and human-readability. Potdar is *not* cited as the source of this specific ranking — its own benchmark favors Sum/Backward-Difference contrast coding on one small dataset and makes no interpretability claim — so the ordering and the specific numeric scores are project-defined parameters, not universal constants. The gap between adjacent values encodes a design judgment about relative interpretability cost. One of the analysis-and-polish deliverables is a **rubric sensitivity analysis**: each score is perturbed ±10% and the MILP is re-run on the full evaluation matrix, then the fraction of encoding decisions that change is recorded. Stable decisions validate that specific values are not load-bearing. Unstable decisions identify which rubric entries are material research parameters rather than neutral assumptions.

---

## Build Plan

The canonical record of every resolved decision, with its rationale, is the *Open Design Decisions* table near the end of this document. The per-phase "Binding decisions" below summarize what is resolved in that phase; on any discrepancy, the table governs.

### Foundation
Set up the repo, data pipeline, and EDA. Land the feature profiler.

**Binding decisions (resolved):**
- **Reproducibility constants** live in `config/defaults.py`: `RANDOM_SEED = 42`, `LOF_THRESHOLD_PERCENTILE = 96.5`, `LOFO_MAX_ANOMALIES = 100`, `LOFO_MAX_FEATURES = 50`, `LOFO_MODE = 'feature'`. Every module accepts a `random_state` kwarg defaulting to `RANDOM_SEED`. Applies to k-means (the feature segmenter), stratified splits and LOF subsampling (the encoding evaluator), and MLP training (the representation analysis).
- **Ordinal annotation policy:** `config/ordinal_features.yaml` lists each feature with a meaningful value ordering alongside its explicit value sequence. Annotated during EDA while feature semantics are freshest. Listed defaults to 0.70 rubric score; unlisted defaults to 0.30 (see *Interpretability Scoring Rubric* and the *Open Design Decisions* table).

**Criterion:** clear knowledge of which IEEE-CIS features fall into each statistical category by the end of this phase.

**Deliverables:** `notebooks/01_eda.ipynb`, `src/feature_profiler.py`, `README.md` (initial reproduction guide; finalized in the analysis-and-polish phase), `config/defaults.py`, `config/ordinal_features.yaml`

---

###  Feature Segmentation
Land the feature segmenter.

**Binding decision (resolved):** Use the 5 domain-labeled segments (*"transaction amount"*, *"identity/device"*, *"behavioral frequency"*, *"temporal/timing"*, *"card/account"*) instead of purely statistical segments. Domain labels are more interpretable, make for a better demo, and the fixed 5-segment count anchors the representation analysis's per-segment probe-accuracy and capacity metrics. The segmentation approach is hybrid (rule-based on cardinality and dtype, plus k-means on the feature-profile vector); only the output vocabulary is fixed.

**Criterion:** segments make intuitive sense given dataset knowledge.

**Deliverables:** `src/feature_segmenter.py`, `tests/test_segmenter.py`, `outputs/segment_assignments.json`

---

###  Encoding Candidates + Evaluation
Land the encoding evaluator. Most compute-intensive phase.

**Binding decisions (resolved):**
- LOF evaluation runs on a **10% stratified sample**, not the full ~590K rows; results cached to disk.
- Evaluation is **scoped to top-N features by variance / MI**; the remainder are treated as passthrough (see *LOF evaluation scope* in *Open Design Decisions*).
- All interpretability and loss scores live in [0, 1]; dimensionality is a positive integer per feature × encoding pair.

**Deliverables:** `src/encoding_evaluator.py`, `tests/test_encoding_evaluator.py`, `outputs/evaluation_matrix.csv`

---

###  MILP Solver
Land the MILP selector using PuLP + HiGHS.

**Binding decisions (resolved):**
- **Solver backend:** PuLP + HiGHS (open source, no license issues).
- **`D_max` normalization anchor:** the maximum possible post-encoding dimensionality, computed analytically from `evaluation_matrix.csv` *before* the solver runs. Not a constraint; a normalization constant that puts `dim/D_max` in [0, 1]. See [docs/rationale.md](rationale.md) for the why.
- **Objective form:** `α·loss + β·(dim/D_max) + γ·(1−xplain)`, all three terms in [0, 1] so weights are comparable.
- **Infeasibility behavior:** surface a clear error (e.g. budget too small for any feasible assignment); never return a degenerate result silently.

**Demo architecture gate (resolved as gated upgrade):** Once the solver works, run the LP relaxation alongside the full MILP across the ~60 (α, β, γ, budget) grid points. Three criteria for the live-LP upgrade:
1. ≥95% of grid points produce LP-rounded decision matrices identical to the MILP solution.
2. No post-rounding budget violations on any grid point.
3. Every solve in under 500ms.

If all three pass, the Streamlit demo uses live LP relaxation (continuous slider response). Otherwise, sliders snap to the precomputed grid. **The precomputed grid is the default; live LP is the upgrade path, not a requirement.**

**Deliverables:** `src/milp_selector.py`, `tests/test_milp_selector.py`, `outputs/sample/encoding_decisions/`

---

###  Anomaly Detection + Baseline Comparison
Land the anomaly detector.

**Binding decisions (resolved):**
- **Headline metric:** PR-AUC. Threshold-free, and correct for ranking under heavy class imbalance.
- **Threshold for P/R/F1:** prevalence-percentile. Flag the top `100 − LOF_THRESHOLD_PERCENTILE` (= 3.5%) of transactions by LOF score, matching the known IEEE-CIS positive rate.
- **No SMOTE or resampling.** LOF is an unsupervised density estimator; class imbalance does not bias it the same way it biases supervised learners.
- **Baseline:** uniform OHE on all categoricals + passthrough on continuous, run through the same LOF pipeline. PR curves are compared directly.

**Deliverables:** `src/anomaly_detector.py`, `tests/test_anomaly_detector.py`, `notebooks/02_results.ipynb`, `outputs/metrics/`

---

###  Explanation Layer
Land the explainer. The key design challenge: scikit-learn's LOF gives scores but not per-feature contributions.

**Binding decisions (resolved):**
- **Per-feature attribution method:** leave-one-feature-out around LOF. No surrogate model, no SHAP. Each feature's contribution is the change in LOF score when It's masked.
- **Compute scope** (from `config/defaults.py`): top `LOFO_MAX_ANOMALIES` anomalies by LOF score, scored over the top `LOFO_MAX_FEATURES` by MI, re-using the encoding-evaluator 10% stratified sample (not the full ~590K rows). `LOFO_MODE` selects per-feature or per-segment granularity.

**Criterion:** explanations make domain sense on a hand-validated sample drawn from the top 100.

**Deliverables:** `src/explainer.py`, `tests/test_explainer.py`, `outputs/explanations/sample_explanations.json`

---

###  Representation Geometry Analysis + Streamlit Demo
Land the representation analysis and the demo.

**Binding decisions (resolved), representation analysis:**
- **Controlled comparison:** `MLP-MILP` and `MLP-OHE` are identical (architecture, `RANDOM_SEED`, hyperparameters); only the input encoding differs. Architecture: 2 hidden layers, ReLU, binary fraud classification head.
- **The MLP is not the anomaly detector.** LOF remains the primary anomaly detector. The MLP is a controlled instrument for studying how encoding choice affects representation geometry. State this clearly in the notebook and write-up.
- **Three metrics** (per the *System Architecture*  *Representation Geometry Analysis* section) computed on frozen first-hidden-layer activations: interference matrix `W^T·W` (off-diagonal Frobenius norm), per-segment linear probe accuracy across the 5 segments, and per-segment capacity as fraction-of-activation-variance per neuron. The capacity metric is a practical operationalization inspired by Scherlis et al. (2022), whose formal definition uses fractional embedding dimension rather than activation variance. Document this caveat in `representation_analysis.py`.
- **Forward-looking note for the analysis:** sparse autoencoders (SAEs) are the current state-of-the-art for extracting interpretable features from superposed representations and are the natural next step beyond this work.

**Binding decisions (resolved), Streamlit demo:**
- Encoding decisions load from the precomputed grid under `outputs/sample/encoding_decisions/` by default; live LP relaxation is gated on the MILP-relaxation benchmark (see the MILP phase).
- Panels: encoding decision table, per-anomaly explanation cards, baseline comparison, representation geometry (interference heatmaps side-by-side + probe accuracy table), and α/β/γ + dimensionality-budget controls.
- Deployed to Streamlit Community Cloud (free).

**Deliverables:** `src/representation_analysis.py`, `tests/test_representation_analysis.py`, `notebooks/03_representation_geometry.ipynb`, `outputs/geometry/`, `demo/app.py`, live Streamlit URL.

---

###  Analysis Document + Write-Up
The write-up that turns the code into a communicable result, not just an implementation.

**Questions `docs/analysis.md` must answer:**
- *LOF detection:* where did MILP-selected encodings outperform the uniform-OHE baseline, and where did they not? Which feature types benefit most from non-OHE encodings? What does the performance and interpretability tradeoff curve look like empirically?
- *Representation geometry:* did `MLP-MILP` exhibit lower off-diagonal interference than `MLP-OHE`? Which segments became more linearly probe-able? What does the per-segment capacity distribution look like, and are there more monosemantic neurons under MILP-selected encodings?
- *Rubric sensitivity:* after perturbing each interpretability rubric score ±10% and re-running the MILP on the full evaluation matrix, what fraction of encoding decisions flip? Which rubric rows are load-bearing rather than neutral?

**Binding decisions (resolved):**
- **Honest null-result reporting.** If `MLP-MILP` and `MLP-OHE` exhibit equivalent superposition, that finding is reported and interpreted as evidence of network compensation, not omitted.
- **Citations.** Findings are connected explicitly to Elhage et al. (2022) *Toy Models of Superposition* and Scherlis et al. (2022) *Polysemanticity and Capacity in Neural Networks* (see *Reference Material*).
- **Reproducibility floor:** the final `README.md` lets a new contributor reproduce all pipeline outputs in under 30 minutes by setting `RANDOM_SEED` in `config/defaults.py` and running the pipeline.

**Deliverables:** `docs/analysis.md`, final `README.md`, rubric sensitivity table, optional blog post draft.

---

## Repo Structure

```
adaptive-encoding-anomaly-detector/
├── README.md
├── pyproject.toml
├── config/
│   ├── defaults.py               # RANDOM_SEED, LOF params, budget levels, weight grid, LOFO scope
│   └── ordinal_features.yaml    # IEEE-CIS features with documented ordinal value sequences
├── data/                          # raw data
├── src/
│   ├── feature_profiler.py
│   ├── feature_segmenter.py
│   ├── encoding_evaluator.py
│   ├── milp_selector.py
│   ├── anomaly_detector.py
│   ├── explainer.py
│   └── representation_analysis.py # MLP training + geometry metrics
├── tests/
│   ├── test_profiler.py
│   ├── test_segmenter.py
│   ├── test_encoding_evaluator.py  #  Scores in [0,1], dimensionality, matrix shape
│   ├── test_milp_selector.py
│   ├── test_anomaly_detector.py    #  Smoke test, thresholds
│   ├── test_explainer.py           #  Three-layer structure
│   └── test_representation_analysis.py  #  W^T·W PSD, probe accuracy, capacity
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_results.ipynb
│   └── 03_representation_geometry.ipynb
├── outputs/                       # gitignored
│   ├── sample/                    # demo artifacts
│   │   └── encoding_decisions/    # MILP solutions for demo
│   ├── segment_assignments.json
│   ├── evaluation_matrix.csv
│   ├── metrics/
│   ├── explanations/
│   └── geometry/                  # interference matrices, probe results, capacity
├── demo/
│   └── app.py
└── docs/
    ├── project-plan.md            # canonical spec (this document)
    ├── rationale.md               # Design notes
    ├── implementation-plan.md     # Planning + Decision log
    └── analysis.md                # Write-up
```


## Open Design Decisions

These should be resolved before or during the relevant phase. Recording them here for reference as the implementation progresses.

| Decision | Options | Recommended | Resolve By |
|---|---|---|---|
| MILP solver backend | PuLP+HiGHS, PuLP+CBC, scipy.optimize.milp | PuLP+HiGHS (open source, fast, no license issues), pinned explicitly via the in-process `HiGHS` class (`highspy`) rather than relying on PuLP's default solver. Note on the PuLP+CBC alternative: the stable PuLP release (3.3.x) still bundles CBC and still ships `PULP_CBC_CMD`, so `pip install pulp` works solver-ready out of the box; the live docs at coin-or.github.io/pulp document the unreleased 4.0 alpha, which unbundles CBC (`pip install pulp[cbc]`) and removes `PULP_CBC_CMD` — a doc-ahead-of-release drift, not the behavior of the version a contributor installs today. Immaterial to this project, which pins HiGHS regardless. 
| MILP objective normalization | Raw dimensionality, divide by D_max, full ideal-point solve | Normalize dimensionality as `dim/D_max`, where `D_max` is the maximum possible post-encoding dimensionality, computed analytically from evaluation_matrix before the MILP runs. Loss and `(1−xplain)` are already in [0,1]. All three terms are then comparable and weights are meaningful. 
| Anomaly signal proxy (feature-profiler MI) | Fraud label (supervised MI), Isolation Forest score (unsupervised), raw variance | Binary fraud label via `_mutual_info_classifier`, used for feature characterization only, not to train the anomaly detector (LOF) or the geometry MLP. Document the distinction explicitly in `feature_profiler.py`. 
| Secondary dataset / generalizability demonstration | PaySim second dataset, structured baseline comparisons, representation geometry analysis | Replace PaySim with a representation geometry analysis. Train MLP-MILP and MLP-OHE and measure superposition via interference matrix, linear probes, and capacity metrics. Demonstrates depth over breadth and directly engages the mechanistic interpretability research agenda. 
| LOF evaluation scope | All 400 features, top-N by variance/MI, full with aggressive subsampling | Top-N with 10% stratified sample + disk caching 
| Feature segment labeling | Purely statistical (k-means on profile vector) vs. domain-labeled | Domain-labeled, because It's more interpretable and demos better. 
| Demo interactivity architecture | Live MILP (infeasible), live LP relaxation with rounding, pre-computed grid | **DEFAULT:** Pre-computed grid of ~60 MILP solutions keyed by (α, β, γ, budget). **UPGRADE PATH:** After the MILP solver is built, run LP relaxation benchmark gate. If LP decisions match MILP on ≥95% of combinations, LP never violates budget constraint after rounding, and LP solves in <500ms, switch demo to live LP relaxation. Document whichever approach is used and why. 
| Reproducibility / random seeds | Global seed in config, per-module seeds, no explicit seeding | `RANDOM_SEED = 42` in `config/defaults.py`. All modules accept a `random_state` parameter defaulting to this constant. Applies to k-means (the feature segmenter), stratified splits and LOF subsampling (the encoding evaluator), and MLP training (the representation analysis). 
| Class imbalance handling (anomaly detector) | SMOTE, threshold tuning, prevalence-percentile threshold, PR-AUC only | PR-AUC is the primary metric (threshold-free, correct for imbalanced ranking). P/R/F1 reported at the prevalence-percentile threshold (top ~3.5% by LOF score, stored as `LOF_THRESHOLD_PERCENTILE` in config). No SMOTE, because LOF is unsupervised. Baseline comparison uses PR curves directly. 
| LOF leave-one-out compute scope (explainer) | All flagged anomalies × all features, capped N × top-K features, segment-level grouping | Hard cap `LOFO_MAX_ANOMALIES = 100` (top 100 by LOF score). Feature scope `LOFO_MAX_FEATURES = 50` (top by MI). LOF re-computation reuses the 10% stratified sample from the encoding evaluator. `LOFO_MODE = 'feature'` or `'segment'` (configurable). All params in `config/defaults.py`. 
| Interpretability rubric source and stability | Cite Potdar et al. for ordering, add sensitivity analysis, leave as-is | Ordering is a project-defined design judgment, informed by (not derived from) Potdar et al. (2017); Potdar's own benchmark favors contrast coding and makes no interpretability ranking, so the ordering and specific values are project-defined parameters. Perturb each score ±10% and measure the fraction of encoding decisions that change. Ordinal row split: 0.70 (ordering confirmed, listed in ordinal_features.yaml) and 0.30 (ordering unconfirmed). 
| Ordinal feature ordering assumption | Manual annotation, inferred from feature name, inferred from distribution | Manual annotation during the foundation-phase EDA. `config/ordinal_features.yaml` lists features with documented value sequences. Features not listed receive 0.30 ordinal score (uncertain ordering treated as liability). Ordinal encoding not applied without documented ordering. 
| Test coverage scope | Tests for all modules, tests for novel modules only, integration tests only | Staggered per module: each `tests/test_<module>.py` is added when its module is built. The module-to-phase mapping lives in the *Build Plan* section above and in `docs/implementation-plan.md`'s remaining-work list; don't restate it here, to avoid drift. | Staggered |

---

## Interpretability

- **Superposition.** The representation-geometry analysis asks whether upstream encoding decisions affect the geometry of learned representations, measured via the interference matrix, linear probe accuracy, and capacity metrics developed in Elhage et al. (2022) and Scherlis et al. (2022).

- **Architectural interpretability over post-hoc explanation.** Internalized decisions on encoding selection are transparent and auditable by design, rather than attempting post-hoc explanation of outputs after the fact. The MILP decision matrix is treated as an explanation layer.

- **Null-result Reporting.** The representation geometry hypothesis may not hold. If MLP-MILP and MLP-OHE exhibit equivalent superposition, that's interesting too! It may suggest that the network learns to compensate for poor input encoding quality. The analysis will report this honestly either way. A null result is a legitimate finding that invites further investigation.

The analysis write-up will reference representation-geometry findings directly. The Streamlit demo's interference matrix heatmap panel will provide a visual inspired by the presentation in Toy Models of Superposition.

---

## Reference Material

- **Paper:** Li, Z. & Fan, R. (2026). *Explainable Heterogeneous Anomaly Detection in Financial Networks via Adaptive Expert Routing.* arXiv:2510.17088v2. https://arxiv.org/abs/2510.17088
- **Dataset:** IEEE-CIS Fraud Detection. https://www.kaggle.com/competitions/ieee-fraud-detection
- **Superposition:** Elhage, N. et al. (2022). *Toy Models of Superposition.* Transformer Circuits Thread. https://transformer-circuits.pub/2022/toy_model/index.html
- **Capacity:** Scherlis, A. et al. (2022). *Polysemanticity and Capacity in Neural Networks.* arXiv:2210.01892. https://arxiv.org/abs/2210.01892
- **Encoding reference:** Potdar, K., Pardawala, T. S., & Pai, C. D. (2017). A Comparative Study of Categorical Variable Encoding Techniques for Neural Network Classifiers. *International Journal of Computer Applications*, 175(4), 7–9. DOI 10.5120/ijca2017915495. https://www.ijcaonline.org/archives/volume175/number4/28474-2017915495/ — Cited only for the *qualitative* observation that encodings differ in interpretability/structure, not for an encoding *performance* ranking: the paper's own benchmark (a single small UCI Car Evaluation set) favors Sum and Backward-Difference contrast coding, so it does not establish OHE/ordinal as a baseline and its result is not assumed to transfer to high-cardinality fraud features.
- **LOF:** Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers. ACM SIGMOD.
- **PuLP docs:** https://coin-or.github.io/pulp/
- **HiGHS solver:** https://highs.dev/
