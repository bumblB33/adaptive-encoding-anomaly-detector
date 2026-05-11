# Portfolio Project Plan
## Adaptive Encoding Anomaly Detector

**Author:** Brittany Walker  
**Created:** April 2026  
**Target Application Context:** Anthropic Research Scientist, Interpretability  
**Estimated Timeline:** 6 to 8 weeks

---

## Motivation

This project synthesizes two research threads.

1. **Explainable Heterogeneous Anomaly Detection** (Li & Fan, 2026, arXiv:2510.17088v2): Financial anomalies arise from heterogeneous mechanisms (price shocks, liquidity freezes, contagion cascades, momentum reversals). The paper addresses this with a mixture-of-experts architecture in which routing weights serve as interpretable proxies for mechanism attribution. The authors call this "architectural interpretability" to distinguish it from post-hoc explanation.

2. **Feature Encoding Tradeoffs in ML:** Feature encoding choice (one-hot, ordinal, target, binary, binning, passthrough) significantly affects model performance and coefficient interpretability, particularly in linear and neural architectures. In practice, most ML pipelines apply a single encoding scheme to all categorical features based on dtype rather than evaluating each feature's cardinality, ordinality, or relationship to the target. That convenience leaves meaningful performance and interpretability gains on the table.

**Hypothesis:** If different anomaly mechanisms have different feature signatures, and encoding choice materially affects how well those signatures are preserved, then encoding selection should be a first-class, optimized, and auditable part of the anomaly detection pipeline, not a preprocessing assumption. The encoding decision matrix is itself an explanation layer.

**Project Proposal:** Encoding selection is framed as a Mixed-Integer Linear Program, turning a heuristic preprocessing step into a principled, constraint-aware optimization with an auditable solution. This draws on my prior implementation experience with MILP-based scheduling optimization.

---

## Dataset

**Primary: IEEE-CIS Fraud Detection** (Kaggle, free with account)

- 400+ raw features: email domains, device types, card networks, product codes, transaction type codes.
- A mix of high-cardinality nominal, low-cardinality ordinal, binary flag, and continuous features. This is exactly the variety where encoding choice matters.
- Ground-truth fraud labels enable concrete precision/recall evaluation, not just unsupervised scoring.
- A well-documented, widely used benchmark with strong community reference material.

**Why not UCI credit card fraud:** Its 30 features are already PCA-transformed into anonymous continuous variables. There is nothing to encode differently, which makes it a dead end for encoding research.

**No secondary dataset.** Rather than demonstrating generalizability via a second dataset, the project demonstrates depth via a representation geometry analysis (Module 7). A small MLP is trained on both encoding regimes, and its hidden representations are analyzed for superposition. That choice connects the project directly to Anthropic's mechanistic interpretability research agenda. See Module 7 and Week 7 for details.

**Data policy:** Raw data is never committed to the repo. The top-level `README.md` contains download instructions.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Raw tabular financial dataset (mixed feature types)      │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 1: Feature Profiler                                      │
│  - Detect data type (continuous, nominal, ordinal, datetime)     │
│  - Compute cardinality, distribution shape, missingness rate     │
│  - Compute mutual information with anomaly signal proxy:         │
│      · Binary fraud label via mutual_info_classif                │
│        (label used for feature characterization only;            │
│         LOF and MLP never see it as a training target)           │
│  Output: feature_profile.json                                    │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 2: Feature Segmenter                                     │
│  - Cluster features by profile into 5 domain-labeled segments    │
│  - Inspired by the paper's mechanism-aware segmentation          │
│  - Segments: "transaction amount", "identity/device",            │
│    "behavioral frequency", "temporal/timing", "card/account"     │
│  Output: segment_assignments.json                                │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 3: Encoding Candidate Evaluator                          │
│  - For each feature × encoding pair, compute:                    │
│      · Encoded dimensionality (known analytically)               │
│      · Detection loss on held-out fold (LOF, subsampled)         │
│      · Interpretability score (rubric-based; see below)          │
│  - Results cached to disk (expensive to recompute)               │
│  Output: evaluation_matrix.csv                                   │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 4: MILP Encoding Selector                                │
│  - Decision variables: x[feature, encoding] ∈ {0, 1}            │
│  - Pre-computation: D_max = sum of max encoding dimensionality   │
│    per feature (computed analytically from evaluation_matrix)    │
│  - Objective: minimize α·loss + β·(dim/D_max) + γ·(1−xplain)   │
│    · All three terms normalized to [0,1]. Weights become        │
│      meaningful and comparable across objectives.                │
│    · loss ∈ [0,1] naturally; dim/D_max ∈ [0,1] by construction │
│    · (1−xplain) ∈ [0,1] by rubric definition                   │
│  - Constraints:                                                  │
│      · Exactly one encoding per feature                          │
│      · Total post-encoding dimensionality ≤ budget               │
│      · Minimum explainability floor per segment                  │
│  - Solver: PuLP with HiGHS backend (open source, no license)    │
│  Output: encoding_decisions.json (auditable decision matrix)     │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 5: Anomaly Detector                                      │
│  - Apply selected encodings to full dataset                      │
│  - Run LOF on the MILP-selected encoded feature space            │
│  - Run identical pipeline with uniform OHE as baseline           │
│  - Class imbalance handling (IEEE-CIS: ~3.5% positive rate):    │
│      · Primary metric: Average Precision (PR-AUC). It is        │
│        threshold-free and correct for ranking tasks with         │
│        heavy imbalance.                                          │
│      · Threshold for P/R/F1: prevalence-percentile. Flag the    │
│        top LOF_THRESHOLD_PERCENTILE of transactions by LOF      │
│        score, matching the known ~3.5% fraud rate; stored in    │
│        config.                                                   │
│      · No SMOTE or resampling. LOF is an unsupervised density   │
│        estimator; class imbalance does not bias it the same way │
│        it biases supervised learners.                            │
│  - Baseline comparison: compare PR curves directly; report      │
│    PR-AUC as primary scalar summary                              │
│  Output: anomaly scores, metrics comparison, PR curves           │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 6: Structured Explainer                                  │
│  - Per flagged observation, produces a three-layer explanation:  │
│    Layer 1: Which features drove the LOF score                   │
│             (leave-one-feature-out neighborhood decomposition)   │
│    Layer 2: Why those features were encoded the way they were    │
│             (from the MILP decision matrix + rubric)             │
│    Layer 3: Which segment they belong to and anomaly implication │
│  - Compute scope (configured in config/defaults.py):            │
│      · Explain top LOFO_MAX_ANOMALIES (default: 100) by LOF     │
│        score, not all ~17K flagged cases                         │
│      · Feature scope: top LOFO_MAX_FEATURES (default: 50) by MI │
│        with fraud label, not all 400 features                    │
│      · LOF re-computation reuses the 10% stratified sample from │
│        Module 3 (not the full 590K-row dataset)                  │
│      · LOFO_MODE = 'feature' (per-feature) or 'segment'         │
│        (per-segment group, ~5x faster); configurable             │
│  Output: explanation objects (JSON + human-readable text)        │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 7: Representation Geometry Analysis                      │
│  Train two identical small MLPs (same arch, seed, hyperparams): │
│    · MLP-MILP: trained on MILP-selected encoded features         │
│    · MLP-OHE:  trained on OHE-everywhere baseline features       │
│  The MLP is NOT the anomaly detector. LOF remains primary.       │
│  The MLP is a controlled vehicle for studying how encoding       │
│  choice affects learned representation geometry.                 │
│  Measure superposition via three metrics on frozen activations:  │
│    1. Interference matrix: W^T·W for first hidden layer.         │
│       Diagonal dominance = orthogonal features = low             │
│       superposition. Report Frobenius norm of off-diagonal mass. │
│    2. Per-segment linear probe accuracy: train a linear          │
│       classifier on frozen activations to predict each of the    │
│       5 feature segment labels. Higher accuracy = signal from    │
│       that segment is more linearly accessible in the hidden     │
│       layer = less buried in superposition.                      │
│    3. Per-segment capacity: for each neuron, compute fraction    │
│       of activation variance attributable to each segment.       │
│       Monosemantic neurons score near 1 for one segment.         │
│       (Practical operationalization inspired by Scherlis et al. │
│        (2022); their formal definition uses fractional embedding │
│        dimension rather than activation variance.)               │
│  Testable hypothesis: MLP-MILP exhibits lower off-diagonal       │
│  interference, higher probe accuracy, and higher per-segment     │
│  capacity than MLP-OHE. In other words, principled encoding      │
│  reduces superposition in learned representations.               │
│  Output: geometry_metrics.json, interference_matrix_{milp,ohe}  │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  DEMO: Streamlit App                                             │
│  - Upload dataset or use bundled sample                          │
│  - Tune α/β/γ weights and dimensionality budget interactively    │
│  - Encoding decisions update from pre-computed grid (not live)   │
│  - Inspect per-anomaly explanation cards                         │
│  - Baseline comparison panel (OHE-everywhere vs. MILP-selected) │
│  - Representation geometry panel: interference matrix heatmaps  │
│    side-by-side (MLP-MILP vs MLP-OHE) + probe accuracy table    │
│  - Deployed: Streamlit Community Cloud (free)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Interpretability Scoring Rubric

This rubric drives the γ term in the MILP objective. It is a parameter of the system, not hardcoded truth. One of the project's core empirical contributions is showing how changing γ shifts the encoding decisions along the performance and interpretability tradeoff curve.

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

**Source and justification:** The qualitative ordering of encodings by interpretability follows comparative characterizations in Potdar et al. (2017). The specific numeric scores are project-defined parameters, not universal constants. The gap between adjacent values encodes a design judgment about relative interpretability cost. One of the Week 8 deliverables is a **rubric sensitivity analysis**: each score is perturbed ±10% and the MILP is re-run on the full evaluation matrix, then the fraction of encoding decisions that change is recorded. Stable decisions validate that specific values are not load-bearing. Unstable decisions identify which rubric entries are material research parameters rather than neutral assumptions.

---

## Weekly Plan

### Week 1: Foundation
Set up the repo, data pipeline, and EDA. Land Module 1.

**Binding decisions (resolved):**
- **Reproducibility constants** live in `config/defaults.py`: `RANDOM_SEED = 42`, `LOF_THRESHOLD_PERCENTILE = 96.5`, `LOFO_MAX_ANOMALIES = 100`, `LOFO_MAX_FEATURES = 50`, `LOFO_MODE = 'feature'`. Every module accepts a `random_state` kwarg defaulting to `RANDOM_SEED`. Applies to k-means (Module 2), stratified splits and LOF subsampling (Module 3), and MLP training (Module 7).
- **Ordinal annotation policy:** `config/ordinal_features.yaml` lists each feature with a meaningful value ordering alongside its explicit value sequence. Annotated during EDA while feature semantics are freshest. Listed → 0.70 rubric score; unlisted → 0.30 (see *Interpretability Scoring Rubric* and the *Open Design Decisions* table).

**Criterion:** clear knowledge of which IEEE-CIS features fall into each statistical category by end of week.

**Deliverables:** `notebooks/01_eda.ipynb`, `src/feature_profiler.py`, `README.md` (initial reproduction guide; finalized in Week 8), `config/defaults.py`, `config/ordinal_features.yaml`

---

### Week 2: Feature Segmentation
Land Module 2.

**Binding decision (resolved):** Use the 5 domain-labeled segments (*"transaction amount"*, *"identity/device"*, *"behavioral frequency"*, *"temporal/timing"*, *"card/account"*) instead of purely statistical segments. Domain labels are more interpretable, make for a better demo, and the fixed 5-segment count anchors Module 7's per-segment probe-accuracy and capacity metrics. The segmentation approach is hybrid (rule-based on cardinality and dtype, plus k-means on the feature-profile vector); only the output vocabulary is fixed.

**Criterion:** segments make intuitive sense given dataset knowledge.

**Deliverables:** `src/feature_segmenter.py`, `tests/test_segmenter.py`, `outputs/segment_assignments.json`

---

### Week 3: Encoding Candidates + Evaluation
Land Module 3. Most compute-intensive week.

**Binding decisions (resolved):**
- LOF evaluation runs on a **10% stratified sample**, not the full ~590K rows; results cached to disk.
- Evaluation is **scoped to top-N features by variance / MI**; the remainder are treated as passthrough (see *LOF evaluation scope* in *Open Design Decisions*).
- All interpretability and loss scores live in [0, 1]; dimensionality is a positive integer per feature × encoding pair.

**Deliverables:** `src/encoding_evaluator.py`, `tests/test_encoding_evaluator.py`, `outputs/evaluation_matrix.csv`

---

### Week 4: MILP Solver
Land Module 4 using PuLP + HiGHS.

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

### Week 5: Anomaly Detection + Baseline Comparison
Land Module 5.

**Binding decisions (resolved):**
- **Headline metric:** PR-AUC. Threshold-free, and correct for ranking under heavy class imbalance.
- **Threshold for P/R/F1:** prevalence-percentile. Flag the top `100 − LOF_THRESHOLD_PERCENTILE` (= 3.5%) of transactions by LOF score, matching the known IEEE-CIS positive rate.
- **No SMOTE or resampling.** LOF is an unsupervised density estimator; class imbalance does not bias it the same way it biases supervised learners.
- **Baseline:** uniform OHE on all categoricals + passthrough on continuous, run through the same LOF pipeline. PR curves are compared directly.

**Deliverables:** `src/anomaly_detector.py`, `tests/test_anomaly_detector.py`, `notebooks/02_results.ipynb`, `outputs/metrics/`

---

### Week 6: Explanation Layer
Land Module 6. The key design challenge: scikit-learn's LOF gives scores but not per-feature contributions.

**Binding decisions (resolved):**
- **Per-feature attribution method:** leave-one-feature-out around LOF. No surrogate model, no SHAP. Each feature's contribution is the change in LOF score when it is masked.
- **Compute scope** (from `config/defaults.py`): top `LOFO_MAX_ANOMALIES` anomalies by LOF score, scored over the top `LOFO_MAX_FEATURES` by MI, re-using the Module-3 10% stratified sample (not the full ~590K rows). `LOFO_MODE` selects per-feature or per-segment granularity.

**Criterion:** explanations make domain sense on a hand-validated sample drawn from the top 100.

**Deliverables:** `src/explainer.py`, `tests/test_explainer.py`, `outputs/explanations/sample_explanations.json`

---

### Week 7: Representation Geometry Analysis + Streamlit Demo
Land Module 7 and the demo.

**Binding decisions (resolved), Module 7:**
- **Controlled comparison:** `MLP-MILP` and `MLP-OHE` are identical (architecture, `RANDOM_SEED`, hyperparameters); only the input encoding differs. Architecture: 2 hidden layers, ReLU, binary fraud classification head.
- **The MLP is not the anomaly detector.** LOF remains the primary anomaly detector. The MLP is a controlled instrument for studying how encoding choice affects representation geometry. State this clearly in the notebook and write-up.
- **Three metrics** (per the *System Architecture* Module 7 box) computed on frozen first-hidden-layer activations: interference matrix `W^T·W` (off-diagonal Frobenius norm), per-segment linear probe accuracy across the 5 segments, and per-segment capacity as fraction-of-activation-variance per neuron. The capacity metric is a practical operationalization inspired by Scherlis et al. (2022), whose formal definition uses fractional embedding dimension rather than activation variance. Document this caveat in `representation_analysis.py`.
- **Forward-looking note for the analysis:** sparse autoencoders (SAEs) are the current state-of-the-art for extracting interpretable features from superposed representations and are the natural next step beyond this work.

**Binding decisions (resolved), Streamlit demo:**
- Encoding decisions load from the precomputed grid under `outputs/sample/encoding_decisions/` by default; live LP relaxation is gated on the Week-4 benchmark (see Week 4).
- Panels: encoding decision table, per-anomaly explanation cards, baseline comparison, representation geometry (interference heatmaps side-by-side + probe accuracy table), and α/β/γ + dimensionality-budget controls.
- Deployed to Streamlit Community Cloud (free).

**Deliverables:** `src/representation_analysis.py`, `tests/test_representation_analysis.py`, `notebooks/03_representation_geometry.ipynb`, `outputs/geometry/`, `demo/app.py`, live Streamlit URL.

---

### Week 8: Analysis Document + Write-Up
The document that makes this portfolio-ready rather than just code.

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
├── data/                          # raw data lives here locally; nothing committed
├── src/
│   ├── feature_profiler.py
│   ├── feature_segmenter.py
│   ├── encoding_evaluator.py
│   ├── milp_selector.py
│   ├── anomaly_detector.py
│   ├── explainer.py
│   └── representation_analysis.py # Module 7: MLP training + geometry metrics
├── tests/
│   ├── test_profiler.py
│   ├── test_segmenter.py
│   ├── test_encoding_evaluator.py  # Week 3: scores in [0,1], dimensionality, matrix shape
│   ├── test_milp_selector.py
│   ├── test_anomaly_detector.py    # Week 5: smoke test, threshold behavior
│   ├── test_explainer.py           # Week 6: three-layer structure, edge cases
│   └── test_representation_analysis.py  # Week 7: W^T·W PSD, probe accuracy, capacity
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_results.ipynb
│   └── 03_representation_geometry.ipynb
├── outputs/                       # gitignored except the outputs/sample/ carve-out
│   ├── sample/                    # tracked: demo-required artifacts
│   │   └── encoding_decisions/    # ~60 MILP solutions; demo loads from here
│   ├── segment_assignments.json
│   ├── evaluation_matrix.csv
│   ├── metrics/
│   ├── explanations/
│   └── geometry/                  # interference matrices, probe results, capacity
├── demo/
│   └── app.py
└── docs/
    ├── portfolio_project_plan.md  # canonical spec (this document)
    ├── rationale.md               # design-rationale notes
    ├── implementation-plan.md     # living plan reconciled with code
    └── analysis.md                # Week-8 write-up
```

---

## Open Design Decisions

These should be resolved before or during the relevant week. Recording them here for reference as the implementation progresses.

| Decision | Options | Recommended | Resolve By |
|---|---|---|---|
| MILP solver backend | PuLP+HiGHS, PuLP+CBC, scipy.optimize.milp | PuLP+HiGHS (open source, fast, no license issues) | Week 4 |
| MILP objective normalization | Raw dimensionality, divide by D_max, full ideal-point solve | Normalize dimensionality as `dim/D_max`, where `D_max` is the maximum possible post-encoding dimensionality, computed analytically from evaluation_matrix before the MILP runs. Loss and `(1−xplain)` are already in [0,1]. All three terms are then comparable and weights are meaningful. | Pre-Week 4 |
| Anomaly signal proxy (Module 1 MI) | Fraud label (supervised MI), Isolation Forest score (unsupervised), raw variance | Binary fraud label via `mutual_info_classif`, used for feature characterization only, not to train the anomaly detector (LOF) or the geometry MLP. Document the distinction explicitly in `feature_profiler.py`. | Week 1 |
| Secondary dataset / generalizability demonstration | PaySim second dataset, structured baseline comparisons, representation geometry analysis | Replace PaySim with Module 7 representation geometry analysis. Train MLP-MILP and MLP-OHE and measure superposition via interference matrix, linear probes, and capacity metrics. Demonstrates depth over breadth and directly engages the mechanistic interpretability research agenda. | Pre-Week 7 |
| LOF evaluation scope | All 400 features, top-N by variance/MI, full with aggressive subsampling | Top-N with 10% stratified sample + disk caching | Week 3 |
| Feature segment labeling | Purely statistical (k-means on profile vector) vs. domain-labeled | Domain-labeled, because it is more interpretable and demos better. | Week 2 |
| Demo interactivity architecture | Live MILP (infeasible), live LP relaxation with rounding, pre-computed grid | **DEFAULT:** Pre-computed grid of ~60 MILP solutions keyed by (α, β, γ, budget). **UPGRADE PATH:** After Week 4 solver is built, run LP relaxation benchmark gate. If LP decisions match MILP on ≥95% of combinations, LP never violates budget constraint after rounding, and LP solves in <500ms, switch demo to live LP relaxation. Document whichever approach is used and why. | Week 4 gate → Week 7 |
| Reproducibility / random seeds | Global seed in config, per-module seeds, no explicit seeding | `RANDOM_SEED = 42` in `config/defaults.py`. All modules accept a `random_state` parameter defaulting to this constant. Applies to k-means (Module 2), stratified splits and LOF subsampling (Module 3), and MLP training (Module 7). | Week 1 |
| Class imbalance handling (Module 5) | SMOTE, threshold tuning, prevalence-percentile threshold, PR-AUC only | PR-AUC is the primary metric (threshold-free, correct for imbalanced ranking). P/R/F1 reported at the prevalence-percentile threshold (top ~3.5% by LOF score, stored as `LOF_THRESHOLD_PERCENTILE` in config). No SMOTE, because LOF is unsupervised. Baseline comparison uses PR curves directly. | Pre-Week 5 |
| LOF leave-one-out compute scope (Module 6) | All flagged anomalies × all features, capped N × top-K features, segment-level grouping | Hard cap `LOFO_MAX_ANOMALIES = 100` (top 100 by LOF score). Feature scope `LOFO_MAX_FEATURES = 50` (top by MI). LOF re-computation reuses the 10% stratified sample from Module 3. `LOFO_MODE = 'feature'` or `'segment'` (configurable). All params in `config/defaults.py`. | Pre-Week 6 |
| Interpretability rubric source and stability | Cite Potdar et al. for ordering, add sensitivity analysis, leave as-is | Ordering follows Potdar et al. (2017) qualitative characterizations. Specific values are project-defined parameters. Week 8 rubric sensitivity analysis: perturb each score ±10% and measure the fraction of encoding decisions that change. Ordinal row split: 0.70 (ordering confirmed, listed in ordinal_features.yaml) and 0.30 (ordering unconfirmed). | Pre-Week 4 |
| Ordinal feature ordering assumption | Manual annotation, inferred from feature name, inferred from distribution | Manual annotation during Week 1 EDA. `config/ordinal_features.yaml` lists features with documented value sequences. Features not listed receive 0.30 ordinal score (uncertain ordering treated as liability). Ordinal encoding not applied without documented ordering. | Week 1 |
| Test coverage scope | Tests for all modules, tests for novel modules only, integration tests only | Staggered per module: each `tests/test_<module>.py` is added when its module is built. The module-to-week mapping lives in the *Weekly Plan* section above and in `docs/implementation-plan.md`'s remaining-work list; don't restate it here, to avoid drift. | Staggered |

---

## Connection to Anthropic Application

This project is targeted at the Research Scientist, Interpretability role. The connection to Anthropic's mechanistic interpretability agenda is as follows.

- **Direct engagement with superposition.** Module 7 asks whether upstream encoding decisions affect the geometry of learned representations, measured via the interference matrix, linear probe accuracy, and capacity metrics developed in Elhage et al. (2022) and Scherlis et al. (2023). This is not an analogy to mechanistic interpretability; it is an empirical pilot study in it.

- **Architectural interpretability over post-hoc explanation.** The project's core framing mirrors Anthropic's philosophy: make the system's internal decisions (encoding selection) transparent and auditable by design, rather than explaining a black-box output after the fact. The MILP decision matrix is itself an explanation layer.

- **Falsifiable hypotheses and honest null-result reporting.** The representation geometry hypothesis may not hold. If MLP-MILP and MLP-OHE exhibit equivalent superposition, that is an interesting finding: it would suggest that the network learns to compensate for input encoding quality. The analysis document reports this honestly either way, consistent with Anthropic's stated value of communicating null results.

- **Research-to-engineering translation.** The project takes two academic research threads (heterogeneous anomaly detection and the superposition hypothesis) and builds concrete, reproducible tooling around them, producing auditable artifacts at every stage.

The `docs/analysis.md` document and the application essay on interpretability motivations can reference the Module 7 findings directly. The Streamlit demo's interference matrix heatmap panel provides a visual that is immediately legible to anyone familiar with the Toy Models of Superposition paper.

---

## Reference Material

- **Paper:** Li, Z. & Fan, R. (2026). *Explainable Heterogeneous Anomaly Detection in Financial Networks via Adaptive Expert Routing.* arXiv:2510.17088v2. https://arxiv.org/abs/2510.17088
- **Dataset:** IEEE-CIS Fraud Detection. https://www.kaggle.com/competitions/ieee-fraud-detection
- **Superposition:** Elhage, N. et al. (2022). *Toy Models of Superposition.* Transformer Circuits Thread. https://transformer-circuits.pub/2022/toy_model/index.html
- **Capacity:** Scherlis, A. et al. (2022). *Polysemanticity and Capacity in Neural Networks.* arXiv:2210.01892. https://arxiv.org/abs/2210.01892
- **Encoding reference:** Potdar, K. et al. (2017). A Comparative Study of Categorical Variable Encoding Techniques for Neural Networks. *ICCAIRO.*
- **LOF:** Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers. ACM SIGMOD.
- **PuLP docs:** https://coin-or.github.io/pulp/
- **HiGHS solver:** https://highs.dev/
