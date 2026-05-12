# Implementation Plan: Adaptive Encoding Anomaly Detector

**Status:** In progress. Week 1 complete — scaffolding pinned by tests,
ordinal annotation seeded from EDA, Module 1 (feature profiler) built
with passing tests. Module 2 (Week 2) is the next module to start.
**Last synced:** 2026-05-11
**Primary repo:** adaptive-encoding-anomaly-detection (this repo)
**Additional repos:** none
**Plan path:** docs/implementation-plan.md

---

## Feature summary

A portfolio project for an Anthropic Research Scientist (Interpretability)
application. The work rests on two separable claims.

1. **Engineering (Modules 1 through 6).** Feature encoding selection is framed as a
   Mixed-Integer Linear Program (PuLP + HiGHS) with a normalized,
   constraint-aware objective `α·loss + β·(dim/D_max) + γ·(1−xplain)`. LOF
   is the primary anomaly detector on the IEEE-CIS Fraud Detection dataset.
   The MILP-selected encoding regime is compared against a uniform-OHE
   baseline on PR-AUC.

2. **Interpretability (Module 7).** Two identical MLPs (`MLP-MILP` and
   `MLP-OHE`) are trained on the two encoding regimes. They share the same
   architecture, the same seed, and the same hyperparameters; only the
   inputs differ. Superposition is measured three ways on frozen
   activations: interference matrix `Wᵀ·W` (off-diagonal Frobenius norm),
   per-segment linear probe accuracy, and per-segment capacity. The MLPs
   are controlled instruments for studying representation geometry, not
   anomaly detectors.

Deliverable cadence is an 8-week build; current status is end-of-Week-1
scaffolding.

---

## Source inventory

| Source | Type | Identifier | Last synced |
|---|---|---|---|
| Canonical spec | Markdown (tracked) | [docs/portfolio_project_plan.md](portfolio_project_plan.md) | 2026-05-11 |
| Design-rationale notes | Markdown (tracked) | [docs/rationale.md](rationale.md) | 2026-05-11 |
| Top-level README (reproducibility entry point) | Markdown (tracked) | [README.md](../README.md) | 2026-05-11 |
| Project instructions for Claude Code | Markdown (gitignored, load-bearing) | [CLAUDE.md](../CLAUDE.md) | 2026-05-11 |

`CLAUDE.md` is gitignored (a personal instruction file for Claude Code)
but documents the same invariants this plan tracks. It is included in
the inventory because it is consulted on every Claude Code session and
its drift would matter.

---

## Acceptance criteria

Status key: `[x]` met • `[~]` partially met • `[ ]` not started.
Each item's bold lead phrase is the stable anchor; cross-references
elsewhere in this doc use the lead phrase, not a positional ID.

### Engineering claim (Modules 1 through 6)

- [x] **Module 1 — feature profile.** Produces a per-feature profile with
  detected type, cardinality, distribution shape, missingness rate, and
  mutual information against the binary fraud label via
  `mutual_info_classif`. The MI value is used for characterization only;
  LOF and the Module 7 MLPs never see the label as a training target.
  **Evidence:** [src/feature_profiler.py](../src/feature_profiler.py)
  exposes `profile_dataframe`, `save_profile`, `load_profile`; the
  "characterization only" invariant is documented in the module
  docstring. Behavior pinned by 15 tests in
  [tests/test_profiler.py](../tests/test_profiler.py): coverage of every
  column, label MI is `None`, signal columns have higher MI than noise,
  detected-type/cardinality/missing-rate accuracy, reproducible MI under
  a fixed `random_state`, save/load round-trip. Smoke-run on a 50K
  stratified IEEE-CIS sample produced sensible MI rankings (V-features
  dominate); the output `outputs/feature_profile.json` is gitignored
  but recomputable from `data/raw/`.
- [ ] **Module 2 — feature segmentation.** Segments features into the 5
  domain-labeled groups ("transaction amount", "identity/device",
  "behavioral frequency", "temporal/timing", "card/account") using a
  hybrid of rule-based logic (cardinality and dtype) and k-means on the
  feature-profile vector. Module 7's per-segment metrics (see *Module 7
  — three superposition metrics* below) bind to this fixed 5-segment
  vocabulary. **Evidence:** no `src/feature_segmenter.py` exists yet.
- [ ] **Module 3 — encoding evaluation matrix.** Produces
  `outputs/evaluation_matrix.csv` for the top-N features by variance
  and/or MI, where N is pinned in `config/defaults.py` as
  `ENCODING_EVAL_MAX_FEATURES`. Features outside the top-N are recorded
  with the passthrough encoding only. Each row in the matrix represents
  one (evaluated feature, encoding) pair and contains detection loss
  (LOF on a 10% stratified sample), encoded dimensionality, and a
  rubric-based interpretability score. The rubric reads
  `config/ordinal_features.yaml`: features listed there receive 0.70
  for the ordinal encoding row, and features not listed receive 0.30
  (uncertain-ordering penalty); non-ordinal rubric rows are unaffected.
  Detection loss and rubric score are in [0, 1]; dimensionality is a
  positive integer. **Evidence:** no `src/encoding_evaluator.py` exists
  yet; `ENCODING_EVAL_MAX_FEATURES` not yet defined in
  [config/defaults.py](../config/defaults.py).
- [ ] **Module 4 — MILP solver.** Solves the MILP
  `minimize α·loss + β·(dim/D_max) + γ·(1−xplain)` subject to (one
  encoding per feature, total dim ≤ budget, interpretability floor per
  segment) using PuLP + HiGHS. `D_max` is precomputed analytically from
  `evaluation_matrix.csv` before the solver runs. Infeasibility surfaces
  a clear error rather than a degenerate result. **Evidence:** no
  `src/milp_selector.py` exists yet; `pulp` and `highspy` declared in
  the `milp` optional extra in [pyproject.toml](../pyproject.toml).
- [ ] **Module 5 — LOF and baseline comparison.** Runs LOF on the
  MILP-selected encoded space and on the uniform-OHE baseline, reports
  PR-AUC as the headline metric, and P/R/F1 at the
  `LOF_THRESHOLD_PERCENTILE` (= 96.5) prevalence-percentile threshold
  (i.e., the top ≈ 3.5% of transactions by LOF score). No SMOTE.
  **Evidence:** no `src/anomaly_detector.py` exists yet.
  `LOF_THRESHOLD_PERCENTILE` defined in
  [config/defaults.py](../config/defaults.py).
- [ ] **Module 6 — structured explanations.** Produces a three-layer
  explanation for the top `LOFO_MAX_ANOMALIES` (= 100) anomalies by LOF
  score, running leave-one-out over the top `LOFO_MAX_FEATURES` (= 50)
  by MI, on the 10% stratified sample from Module 3, in either
  `'feature'` or `'segment'` mode (`LOFO_MODE`). **Evidence:** no
  `src/explainer.py` exists yet. `LOFO_MAX_ANOMALIES`,
  `LOFO_MAX_FEATURES`, and `LOFO_MODE` defined in
  [config/defaults.py](../config/defaults.py).

### Interpretability claim (Module 7)

- [ ] **Module 7 — identical MLP architectures.** `MLP-MILP` and
  `MLP-OHE` train with identical architecture (2 hidden layers, ReLU,
  binary fraud classification head), the same `RANDOM_SEED`, and the
  same hyperparameters; only the input encoding differs. **Evidence:**
  no `src/representation_analysis.py` exists yet; `torch` declared in
  the `geometry` optional extra in [pyproject.toml](../pyproject.toml).
- [ ] **Module 7 — three superposition metrics.** Computed on frozen
  first-hidden-layer activations: (1) interference matrix `Wᵀ·W` with
  Frobenius norm of off-diagonal terms as scalar summary; (2)
  per-segment linear probe accuracy across the 5 feature segments; (3)
  per-segment capacity as the fraction of each neuron's activation
  variance attributable to each segment. Documentation acknowledges the
  capacity metric as a practical operationalization inspired by
  Scherlis et al. (2022), whose formal definition uses fractional
  embedding dimension. **Evidence:** no code yet.
- [ ] **Module 7 — side-by-side comparison.** Reports all three metrics
  across `MLP-MILP` and `MLP-OHE`, published in
  `notebooks/03_representation_geometry.ipynb` and the Streamlit demo.
  **Evidence:** no notebook yet.

### Demo and write-up

- [ ] **Streamlit demo — panels.** `demo/app.py` exposes: encoding
  decision table, per-anomaly explanation cards, baseline comparison
  panel, representation-geometry panel (interference matrix heatmaps +
  probe accuracy table), and α/β/γ + dimensionality-budget sliders.
  Deployed to Streamlit Community Cloud (free). **Evidence:** only
  [demo/.gitkeep](../demo/.gitkeep); `streamlit` declared in the `demo`
  optional extra in [pyproject.toml](../pyproject.toml).
- [ ] **Streamlit demo — encoding-decision source.** Encoding decisions
  load from a pre-computed grid of ~60 MILP solutions keyed by (α, β,
  γ, budget) by default, committed under
  `outputs/sample/encoding_decisions/` so the Streamlit Cloud demo
  loads them from a clean clone. Upgrade to live LP relaxation is
  gated on the Week-4 benchmark: ≥95% of (α, β, γ, budget) grid points
  produce LP-rounded decision matrices identical to the MILP solution,
  no point violates the dimensionality budget after rounding, and every
  solve runs in <500ms. **Evidence:** no precomputed grid or LP
  benchmark yet.
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
  `random_state` kwarg defaulting to this constant. **Evidence:**
  `RANDOM_SEED` in [config/defaults.py](../config/defaults.py), pinned
  by `test_random_seed_is_42` in
  [tests/test_config_defaults.py](../tests/test_config_defaults.py).
  Per-module enforcement happens as each module is built.
- [ ] **30-minute reproduction.** A new contributor can reproduce all
  pipeline outputs in under 30 minutes by setting `RANDOM_SEED` in
  `config/defaults.py` and running the pipeline end-to-end.
  **Evidence:** the pipeline does not exist yet; reproducibility README
  finalization is a Week-8 deliverable.

---

## Definition of done

- [~] **Per-module tests pass.** Tests are added when the module is
  built (staggered). **Evidence:** `test_config_defaults.py` and
  `test_ordinal_features_yaml.py` cover the only artifacts that exist
  today; `[tool.pytest.ini_options]` configured in
  [pyproject.toml](../pyproject.toml).
- [ ] **Expensive computations cache to `outputs/`.** The encoding
  evaluation grid, MILP solutions, and MLP training each cache their
  result. Sample artifacts needed by the demo live in `outputs/sample/`,
  the only carve-out from the `outputs/` gitignore rule. **Evidence:**
  no expensive computations exist yet; the carve-out (`outputs/*` with
  negations for `outputs/.gitkeep` and `outputs/sample/**`) is in
  [.gitignore](../.gitignore).
- [x] **Raw IEEE-CIS data is never committed.** **Evidence:** `*.zip`
  rule in [.gitignore](../.gitignore); the local
  `data/ieee-fraud-detection.zip` is gitignored;
  [README.md](../README.md) documents the Kaggle download.
- [~] **README.md finalized for reproducibility.** **Evidence:**
  [README.md](../README.md) exists at the repo root with installation,
  Kaggle download steps, and a repo-layout tour. Full reproducibility
  (≤30-min end-to-end run from a clean clone) requires Modules 1
  through 7 to be built; this remains a Week-8 deliverable.
- [ ] **Live Streamlit Community Cloud URL recorded in README.**
  **Evidence:** no demo yet.
- [ ] **Null-result reporting honored in `docs/analysis.md`.** If
  `MLP-MILP` and `MLP-OHE` exhibit equivalent superposition, that
  finding is reported and interpreted as evidence of network
  compensation, not omitted. **Evidence:** Week 8 deliverable.

---

## Current state

Single repo. Week 1 complete: scaffold + pinned config + dependency
manifest from the initial commit, plus the Week-1 deliverables built on
top — ordinal annotation seeded, EDA notebook documenting the ordinal
choices, and Module 1 (feature profiler) implemented and tested.

### Repo layout (what exists)

- [pyproject.toml](../pyproject.toml): package `adaptive-encoding-anomaly-detector` v0.1.0,
  Python ≥ 3.11, core deps `numpy/scipy/scikit-learn/joblib/pandas/pyyaml`,
  optional extras `dev = [pytest]`, `milp = [pulp, highspy]`,
  `geometry = [torch]`, `demo = [streamlit]`, pytest configured against
  `tests/`.
- [config/defaults.py](../config/defaults.py): `RANDOM_SEED = 42`,
  `LOF_THRESHOLD_PERCENTILE = 96.5`, `LOFO_MAX_ANOMALIES = 100`,
  `LOFO_MAX_FEATURES = 50`, `LOFO_MODE = "feature"`. The docstring
  documents the role of each constant.
- [config/ordinal_features.yaml](../config/ordinal_features.yaml):
  two entries — `id_34` (4-value `match_status:N` sequence) and `M4`
  (3-value `M0`/`M1`/`M2` sequence). The defense for these (and the
  deliberate exclusion of `id_23` and `id_15`) lives in the "Ordinal
  candidate investigation" section of `notebooks/01_eda.ipynb`. Every
  other categorical falls back to the 0.30 unlisted score in the
  Module 3 rubric.
- [tests/test_config_defaults.py](../tests/test_config_defaults.py):
  pins all five constants; one test validates `LOFO_MODE ∈ {"feature", "segment"}`.
- [tests/test_ordinal_features_yaml.py](../tests/test_ordinal_features_yaml.py):
  validates schema, namely a top-level mapping from string feature name to a
  non-empty list of ordered values; an empty mapping is explicitly
  accepted as the Week-1-in-progress state.
- [.gitignore](../.gitignore): gitignores `.venv/`, `*.zip`,
  `__pycache__/`, IDE files, build artifacts; `outputs/*` ignored
  except `outputs/.gitkeep` and `outputs/sample/**`; `CLAUDE.md` listed
  individually.
- [src/__init__.py](../src/__init__.py), [config/__init__.py](../config/__init__.py),
  [tests/__init__.py](../tests/__init__.py): empty package markers.
- [src/feature_profiler.py](../src/feature_profiler.py): Module 1
  implementation — `profile_dataframe`, `save_profile`, `load_profile`.
  The fraud label is used for MI characterization only; the module
  docstring states this invariant explicitly.
- [tests/test_profiler.py](../tests/test_profiler.py): 15 tests covering
  every column field plus the MI/label-isolation contract and JSON
  round-trip.
- [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb): Week-1 EDA pass —
  type distribution, categorical landscape, ordinal candidate
  investigation, narrative conclusions feeding Modules 1 and 2.
- [demo/.gitkeep](../demo/.gitkeep), [outputs/.gitkeep](../outputs/.gitkeep):
  directories exist as placeholders.
- [docs/](.): holds this plan plus the canonical spec
  [portfolio_project_plan.md](portfolio_project_plan.md) and
  design-rationale notes [rationale.md](rationale.md). All three
  are tracked. Data-download and reproduction instructions live in
  [README.md](../README.md) at the repo root.

### Local-only (gitignored) artifacts present

- `data/ieee-fraud-detection.zip`: raw dataset, present locally only.
- `CLAUDE.md`: project instructions for Claude Code.

### Module status

- **Module 1, Feature Profiler:** complete.
  [src/feature_profiler.py](../src/feature_profiler.py) +
  [tests/test_profiler.py](../tests/test_profiler.py) (15 tests).
- **Module 2, Feature Segmenter:** not started.
- **Module 3, Encoding Candidate Evaluator:** not started.
- **Module 4, MILP Encoding Selector:** not started; PuLP + HiGHS
  declared in optional extra `milp`.
- **Module 5, Anomaly Detector:** not started; PR-AUC threshold config
  pinned.
- **Module 6, Structured Explainer:** not started; LOFO config pinned.
- **Module 7, Representation Geometry:** not started; `torch` declared
  in optional extra `geometry`.
- **Streamlit demo:** not started; `streamlit` declared in optional
  extra `demo`.
- **Analysis write-up:** not started.

---

## Remaining work

Ordered by Week 1 through Week 8. Each item's bold lead phrase pairs
with the corresponding acceptance criterion above by topic; where one
item spans multiple criteria or another remaining-work item, the body
spells out the linkage by lead phrase.

### Week 1: finish foundation

- [x] **Ordinal annotation.** Two entries seeded in
  [config/ordinal_features.yaml](../config/ordinal_features.yaml):
  `id_34` (`match_status:-1/0/1/2`) and `M4` (`M0/M1/M2`). Every other
  IEEE-CIS categorical falls back to the 0.30 unlisted-feature penalty
  in the Module 3 rubric. Defense for the included pair and explicit
  exclusion of close calls (`id_23`, `id_15`) lives in the "Ordinal
  candidate investigation" section of
  [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb). Schema continues
  to pass
  [tests/test_ordinal_features_yaml.py](../tests/test_ordinal_features_yaml.py).
- [x] **Week-1 EDA notebook.**
  [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) covers type
  distribution across the merged 590,540 × 434 frame, the categorical
  landscape (31 string columns, top-3 values and missingness), and an
  ordinal candidate investigation that justifies the two entries above.
  Final markdown cell orients Modules 1 and 2 (feature profiler
  invariants, segment vocabulary mapping). The notebook is committed
  unexecuted; the raw CSVs it reads from `data/raw/` are gitignored.
- [x] **Module 1 implementation.**
  [src/feature_profiler.py](../src/feature_profiler.py) exposes
  `profile_dataframe(df, label_column='isFraud', random_state=RANDOM_SEED)`
  returning per-column dtype / detected_type / cardinality / missing
  rate / distribution / mutual information, plus `save_profile` and
  `load_profile`. The "label used for characterization only" invariant
  is stated in the module docstring. Behavior is pinned by 15 tests in
  [tests/test_profiler.py](../tests/test_profiler.py); MI reproducibility
  uses `RANDOM_SEED` from
  [config/defaults.py](../config/defaults.py). Smoke-run on a 50K
  stratified IEEE-CIS sample produces sensible MI rankings.

### Week 2: segmentation

- [ ] **Module 2 implementation.** Implement
  `src/feature_segmenter.py`: hybrid rule-based + k-means clustering on
  the feature-profile vector, emitting domain-labeled segments. Accepts
  `random_state` defaulting to `config.defaults.RANDOM_SEED`. Output:
  `outputs/segment_assignments.json`. Add `tests/test_segmenter.py`.

### Week 3: encoding evaluation grid

- [ ] **Module 3 implementation.** Implement
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

### Week 4: MILP

- [ ] **Install `milp` extra.** `pip install -e ".[milp]"` so `pulp`
  and `highspy` are available.
- [ ] **Module 4 implementation.** Implement `src/milp_selector.py`.
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
  carve-out keeps the grid tracked. Supports the *Streamlit demo —
  encoding-decision source* criterion above.
- [ ] **LP-relaxation benchmark gate.** For the same ~60 grid points,
  compare LP-rounded decisions to MILP decisions. Record: (a) the
  fraction of grid points whose full decision matrices are identical
  between LP-rounded and MILP, which is the gate criterion; (b) any
  post-rounding budget violations; (c) per-solve latency. If ≥95% of
  grid points produce identical decision matrices, no point violates
  the budget after rounding, and every solve runs in <500ms, mark the
  demo eligible for the live-LP upgrade. Otherwise the demo stays on
  the precomputed grid. Document the result. Satisfies the *Streamlit
  demo — encoding-decision source* criterion by either route.

### Week 5: anomaly detection and baseline comparison

- [ ] **Pipeline deps for full dataset.** Install pipeline deps as
  needed (kaggle for dataset download is already documented in
  [README.md](../README.md)).
- [ ] **Module 5 implementation.** Implement
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

### Week 6: explanation layer

- [ ] **Module 6 implementation.** Implement `src/explainer.py`. Build
  a leave-one-feature-out wrapper around LOF. Apply compute-scope caps
  from `config.defaults` (`LOFO_MAX_ANOMALIES = 100`,
  `LOFO_MAX_FEATURES = 50`); reuse the Module-3 10% stratified sample
  for the LOF re-computation; respect `LOFO_MODE` (`"feature"` or
  `"segment"`). Emit the three-layer explanation object as JSON +
  human-readable text. Add `tests/test_explainer.py`: verify
  three-layer output structure, valid floats, and that the
  near-zero-contribution edge case is handled. Output:
  `outputs/explanations/sample_explanations.json`.

### Week 7: representation geometry and demo

- [ ] **Install `geometry` extra.** `pip install -e ".[geometry]"`.
- [ ] **Module 7 implementation.** Implement
  `src/representation_analysis.py`. Train `MLP-MILP` and `MLP-OHE`:
  identical 2-hidden-layer ReLU architecture, binary fraud
  classification head, the same `RANDOM_SEED`, the same hyperparameters;
  only the encoded inputs differ. The MLP is not the anomaly detector;
  note this in the docstring. Compute the three metrics on frozen
  first-hidden-layer activations:
  (1) `Wᵀ·W` and its off-diagonal Frobenius norm,
  (2) per-segment linear probe accuracy across the 5 segments,
  (3) per-segment capacity as the fraction of each neuron's activation
  variance attributable to each segment. Document the
  Scherlis-et-al.-2022 operationalization caveat in the module
  docstring. Output `outputs/geometry/` and
  `notebooks/03_representation_geometry.ipynb`. Add
  `tests/test_representation_analysis.py`: verify `Wᵀ·W` is square and
  PSD (off-diagonal Frobenius ≥ 0), per-segment probe accuracy ∈ [0,1],
  and per-neuron capacity sums ≈ 1.0 within floating-point tolerance,
  against a tiny synthetic MLP. Satisfies the three *Module 7* criteria
  above (identical architectures, three metrics, side-by-side
  comparison).
- [ ] **Install `demo` extra.** `pip install -e ".[demo]"`.
- [ ] **Build Streamlit app.** Build `demo/app.py`: encoding decision
  table, anomaly explanation cards, baseline comparison, and a
  representation-geometry panel (interference heatmaps side-by-side +
  probe accuracy table). Wire sliders to load from the precomputed
  grid produced by *Run MILP across the grid* above (default). If the
  *LP-relaxation benchmark gate* passed, add the live-LP-relaxation
  path; otherwise leave sliders snapping to grid points. Deploy to
  Streamlit Community Cloud and record the URL in the README.
  Satisfies the *Streamlit demo — panels* and *Streamlit demo —
  encoding-decision source* criteria above.

### Week 8: analysis and polish

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
  than neutral, and is itself an interpretability finding about the
  MILP objective structure. Publish both the flip-fraction stability
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
  canonical spec calls this out as optional in Week 8.

---

## Open questions and unresolved conflicts

None at this consolidation. Every row of the canonical spec's
Open Design Decisions table is resolved and reflected in this plan.
The Week-4 LP-relaxation benchmark gate is not an open question;
it is an explicit, gated decision that will resolve once Module 4
exists.

---

## Change log

### 2026-05-11 — Week 1 closeout: ordinal annotation, EDA notebook, Module 1

- Populated [config/ordinal_features.yaml](../config/ordinal_features.yaml)
  with `id_34` and `M4` (the only IEEE-CIS categoricals whose string
  surface literally encodes an ordering); all other categoricals fall
  back to the 0.30 unlisted score by design.
- Added [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb): type
  distribution, categorical cardinality table, ordinal candidate
  investigation with explicit exclusion rationale for `id_23` and
  `id_15`, and a closing narrative orienting Modules 1 and 2.
- Added [src/feature_profiler.py](../src/feature_profiler.py) (Module 1)
  with `profile_dataframe` / `save_profile` / `load_profile` and the
  "label used for characterization only" invariant in the module
  docstring; 15 tests in
  [tests/test_profiler.py](../tests/test_profiler.py).
- Flipped acceptance criterion *Module 1 — feature profile* from `[ ]`
  to `[x]` and the three *Week 1* remaining-work items to `[x]`.
- `.gitignore` updated to cover extracted CSVs (`data/*.csv`,
  `data/raw/`) so the IEEE-CIS extracts stay local-only.
- No conflicts surfaced; this is forward progress, not a re-litigation.

---

## External references

### Job posting

- Anthropic, Research Scientist, Interpretability:
  https://job-boards.greenhouse.io/anthropic/jobs/4980427008

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

The canonical spec [docs/portfolio_project_plan.md](portfolio_project_plan.md)
also carries a *Reference Material* section with the academic citations
behind specific design decisions (Li & Fan 2026 for the heterogeneous-
anomaly framing, Potdar et al. 2017 for encoding interpretability
ordering, Breunig et al. 2000 for LOF, plus PuLP and HiGHS solver
docs).
