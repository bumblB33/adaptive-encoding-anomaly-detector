## 2. Glossary

**Encoding**

- **Encoding**. How a raw data column is formatted into model input.

- **One-hot (OHE)**. A singular binary 0/1 column per category. Assumes no inherent order, but increases the number of columns.

- **Ordinal**. Map categories to ordered integers (0, 1, 2, …). This can introduce implicit ordering, and should only be used when the categories have a genuine and meaningful order.

- **Passthrough**. Leave a numeric column as-is, no transform.

- **Cardinality**. How many distinct values a column has.

- **Missingness / missing rate**. Fraction of rows where the value is absent.

- **Dimensionality / dimension**. How many columns the encoded data has.
More categories become more dimensions.

**Optimization**

- **MILP**. Mixed-Integer Linear Program. An optimization algorithm that chooses discrete options to minimize a linear objective under linear constraints. In this context, the linear objective is to pick one encoding per feature.

- **Objective `α·loss + β·(dim/D_max) + γ·(1−xplain)`**. A weighted sum of three costs, each scaled into [0, 1] so they are comparable.

- `loss`. How poorly anomalies are detected under that encoding.

- `dim/D_max`. Encoded size, normalized.

- `1 − xplain`. Un-interpretability (`xplain` is the rubric score below. High interpretability is on one end, with low interpretability (with consequentially lower cost) on the other.).

- `α, β, γ`. Weights that can be tuned to trade off between the three costs.

- **D_max**. The largest total dimensionality possible (sum, over features, of each feature's most-expensive encoding). It's a fixed normalization anchor that puts the size term on a 0–1 scale. It is not a constraint the solver must satisfy.

- **Interpretability rubric (`xplain`)**. A score in [0, 1] for assessing how human-readable each encoding is.

- **Dimension budget**. A hard cap on total encoded columns.

- **Infeasibility**. When no assignment satisfies every constraint (e.g., the budget is too tight to also meet the interpretability floor); the solver should report accordingly.

- **LP relaxation**. Allows for solving the easier continuous version of a problem by allowing fractional choices and rounding them. Faster, but potentially less accurate. Only trusted in this context if it matches the exact MILP on the MILP-relaxation benchmark.

**Anomaly Detection**

- **LOF (Local Outlier Factor)**. Unsupervised outlier detector. Scores a point by how much sparser its local neighborhood is than its neighbors'. A higher score means a data point is more of an outlier compared to similar neighborhoods of data points. The fraud label is never incorporated as input in its scoring assessment.

- **PR-AUC**. Area under the precision–recall curve; the preferred single number when positives (fraud) are rare, because it ignores the large, easy true-negative mass that ROC-AUC rewards.

- **Precision / Recall / F1**. At a chosen cutoff: precision = of the flagged, how many are fraud; recall = of the fraud, how many were flagged; F1 = their harmonic mean.

- **Stratified sample**. A subsample that preserves the class balance (here ~3.5% fraud), so a 10% slice still "looks like" the whole.

- **Mutual information (MI)**. A measure (≥ 0; 0 means independent) of how much knowing a feature reduces uncertainty about the label. Unlike correlation, it catches nonlinear association. Computed with `_mutual_info_classifier`. Used for *characterization only*. See the feature-profiler invariant.

- **k-means**. Clusters points into *k* groups, each point assigned to the nearest group average (centroid).

- **LOFO (leave-one-feature-out)**. Re-run the detector with one feature removed and watch how an anomaly's score moves; the change attributes "blame" to that feature. This is the explainer's engine.

**Interpretability & Neural Network Geometry**

- **MLP**. Multilayer perceptron. A model where input is passed to hidden layers, processed, and returned as output. Training is handled via backpropagation. This project uses a 2-hidden-layer ReLU net with a fraud head.

- **Superposition**. A network representing more features than it has neurons by letting features share overlapping directions; the price is *interference* (features bleed into each other). The "filing cabinet with more folders than drawers" framing in `docs/rationale.md`.

- **Interference matrix `Wᵀ·W`**. `W` is the weight matrix from inputs to the first hidden layer; `Wᵀ·W` ("W-transpose times W") is a square matrix whose off-diagonal entries measure how much two input features get written into the same neurons.

- **Frobenius norm**. The overall "size" of a matrix: the square root of the sum of all its squared entries. Taking it over just the off-diagonal entries collapses interference to a single scalar.

- **Linear probe**. A simple linear classifier trained to read a property (here: which of the five segments a feature belongs to) off the frozen hidden activations. Its accuracy describes how cleanly that property is encoded.

- **Capacity**. The fraction of a neuron's activation variance attributable to each segment. Inspired by but distinct from Scherlis et al. (2022), whose formal definition uses fractional embedding dimension.

**Reproducibility**

- **Random seed**. A fixed number (`RANDOM_SEED = 42`) that makes randomized steps (k-means init, sampling, MLP weight init) repeatable run-to-run.
