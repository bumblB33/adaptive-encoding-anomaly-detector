# Adaptive Encoding Anomaly Detector


## 1. Feature summary

Research the intersection of constraint-based optimization, anomaly detection, and neural-network interpretability.

1. **Encoding Selection Pipeline.** Feature encoding is treated as an optimization problem using a Mixed-Integer Linear Program (MILP). MILP is an optimization algorithm that chooses among discrete options, one encoding per feature, while respecting linear constraints. In the context of this project, the MILP's objective is to minimize `α·loss + β·(dim/D_max) + γ·(1−xplain)`, a weighted sum of detection error, encoded size, and un-interpretability. The solver is **PuLP + HiGHS** (a Python modeling library + open-source solver). The primary detector on the IEEE-CIS Fraud Detection dataset is Local Outlier Factor (LOF), which flags outliers by identifying data points sitting in unusually sparse neighborhoods. The MILP-selected set of encodings is then compared against a plain one-hot encoded (uniform-OHE) baseline by measuring area under the precision–recall curve (PR-AUC). This was chosen because in the labeled dataset, fraud is rare (roughly 3.5% of transactions). PR-AUC was chosen because it ignores the large number of true-negative samples that another metric might reward. The pipeline also produces structured explanations for individual anomalies by running leave-one-feature-out (LOFO) on the top anomalies and features by mutual information (MI).

2. **Interpretability (representation geometry).** Two identical multilayer perceptrons (MLPs) are used. These are basic feedforward neural networks. `MLP-MILP` and `MLP-OHE` are trained on the two encoding regimes. They share the same architecture, the same random seed, and the same hyperparameters. Only the inputs differ. Activations are frozen for both, and superposition for each is measured three ways. Superposition is a state where a neural network is representing more features than it has neurons. It does so by letting them share overlapping directions at the cost of interference. Interference, broadly speaking, is when no clean separation of concept representations per-neuron can be identified via probe. The three measurements used to evaluate superposition are: 

(1) The interference matrix `Wᵀ·W` (summarized by the off-diagonal Frobenius norm) 
(2) Per-segment linear-probe accuracy, measuring for interference. (3) Per-segment capacity: the fraction of each neuron's activation variance attributable to each segment. MLPs are controlled instruments for studying representation geometry. They are distinct from the anomaly detection.


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

## Pipeline

Dataset: IEEE-CIS Fraud Detection dataset from Kaggle. It contains over 400 mixed-type features and ground-truth fraud labels. The fraud rate is around 3.5 percent, making PR-AUC the best metric for evaluating performance.

The system architecture runs through several distinct stages:

1. **Feature Profiler:** Evaluates every feature to determine its data type, cardinality, missingness, and mutual information with the fraud label. The label is used strictly for profiling. The main anomaly detector never sees it as a target.
2. **Feature Segmenter:** Groups features into five domain labels (transaction amount, identity/device, behavioral frequency, temporal/timing, and card/account). This stage uses a hybrid approach, relying on rule-based logic for known naming conventions and k-means clustering for the rest.
3. **Encoding Candidate Evaluator:** Computes the detection loss, encoded dimensionality, and an interpretability score for every feature and encoding pair.
4. **MILP Encoding Selector:** The core optimizer. It selects exactly one encoding per feature under a dimensionality budget using PuLP and the HiGHS solver.
5. **Anomaly Detector:** Runs Local Outlier Factor on the MILP-selected features and compares the PR-AUC against a uniform one-hot encoded baseline.
6. **Structured Explainer:** Uses a leave-one-feature-out method to explain why specific transactions were flagged.
7. **Representation Geometry Analysis:** A separate module to evaluate how the encoding choice impacts neural network superposition.
8. **Streamlit App:** A frontend to adjust the optimization weights and view the anomaly explanations.

## The MILP Objective

Arriving at the optimization objective required some iteration. The solver minimizes three factors: anomaly detection loss, encoded dimensionality, and un-interpretability. The objective is defined as `α·loss + β·(dim/D_max) + γ·(1−xplain)`.

The normalization is what makes this work. Without dividing the dimension by `D_max`, the dimensionality term completely dominates the equation. Post-encoding dimensions can reach into the hundreds, while loss and the interpretability score naturally live between 0 and 1. With dimension unscaled, the solver ignores the other terms and simply tries to shrink the dataset. `D_max` represents the maximum possible dimensionality if every feature used its most expensive encoding. By dividing by this precomputed constant, all three terms are scaled to a 0 to 1 range, so the user-defined weights carry the meaning they are intended to.

The interpretability score (`xplain`) is assigned by a rubric based on human readability. Passthrough continuous variables score a 0.90, while target encoding drops to 0.35. Features with an explicit, confirmed order receive a 0.70 for ordinal encoding, while unconfirmed ordinal features are penalized with a 0.30 to avoid injecting spurious structure.

## Representation Geometry

The central question is whether upstream encoding decisions actually alter the geometry of learned representations. To test this, two identical multilayer perceptrons are trained. One takes the MILP-selected encodings, and the other takes the one-hot baseline.

The goal is to measure superposition, which can be understood through a filing-cabinet analogy. Consider two drawers (neurons) and fifty folders (features). Multiple folders are packed into each drawer for lack of space. As long as two folders from the same drawer are rarely needed at once, everything works fine. When they are needed at the same time, the contents become entangled, and the ability to say that a specific drawer is dedicated to a specific topic is lost.

That entanglement is the cost of superposition. Following work by Elhage (2022) and Scherlis (2022), it is measured in three ways using the frozen first-hidden-layer activations:

* **Interference matrix:** Measures how much the folder directions overlap in weight space, via the Frobenius norm of the off-diagonal mass.
* **Linear probes:** Tests whether the segment label can be predicted from the activations alone. High accuracy means the drawer is structured enough to be read cleanly.
* **Capacity:** Computes the fraction of a neuron's activation variance attributable to a specific segment.

## Current Progress

So far, the foundation and feature segmentation phases are complete. The repository is set up with a pinned random seed for reproducibility. The profiler computes mutual information, and the segmenter sorts features into the five target buckets.

The next step is to build the Encoding Candidate Evaluator to generate the matrix of scores. Once that produces correct output, the MILP selector follows, along with testing of the continuous LP relaxation. If the LP solve matches the exact MILP solution frequently enough, it can be used to make the Streamlit sliders highly responsive. The remaining work is then implementing the Local Outlier Factor detector and analyzing the geometry metrics. The implementation effort there is modest; the principal value lies in understanding how these encodings change the neural network's internal structure.


## 9. External references

### Superposition and polysemanticity

- Elhage, N. et al. (2022). *Toy Models of Superposition.* Transformer Circuits Thread. https://transformer-circuits.pub/2022/toy_model/index.html
- Scherlis, A. et al. (2022). *Polysemanticity and Capacity in Neural Networks.* arXiv:2210.01892. https://ar5iv.labs.arxiv.org/html/2210.01892
- *200 COP in MI: Exploring Polysemanticity and Superposition* (LessWrong). https://www.lesswrong.com/posts/o6ptPu7arZrqRCxyz/200-cop-in-mi-exploring-polysemanticity-and-superposition
- *Superposition is Not Just Neuron Polysemanticity* (Alignment Forum). https://www.alignmentforum.org/posts/8EyCQKuWo6swZpagS/superposition-is-not-just-neuron-polysemanticity

### Multi-objective optimization

- Weighted Sum Method (ScienceDirect topic page). https://www.sciencedirect.com/topics/computer-science/weighted-sum-method

The canonical spec [docs/project-plan.md](project-plan.md)
also carries a *Reference Material* section with the academic citations
behind specific design decisions (Li & Fan 2026, arXiv:2510.17088, for
the heterogeneous-anomaly framing; Potdar et al. 2017, IJCA 175(4), as
informing — not dictating — the encoding interpretability ordering;
Breunig et al. 2000 for LOF, plus PuLP and HiGHS solver docs).
