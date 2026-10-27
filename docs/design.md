# Design

This file owns project intent. The README owns what is built. Where they differ, the README status table is correct and this file is the plan.

## Question

Does the encoding chosen for each feature change what an anomaly detector finds, and how a neural network organizes what it learns?

The second half is a hypothesis, not an established fact. For models that apply an affine transformation to their inputs, such as an MLP, one-hot encoding can mimic any other encoder by learning suitable weights (Zhu et al. 2024, Theorem 1). Other encoders can restrict the first layer but cannot expand it. The practical gap comes from finite data: the weights must be learned from enough samples per category. The geometry experiment tests whether encodings still change how activations are organized. A null result is informative. It would mean the network compensates for the encoding.

## Pipeline

Stages 1 and 2 are built. Stages 3 to 8 are planned.

1. **Feature Profiler.** Dtype, cardinality, missing rate, and MI with the label. The label is used for profiling only.
2. **Feature Segmenter.** Five segments: rules first, k-means for the remainder.
3. **Encoding Candidate Evaluator.** For every feature and encoding pair, compute detection loss, encoded dimensionality, and an interpretability score.
4. **MILP Encoding Selector.** Choose exactly one encoding per feature under a dimension budget with PuLP and HiGHS.
5. **Anomaly Detector.** Run Local Outlier Factor (LOF) on the selected encodings. Compare average precision against a uniform one-hot baseline.
6. **Structured Explainer.** Leave-one-feature-out (LOFO) on the top anomalies and the top features by MI.
7. **Representation Geometry Analysis.** Train two identical MLPs, one per encoding regime, and measure first-hidden-layer structure.
8. **Streamlit App.** Adjust the objective weights and view explanations.

## MILP objective

Minimize `α·loss + β·(dim/D_max) + γ·(1−xplain)`.

- `D_max` is the sum, over features, of each feature's largest encoded dimension. It is a fixed constant, not a constraint.
- Without dividing by `D_max`, the dimension term can reach the hundreds and dominate terms that live between 0 and 1.
- Dividing by `D_max` does not bring all three terms to a common range. `dim/D_max` has a nonzero minimum. `loss` and `xplain` are bounded only if the implementation bounds them. Their realized spreads can differ, so one weight can dominate regardless of its value. Normalize each term by its attainable range (min-max) before applying weights (Chen and Li 2023).
- The model is a multiple-choice knapsack: one item per feature, a total-dimension budget, a linear cost per choice. It is infeasible if the budget is below the sum of the cheapest encoding per feature.
- The weighted-sum method cannot reach points on non-convex parts of the Pareto front (Emmerich and Deutz 2018). Choices here are discrete, so the front may be non-convex. Report this limit with results.
- Per-feature `loss` is treated as a constant for each feature and encoding pair. LOF scores all features jointly, so this is an approximation. The comparison with uniform one-hot is the check.
- HiGHS solves the integer model directly. LP relaxation is a bound, not a solution method here. Rounding a fractional solution can break the budget.

## Interpretability rubric

The `xplain` scores are author-assigned. No source supports the numbers. Run a sensitivity sweep over `α`, `β`, `γ` and over the scores.

| Encoding | `xplain` |
|----------|----------|
| Passthrough continuous | 0.90 |
| Ordinal, order confirmed in `config/ordinal_features.yaml` | 0.70 |
| Ordinal, order unconfirmed | 0.30 |

Target encoding is excluded from the candidates. It is computed from `isFraud`, so its output would carry label information into LOF. That would contradict the rule that the detector never sees the label. Scores for other encodings are not yet assigned.

## Evaluation

- LOF is unsupervised. The label is used only to score the result.
- No step before scoring may use `isFraud`. This excludes target encoding, label-driven feature selection, and tuning on label-based scores.
- Report average precision (`sklearn.metrics.average_precision_score`), not the trapezoidal area, and say which.
- The random-classifier baseline equals the fraud rate, about 0.035. Report lift over 0.035 beside every score.
- Precision and recall do not use true negatives, so the large negative class does not inflate the score (Saito and Rehmsmeier 2015).
- `negative_outlier_factor_` is higher for inliers. Negate it before scoring.
- A score near the baseline may mean fraud is not a density outlier. It does not show that an encoding is poor.
- LOF with one-hot data is not covered by any source found. One-hot data produces duplicate points, which Breunig's definition assumes away, and distance contrast shrinks as dimensionality grows.

## Explainer

LOFO removes one feature, reruns the detector, and records the change in an anomaly's score. It follows the idea of leave-one-covariate-out (Lei et al. 2018), which refits without a variable, and of occlusion sensitivity (Zeiler and Fergus 2014), which masks input without refitting. Both are supervised settings, so the anomaly-score change is a project adaptation. No anomaly-specific LOFO precedent was confirmed. One-at-a-time removal can miss features that are jointly anomalous. LIME fits a local surrogate on perturbed samples and is a different method.

Open: whether the unit removed is an original feature or one dummy column.

## Representation geometry

Two MLPs, `MLP-MILP` and `MLP-OHE`, share architecture, seed, and hyperparameters. Only the inputs differ. Measurements use frozen first-hidden-layer activations and weights.

`W` is the first-layer weight matrix with shape (hidden, inputs). `WᵀW` is then inputs by inputs. Its diagonal is the squared norm of each input's direction. Its off-diagonal entries are overlaps between inputs.

1. **Weight-space overlap.** Off-diagonal Frobenius norm of `WᵀW`. The raw norm scales with the weights, so compare a normalized form. `WᵀW` is unchanged by a rotation of the neuron basis, so it cannot show alignment to single neurons. The levels of one categorical variable never occur together, so their overlap costs nothing under the toy-model logic.
2. **Capacity (Scherlis et al. 2022).** `C_i = (W_i·W_i)² / Σ_j (W_i·W_j)²` for each input column `i`, from weights only.
3. **Segment variance share.** The project's own measure. It is not Scherlis's capacity. The first layer is linear before the ReLU, so the share is defined on pre-activations. For neuron `j` and segment `s`, let `v_js = Var(Σ_{i in s} W_ji · x_i)`. The share is `v_js / Σ_s' v_js'`. Shares sum to 1 per neuron by construction. Covariance between segments is ignored. Report the ignored part as `1 − Σ_s v_js / Var(pre-activation_j)`, which can be negative.
4. **Per-segment linear probe.** For each segment, take its highest-MI raw column and split it at the median. Train a linear classifier to predict that bit from first-layer activations. Report accuracy against the 0.5 chance level. The target comes from raw data, so it is identical for `MLP-MILP` and `MLP-OHE`. One column stands in for its segment. Probe accuracy measures linear decodability. It fits both superposed and monosemantic codes. Neither superposition paper uses probes.

The toy-model results assume sparse, independent features and tied weights. Elhage et al. write that it is "very unclear what to generalize to real networks". Input sparsity and column correlation differ between OHE and MILP encodings and can confound the comparison. Report both beside every geometry result.

## Post-project extension: Shapley variance decomposition

The variance share ignores covariance between segments. A Shapley-style decomposition splits that covariance fairly. Treat each segment's input block as a player and the variance of the neuron's pre-activation as the value of a coalition. A segment's share is its average marginal contribution over all orders of adding segments. With five segments there are 120 orders. This is planned after the main project and is not required for it. It has not been designed or checked against a source.

## References

Evidence labels: **Read** means the full text was read. **Metadata** means a bibliographic record only. Details are in the audit report kept outside version control.

**Superposition and capacity**
- Elhage, N., et al. (2022). Toy Models of Superposition. Transformer Circuits Thread. https://transformer-circuits.pub/2022/toy_model/index.html. Also arXiv:2209.10652. Read.
- Scherlis, A., Sachan, K., Jermyn, A. S., Benton, J., Shlegeris, B. (2022). Polysemanticity and Capacity in Neural Networks. arXiv:2210.01892. https://arxiv.org/abs/2210.01892. Read.

**Anomaly detection and explanation**
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. Proc. ACM SIGMOD, pp. 93-104. doi:10.1145/335191.335388. Read.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., Wasserman, L. (2018). Distribution-Free Predictive Inference for Regression. JASA 113(523):1094-1111. arXiv:1604.04173. Read.
- Zeiler, M. D., Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. ECCV 2014, pp. 818-833. arXiv:1311.2901. Metadata. The occlusion wording came through a summarizing tool. Check Section 4.2 before quoting.
- Ribeiro, M. T., Singh, S., Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD 2016, pp. 1135-1144. arXiv:1602.04938. Read. Cited as a contrast: a sampled local surrogate, not a feature-removal method.
- Li, Z., Fan, R. (2025, rev. 2026). Explainable Heterogeneous Anomaly Detection in Financial Networks via Adaptive Expert Routing. arXiv:2510.17088. Read. Cited as a contrast: it builds explanation into the model, where this project explains after the fact.

**Encoding and representation**
- Potdar, K., et al. (2017). A Comparative Study of Categorical Variable Encoding Techniques for Neural Network Classifiers. International Journal of Computer Applications 175(4):7-9. doi:10.5120/ijca2017915495. Read. Seven encoders, one small dataset, one split. It supports the point that encoder choice moved accuracy there, and that ordinal coding implies an order that may not exist.
- Zhu, W., Qiu, R., Fu, Y. (2024). Comparative Study on the Performance of Categorical Variable Encoders in Classification and Regression Tasks. arXiv:2401.09682. Read.
- Plate, T. Distributed representations. In L. Nadel (Ed.), Encyclopedia of Cognitive Science. Wiley. doi:10.1002/0470018860.s00059. Metadata. Online 2006 per Crossref. Sources give 2002, 2003, and 2005 for the set, so no print year is cited.

**Optimization**
- Emmerich, M. T. M., Deutz, A. H. (2018). A tutorial on multiobjective optimization: fundamentals and evolutionary methods. Natural Computing 17(3):585-609. doi:10.1007/s11047-018-9685-y. Read.
- Marler, R. T., Arora, J. S. (2010). The weighted sum method for multi-objective optimization: new insights. Structural and Multidisciplinary Optimization 41(6):853-862. doi:10.1007/s00158-009-0460-7. Metadata.
- Das, I., Dennis, J. E. (1997). A closer look at drawbacks of minimizing weighted sums of objectives for Pareto set generation in multicriteria optimization problems. Structural Optimization 14(1):63-69. doi:10.1007/BF01197559. Metadata.
- Chen, T., Li, M. (2023). The weights can be harmful: Pareto search versus weighted search in multi-objective search-based software engineering. ACM TOSEM 32(1). doi:10.1145/3514233. Read as the arXiv preprint 2202.03728.
- Bednarczuk, E. M., Kaliszewski, I., Miroforidis, J. (2024). Approximate solutions to the multiple-choice knapsack problem by multiobjectivization and Chebyshev scalarization. Operations Research and Decisions 34(4). arXiv:2311.08839. Read as the preprint.
- Kellerer, H., Pferschy, U., Pisinger, D. (2004). Knapsack Problems. Springer. doi:10.1007/978-3-540-24777-7. Metadata.

**Evaluation and data**
- Saito, T., Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE 10(3):e0118432. The abstract and baseline sentences were quoted from the PLOS page through a summarizing tool.
- IEEE-CIS Fraud Detection, Kaggle. https://www.kaggle.com/competitions/ieee-fraud-detection. The page could not be read by the audit. Dataset facts come from secondary copies of the host text.

## Unverified

- The rubric values are author judgment.
- The identity table shape and coverage have not been measured. Run `pd.read_csv(...).shape` after download.
- The Kaggle CLI 2.2.5 accepts both the `-c` and positional forms. The download itself was not run, because no Kaggle credentials were available.
- The Kaggle rules, licence, and redistribution terms were not read.
- Whether LOF behaves usefully on one-hot or mixed data is untested.
