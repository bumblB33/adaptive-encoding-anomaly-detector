# Adaptive Encoding Anomaly Detector

Production machine learning pipelines may treat feature encoding and data preprocessing as an afterthought. This project asks whether encoding choices upstream of a neural network change what the network can represent. If so, then choosing those encodings deserves the same care we give to model selection. Encoding selection is handled via an optimization algorithm (MILP - Mixed-Integer Linear Program). The resulting decision matrix doubles as an auditable explanation of why each feature was encoded the way it was.

(There's a glossary in docs/glossary.md :) 

## Summary

Research the intersection of constraint-based optimization, anomaly detection, and neural-network interpretability.

1. **Encoding Selection Pipeline.** Feature encoding is treated as an optimization problem using a Mixed-Integer Linear Program (MILP). MILP is an optimization algorithm that chooses among discrete options, one encoding per feature, while respecting linear constraints. In the context of this project, the MILP's objective is to minimize `α·loss + β·(dim/D_max) + γ·(1−xplain)`, a weighted sum of detection error, encoded size, and un-interpretability. The solver is **PuLP + HiGHS** (a Python modeling library + open-source solver). The primary detector on the IEEE-CIS Fraud Detection dataset is Local Outlier Factor (LOF), which flags outliers by identifying data points sitting in unusually sparse neighborhoods. The MILP-selected set of encodings is then compared against a plain one-hot encoded (uniform-OHE) baseline by measuring area under the precision–recall curve (PR-AUC). PR-AUC was chosen as a metric because in the labeled dataset, fraud is rare (roughly 3.5% of transactions). PR-AUC ignores the large number of true-negative samples that another metric might reward. The pipeline also produces structured explanations for individual anomalies by running leave-one-feature-out (LOFO) on the top anomalies and features by mutual information (MI).

2. **Interpretability (representation geometry).** Two identical multilayer perceptrons (MLPs) are used. These are basic feedforward neural networks. `MLP-MILP` and `MLP-OHE` are trained on the two encoding regimes. They share the same architecture, the same random seed, and the same hyperparameters. Only the inputs differ. Activations are frozen for both, and superposition for each is measured three ways. Superposition is a state where a neural network is representing more features than it has neurons. It does so by letting them share overlapping directions at the cost of interference. Interference, broadly speaking, is when no clean separation of concept representations per-neuron can be identified via probe. The three measurements used to evaluate superposition are: 

(1) The interference matrix `Wᵀ·W` (summarized by the off-diagonal Frobenius norm) 
(2) Per-segment linear-probe accuracy, measuring for interference. (3) Per-segment capacity: the fraction of each neuron's activation variance attributable to each segment. MLPs are controlled instruments for studying representation geometry. They are distinct from the anomaly detection.
If the geometry experiment returns a null result, that's also interesting. It may suggest that the network compensates in some way for 'less optimal' encodings, which would warrant further research. 

---

## The MILP Objective

The optimization solver minimizes three factors: anomaly detection loss, encoded dimensionality, and un-interpretability. The objective is defined as `α·loss + β·(dim/D_max) + γ·(1−xplain)`.

`D_max` represents the maximum possible dimensionality if every feature used its most expensive encoding. Dividing by this precomputed constant scales all three terms into the 0 to 1 range.

**Note** Without dividing the dimension by `D_max`, the dimensionality term completely dominates the equation. Post-encoding dimensions can reach into the hundreds, while loss and the interpretability score live between 0 and 1. With dimension unscaled, the solver ignores the other terms and simply tries to shrink the dataset. 

The interpretability score (`xplain`) is assigned by a rubric based on human readability. Passthrough continuous variables score a 0.90, while target encoding drops to 0.35. Features with an explicit, confirmed order receive a 0.70 for ordinal encoding, while unconfirmed ordinal features are penalized with a 0.30 to avoid injecting spurious structure.

## Representation Geometry

To test whether upstream encoding decisions actually alter the geometry of learned representations, two identical multilayer perceptrons are trained. One takes the MILP-selected encodings, and the other takes the one-hot baseline.

The goal is to measure superposition, which can be understood through a filing-cabinet analogy. Consider two drawers (neurons) and fifty folders (features). Multiple folders are packed into each drawer for lack of space. As long as two folders from the same drawer are rarely needed at once, everything works fine. When they are needed at the same time, the contents become entangled, and the ability to say that a specific drawer is dedicated to a specific topic is lost.

That entanglement is the cost of superposition. It's measured three ways using the frozen first-hidden-layer activations:

* **Interference matrix:** Measures how much the 'folder' directions overlap in weight space, via the Frobenius norm of the off-diagonal mass.
* **Linear probes:** Tests whether the segment label can be predicted from the activations alone. High accuracy means the 'drawer' is structured enough to be read cleanly.
* **Capacity:** Computes the fraction of a neuron's activation variance attributable to a specific segment.

## Dataset

IEEE-CIS Fraud Detection from Kaggle. Roughly 590,000 transactions with more than 400 raw features (email domains, device types, card networks, product codes, transaction type codes) and ground-truth fraud labels. True-positive fraud rate represents roughly ~3.5% of transactions. Raw data is not committed to the repository.

## Architecture 

1. **Feature Profiler:** Evaluates every feature to determine its data type, cardinality, missingness, and mutual information with the fraud label. The label is used strictly for profiling. The main anomaly detector never sees it as a target.
2. **Feature Segmenter:** Groups features into five domain labels (transaction amount, identity/device, behavioral frequency, temporal/timing, and card/account). Relies on rule-based logic for known naming conventions and k-means clustering for the rest.
3. **Encoding Candidate Evaluator:** Computes the detection loss, encoded dimensionality, and an interpretability score for every feature and encoding pair.
4. **MILP Encoding Selector:** Optimization solver selects exactly one encoding per feature under a dimensionality budget using PuLP and the HiGHS solver.
5. **Anomaly Detector:** Runs Local Outlier Factor on the MILP-selected features and compares the PR-AUC against a uniform one-hot encoded baseline.
6. **Structured Explainer:** Uses LOFO (leave-one-feature-out) to explain why specific transactions were flagged.
7. **Representation Geometry Analysis:** A separate module to evaluate how the encoding choice impacts neural network superposition.
8. **Streamlit App:** A frontend for interactively adjusting the optimization weights and viewing the associated anomaly explanations.

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

## Stack

Python 3.11>= 
- numpy 
- scipy 
- scikit-learn
- pandas
- joblib
- pyyaml. 
- pytest
- PuLP and HiGHS Solvers (for the MILP)
- PyTorch (for the geometry MLPs)
- Streamlit (for the demo.)

## Sources

### Academic References

- Elhage et al. (2022). Toy Models of Superposition. Transformer Circuits Thread. https://transformer-circuits.pub/2022/toy_model/index.html
- Scherlis et al. (2022). Polysemanticity and Capacity in Neural Networks. arXiv:2210.01892. https://ar5iv.labs.arxiv.org/html/2210.01892
- Li and Fan (2026). Explainable Heterogeneous Anomaly Detection in Financial Networks via Adaptive Expert Routing. arXiv:2510.17088.
- Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers. ACM SIGMOD.
- Potdar et al. (2017). A Comparative Study of Categorical Variable Encoding Techniques for Neural Network Classifiers. International Journal of Computer Applications, 175(4), 7–9.

### Multi-objective optimization

- Weighted Sum Method (ScienceDirect topic page). https://www.sciencedirect.com/topics/computer-science/weighted-sum-method
 
### Weight Adaptation Impact (Interesting, but out of scope.)

https://arxiv.org/html/2605.08137v1
https://arxiv.org/html/2605.08137v1

