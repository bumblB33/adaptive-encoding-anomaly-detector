# Student project: does encoding change what a fraud detector finds?

Learners run, change, and build the pipeline in `docs/design.md`. The project runs 6 weeks, about 2-3 hours per week. The plan assumes that session length. It was not specified.

## Who this is for

A learner who writes basic Python: loops, functions, and `pandas` basics such as reading a CSV and selecting columns. No machine learning background is needed. No linear algebra is needed before Week 6. Week 6 teaches the one idea it uses.

## What you leave with

- A working pipeline that profiles features, picks an encoding for each, flags unusual transactions, and explains the flags.
- A written decision memo that says which encoding choice helped, by how much, and how sure you are.
- The habit of reporting a result next to its baseline.

## How each week works

1. **Predict.** Write one sentence saying what you expect, before you run anything.
2. **Do.** Run, change, or build something. You see a result in the first 20 minutes.
3. **Explain.** Read the short concept note. Compare it with your prediction.
4. **Show.** Produce one visible result: a table, a chart, or a number with its baseline.

A wrong prediction is a good outcome. Record it. The memo uses these records.

## Before Week 1

- Create a Kaggle account and accept the competition rules at https://www.kaggle.com/competitions/ieee-fraud-detection/rules.
- Do not share the data files. Each learner downloads their own copy.
- Follow the Setup section of `README.md`.
- All work uses a stratified 10% sample, which keeps the fraud rate near 3.5%. Time a run on 5,000 rows before scaling up. Runtime on the full 590,540 rows has not been measured.

## Week 1: Meet the data (Stage 1, Feature Profiler)

**Predict.** Which three columns say the most about fraud?

**Do.** Load the sample and run `profile_dataframe`. Sort columns by missing rate, then by cardinality, then by mutual information (MI) with `isFraud`.

**Explain.** Cardinality is how many distinct values a column has. Mutual information measures how much a column reduces uncertainty about the label. The profiler uses the label here only to describe columns. The detector never trains on it.

**Show.** A table of 5 surprising columns, each with a one-line reason.

**Trap.** `TransactionDT` is a time offset, not a date. MI estimates for columns with many categories can mislead. Say so in the table notes.

## Week 2: Group and encode (Stage 2, Segmenter; start of Stage 3)

**Predict.** Which segment will hold the most columns?

**Do.** Run `segment_features`. Count columns per segment. Find one column you think sits in the wrong segment and say why. Then encode three columns three ways: passthrough, one-hot, and ordinal using `config/ordinal_features.yaml`. Record how many columns each choice creates.

**Explain.** One-hot makes one 0/1 column per category and adds no false order. Ordinal maps categories to numbers, which invents an order unless the categories really have one. Dimension is the count of encoded columns.

**Show.** A table with one row per column and encoding: name, encoding, dimension.

## Week 3: Find the unusual ones (Stage 5 baseline; Stage 3 loss)

**Predict.** If you flag the top 3.5% by Local Outlier Factor (LOF) score, what share of flags are fraud?

**Do.** Build uniform one-hot features for the sample. Run LOF. Score with average precision against `isFraud`. Then compute the score a random guess gets, which is the fraud rate, about 0.035. Report your score divided by 0.035.

**Explain.** LOF scores a point by how much less dense its neighborhood is than its neighbors' neighborhoods. It does not see the label. Average precision uses precision and recall, so the many non-fraud rows do not inflate it.

**Show.** One number with its baseline and lift.

**Traps.** scikit-learn's `negative_outlier_factor_` is higher for normal points. Negate it before scoring. A score near 0.035 does not mean your code is broken. Fraud may not look like a density outlier.

**Do (second half).** For 10 to 20 chosen columns, build the `loss` for each column and encoding by running LOF on that column's encoding alone. Cost grows with every extra run, so time one run first.

## Week 4: Let an optimizer choose (Stage 4, MILP Selector)

**Predict.** Will the chosen encodings beat uniform one-hot?

**Do.**
1. Solve a 4-column toy problem by listing every combination by hand. Find the best one.
2. Solve the same problem with PuLP and HiGHS. Check the answers match.
3. Scale up to your 10 to 20 columns. Change the weights `α`, `β`, `γ` and watch the choices change.
4. Set the dimension budget too low. Read the infeasibility message.

**Explain.** The optimizer picks exactly one encoding per column under a size budget. The objective adds three costs: detection loss, size, and un-interpretability. Normalize each cost by its own range before weighting, or one cost can overwhelm the others.

**Show.** A chart of detection score against weight setting, with the uniform one-hot score as a line.

**Note.** The interpretability scores come from the project author. Change one score and report what moves.

## Week 5: Explain and present (Stage 6, Explainer; Stage 8, Streamlit)

**Predict.** Which column explains the most flagged transactions?

**Do.** Pick the top 20 flagged transactions. For each, remove one column at a time, rerun LOF, and record how far its score falls. The column with the largest drop gets the blame. Then wire the weight sliders into the Streamlit template.

**Explain.** Leave-one-feature-out is a what-if test. It misses columns that matter only together.

**Show.** A table of transaction, top blamed column, and score change. A running app with sliders.

## Week 6: Look inside the network (Stage 7, Representation Geometry)

**Predict.** Will two networks that differ only in encoding organize their first layer differently?

**Do.** Train `MLP-MILP` and `MLP-OHE` with the same seed, architecture, and settings. Compute four measures on the first layer:

1. Weight overlap: how much input columns share neurons.
2. Capacity per input column, from weights.
3. Segment variance share: how much of each neuron's variation comes from each segment.
4. Probe accuracy: can a simple classifier read a segment's key column from the activations?

**Explain.** A network can pack more ideas than it has neurons by letting ideas share directions. Picture a filing cabinet with more folders than drawers. Folders sharing a drawer get mixed up when needed together.

**Show.** A table of the four measures for both networks.

**Traps.** The theory behind these measures was developed on small toy networks with sparse, independent features. Whether it carries over to this data is unknown. The two encodings also differ in input sparsity, which can change the numbers by itself. A tie between the networks is a valid, interesting result.

## Final deliverable: the decision memo

One page:
- The question, in your words.
- Your best encoding choice and its score, with the 0.035 baseline and the lift.
- Two predictions you got wrong and what you learned.
- One thing you could not conclude, and why.

## Instructor notes

**Status.** Only Stages 1 and 2 exist as code. Stages 3 to 8 must be written as scaffolds, with function stubs, small tests, and a reference solution for each. No reference solutions exist yet. Eight stages in six weeks is a heavy load for this learner, so Weeks 3 to 6 rely on scaffolds.

**Unresolved.**
- The Kaggle rules and data licence were not read. Read them before distributing anything.
- Runtime of LOF on one-hot features has not been measured, on the sample or the full set.
- Identity table shape and coverage have not been measured.
- Whether LOF works usefully on one-hot data is untested.
- The teaching approach in this plan is a design choice. No pedagogy literature was researched for it.

**Extension after the project.** The Shapley-style variance decomposition in `docs/design.md`.
