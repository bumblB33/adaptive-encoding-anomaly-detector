# Design rationale

Small notes on the non-obvious design decisions in this project. Each one captures
*why* a decision was made, not *what* the decision is. `portfolio_project_plan.md`
and `docs/implementation-plan.md` cover the what. Use this doc when you
need to defend or revisit a choice.

---

## MILP objective normalization: why all three terms live in [0, 1]

The MILP objective is `α·loss + β·(dim/D_max) + γ·(1 − xplain)`. Every
term is normalized to [0, 1] *before* the weights `(α, β, γ)` are applied.

Without normalization, the dimensionality term dominates by accident.
Raw post-encoding dimensions can reach the high tens or low hundreds,
while `loss` and `(1 − xplain)` live between 0 and 1 by construction. If `dim`
is unnormalized, β behaves as if it were 50 to 100 times whatever you set it to.
The solver effectively optimizes only the dimensionality term, and α
and γ become noise.

**Why this matters for the user-facing interface:** without normalization,
the weights `(α, β, γ)` are *not* priority statements. They are
artifacts of the different scales of the three objectives. Telling a
reviewer "the model weights loss at 40% and interpretability at 30%"
is a lie unless the three terms are on a common scale first.
Normalization is what makes the weights mean what they appear to mean.

`D_max` is the normalization anchor for the dim term. It is the sum of
the maximum-dimensionality encoding per feature, computed analytically
from `evaluation_matrix.csv` *before* the solver runs. It is not a
constraint the solver enforces. It is a precomputed constant the
objective uses to put `dim/D_max` in [0, 1].

---

## Superposition: the filing-cabinet analogy

Useful intuition for explaining what the three Module 7 metrics measure.

Picture two drawers (neurons) and fifty folders (features). You pack multiple
folders into each drawer because you do not have fifty drawers. That works
fine as long as you rarely need two folders from the same drawer at
once. When you do, the contents are entangled. You cannot cleanly pull
out just one. You lose the ability to say "drawer 1 is about X."

That is the interpretability cost of superposition. Each of the three
Module 7 metrics asks the same question from a different angle:

- **Interference matrix** asks how much the folder *directions* overlap
  in weight space. Off-diagonal Frobenius mass is high when folders
  share drawers.
- **Linear probes** ask whether you can recover "is this drawer holding
  any folder of type X?" from the activations alone. High per-segment
  accuracy means the drawer is structured enough to be read.
- **Capacity** asks how cleanly any one drawer corresponds to one
  folder. A capacity distribution skewed toward 1.0 per neuron means
  each drawer holds (mostly) one folder.

The three metrics triangulate the same underlying property, namely how
much superposition the network is running, from weight geometry,
activation decodability, and neuron specialization. Any one metric in
isolation is suggestive; all three together are convergent evidence.
