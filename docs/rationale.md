# Design Overview

Small notes on design decisions in this project.

---

## Superposition: the filing-cabinet analogy

We use three representation-geometry metrics to measure the same
underlying property, how much superposition the network is running, from
three angles: weight geometry, activation decodability, and neuron
specialization.

Picture two drawers (neurons) and fifty folders (features). You pack
multiple folders into each drawer because you do not have fifty drawers.
That works fine as long as you rarely need two folders from the same
drawer at once. When you do, the contents are entangled: you cannot
cleanly pull out just one, and you lose the ability to say "drawer 1 is
about X."

That entanglement is the interpretability cost of superposition, and
each of the three metrics probes it from a different angle.

- **Interference matrix** asks how much the folder *directions* overlap in weight space. Its off-diagonal Frobenius mass is the combined size of the entries that pair *different* features (where "Frobenius" just means the square root of the summed squares of those entries); that mass is high when the same folders share drawers, i.e. when the same neurons are used to encode multiple features.
- **Linear probes** ask whether you can recover "is this drawer holding any folder of type X?" from the activations alone. High per-segment accuracy means the drawer is structured enough to be read.
- **Capacity** asks how cleanly any one drawer corresponds to one folder. A capacity distribution skewed toward 1.0 per neuron means the majority of the drawer's contents are dedicated (mostly) to one folder.

## MILP objective normalization: why all three terms live in [0, 1]

The MILP objective is `α·loss + β·(dim/D_max) + γ·(1 − xplain)`. Reading
the notation:

- `loss`: how poorly anomalies are detected under an encoding (lower is
  better).
- `dim`: the post-encoding dimensionality (how many columns the encoding
  produces); `D_max` is the largest that total could be, so `dim/D_max`
  is encoded size on a 0–1 scale.
- `xplain`: the interpretability rubric score, in [0, 1] (higher is more
  human-readable); `1 − xplain` is therefore the *un*-interpretability
  cost.
- `α, β, γ`: the weights the user sets to trade off the three terms.
- `·` is multiplication throughout.

Every term is normalized to [0, 1] before the weights `(α, β, γ)` are
applied, and the rest of this note is about why that matters.

Without normalization, the dimensionality term dominates. Raw
post-encoding dimensions can reach the high tens or low hundreds, while
`loss` and `(1 − xplain)` live between 0 and 1 by construction, so an
unnormalized `dim` makes β behave as if it were 50 to 100 times whatever
you set it to. The solver then optimizes only the dimensionality term,
and α and γ become noise.

Normalization also fixes what the weights *mean*. Without it, `(α, β, γ)`
are not priority statements at all; they are artifacts of the three
objectives' different scales, so claiming "the model weights loss at 40%
and interpretability at 30%" is misleading. Putting the three terms on a
common scale first is what makes the weights mean what they appear to.

`D_max` is the normalization anchor for the dim term: the sum of the
maximum-dimensionality encoding per feature, computed analytically from
`evaluation_matrix.csv` *before* the solver runs. It is a precomputed
constant the objective uses to put `dim/D_max` in [0, 1]; it is not a
constraint the solver enforces. The solver is free to find solutions
with `dim > D_max` if that helps the overall objective; what it cannot do
is "game" the objective by inflating `dim` to drown out the other terms.

---
