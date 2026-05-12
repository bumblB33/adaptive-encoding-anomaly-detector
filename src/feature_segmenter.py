"""Module 2: assign every feature to one of five domain segments.

The segment vocabulary is fixed:

- ``"transaction amount"``
- ``"identity/device"``
- ``"behavioral frequency"``
- ``"temporal/timing"``
- ``"card/account"``

Module 7's per-segment metrics (interference matrix, linear-probe
accuracy, capacity) bind to this exact vocabulary, so cluster IDs are
never user-facing — every column lands on one of the five strings.

The pipeline is the hybrid the plan calls for:

1. **Rule pre-pass.** IEEE-CIS column names follow well-known
   conventions (``TransactionAmt``, ``id_*``, ``Device*``,
   ``*emaildomain``, ``C*``, ``D*``/``TransactionDT``, ``card*``,
   ``addr*``, ``M*``, ``ProductCD``). Anything that matches a pattern
   gets mapped directly to its segment.
2. **K-means fallback.** Residual columns (the anonymized ``V*``
   engineered features and any future unknowns) are vectorized from
   the Module 1 profile (log-cardinality, missing rate, mutual
   information, an ``is_categorical`` indicator), standardized using
   rule-mapped columns as the basis, and clustered with k-means
   (``k = min(5, n_residuals)``). Each cluster centroid is then mapped
   to the segment whose centroid (computed from rule-mapped columns)
   is nearest in standardized profile-vector space. The result is the
   segment string, not the cluster id.

The label column (``isFraud`` by default) is excluded from segment
assignment — it is a target, not a feature, and Modules 5 and 7
must not see it as part of any segment.

See docs/portfolio_project_plan.md "Module 2" for the spec and
docs/implementation-plan.md "Module 2 — feature segmentation" for
the acceptance criterion.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config.defaults import RANDOM_SEED

SEGMENT_LABELS: tuple[str, ...] = (
    "transaction amount",
    "identity/device",
    "behavioral frequency",
    "temporal/timing",
    "card/account",
)

_RULE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^TransactionAmt$"), "transaction amount"),
    (re.compile(r"^dist\d+$"), "transaction amount"),
    (re.compile(r"^id_\d+$"), "identity/device"),
    (re.compile(r"^Device.*$"), "identity/device"),
    (re.compile(r".+emaildomain$"), "identity/device"),
    (re.compile(r"^C\d+$"), "behavioral frequency"),
    (re.compile(r"^TransactionDT$"), "temporal/timing"),
    (re.compile(r"^D\d+$"), "temporal/timing"),
    (re.compile(r"^card\d+$"), "card/account"),
    (re.compile(r"^addr\d+$"), "card/account"),
    (re.compile(r"^M\d+$"), "card/account"),
    (re.compile(r"^ProductCD$"), "card/account"),
)


def _apply_rules(column: str) -> str | None:
    for pattern, label in _RULE_PATTERNS:
        if pattern.match(column):
            return label
    return None


def _profile_vector(entry: dict[str, Any]) -> list[float]:
    mi = entry.get("mutual_information")
    return [
        float(np.log1p(float(entry["cardinality"]))),
        float(entry["missing_rate"]),
        float(mi) if mi is not None else 0.0,
        1.0 if entry["detected_type"] == "categorical" else 0.0,
    ]


def segment_features(
    profile: dict[str, dict[str, Any]],
    label_column: str = "isFraud",
    random_state: int = RANDOM_SEED,
) -> dict[str, str]:
    """Map every feature column in ``profile`` to a segment label.

    Parameters
    ----------
    profile:
        Output of ``feature_profiler.profile_dataframe`` (keyed by
        column name; each entry has ``detected_type``, ``cardinality``,
        ``missing_rate``, ``mutual_information``, ...).
    label_column:
        Column name to exclude from segmentation. Defaults to
        ``"isFraud"``. Absent keys are tolerated.
    random_state:
        Seed for k-means; defaults to ``config.defaults.RANDOM_SEED``.

    Returns
    -------
    dict[str, str]
        Column name → one of ``SEGMENT_LABELS``. Suitable for
        JSON-serializing to ``outputs/segment_assignments.json``.
    """
    feature_cols = [c for c in profile.keys() if c != label_column]

    assignments: dict[str, str] = {}
    residuals: list[str] = []
    for col in feature_cols:
        rule_label = _apply_rules(col)
        if rule_label is not None:
            assignments[col] = rule_label
        else:
            residuals.append(col)

    if not residuals:
        return assignments

    rule_cols = [c for c in feature_cols if c in assignments]
    if not rule_cols:
        # Pathological: no IEEE-CIS-style anchors to learn segment
        # geometry from. Drop residuals into "card/account" — the most
        # heterogeneous of the five buckets — so the contract still
        # holds. Realistically this branch should never fire.
        for col in residuals:
            assignments[col] = SEGMENT_LABELS[-1]
        return assignments

    rule_vectors = np.array([_profile_vector(profile[c]) for c in rule_cols])
    residual_vectors = np.array([_profile_vector(profile[c]) for c in residuals])

    scaler = StandardScaler()
    rule_vectors_scaled = scaler.fit_transform(rule_vectors)
    residual_vectors_scaled = scaler.transform(residual_vectors)

    n_clusters = min(len(SEGMENT_LABELS), len(residuals))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    cluster_ids = kmeans.fit_predict(residual_vectors_scaled)

    segment_centroids: dict[str, np.ndarray] = {}
    for label in SEGMENT_LABELS:
        members = [
            rule_vectors_scaled[i]
            for i, c in enumerate(rule_cols)
            if assignments[c] == label
        ]
        if members:
            segment_centroids[label] = np.mean(members, axis=0)

    cluster_to_segment: dict[int, str] = {}
    for cid in range(n_clusters):
        cluster_centroid = kmeans.cluster_centers_[cid]
        cluster_to_segment[cid] = min(
            segment_centroids,
            key=lambda label: float(
                np.linalg.norm(cluster_centroid - segment_centroids[label])
            ),
        )

    for col, cid in zip(residuals, cluster_ids):
        assignments[col] = cluster_to_segment[int(cid)]

    return assignments


def save_segments(assignments: dict[str, str], path: Path | str) -> None:
    """Persist assignments as JSON. Parent directories are created as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(assignments, f, indent=2, sort_keys=True)


def load_segments(path: Path | str) -> dict[str, str]:
    """Inverse of ``save_segments``."""
    with open(path) as f:
        return json.load(f)
