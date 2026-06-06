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
    """[log1p(cardinality), missing_rate, mutual_information, is_categorical]."""
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
    """Map each feature column in profile to one of SEGMENT_LABELS.

    Rule pre-pass on IEEE-CIS name conventions; residual columns (e.g. V*) are
    k-means clustered on their profile vector (standardized against the
    rule-mapped columns) and each cluster mapped to the nearest rule-segment
    centroid. label_column is excluded. It's a target, not a feature. If no
    column matches a rule, residuals fall back to the last label. random_state
    seeds k-means.
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
    """Write assignments to path as JSON (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(assignments, f, indent=2, sort_keys=True)


def load_segments(path: Path | str) -> dict[str, str]:
    """Inverse of save_segments."""
    with open(path) as f:
        return json.load(f)
