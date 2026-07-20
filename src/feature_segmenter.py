import re
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
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
    """[log1p(cardinality), missing_rate, mutual_info, is_categorical]."""
    mutual_info = entry.get("mutual_info")
    return [
        float(np.log1p(float(entry["cardinality"]))),
        float(entry["missing_rate"]),
        float(mutual_info) if mutual_info is not None else 0.0,
        1.0 if entry["detected_type"] == "categorical" else 0.0,
    ]


def segment_features(
    profile: dict[str, dict[str, Any]],
    label_column: str = "isFraud",
    random_state: int = RANDOM_SEED,
) -> dict[str, str]:
    """
    Unassigned (residual) columns are k-means clustered on their profile vector
    (standardized against mapped columns) and mapped to the nearest rule-segment centroid.
    The label_column is excluded.
    If no column matches a rule, residuals fall back to the last label.
    """
    feature_cols = [col for col in profile.keys() if col != label_column]

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

    rule_cols = [col for col in feature_cols if col in assignments]
    if not rule_cols:
        for col in residuals:
            assignments[col] = SEGMENT_LABELS[-1]
        return assignments

    rule_vectors = np.array([_profile_vector(profile[col]) for col in rule_cols])
    residual_vectors = np.array([_profile_vector(profile[col]) for col in residuals])

    scaler = StandardScaler()
    rule_vectors_scaled = scaler.fit_transform(rule_vectors)
    residual_vectors_scaled = scaler.transform(residual_vectors)

    n_clusters = min(len(SEGMENT_LABELS), len(residuals))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    cluster_ids = kmeans.fit_predict(residual_vectors_scaled)

    segment_labels = []
    segment_centroids = []
    for label in SEGMENT_LABELS:
        members = rule_vectors_scaled[[assignments[col] == label for col in rule_cols]]
        if len(members):
            segment_labels.append(label)
            segment_centroids.append(members.mean(axis=0))

    nearest = pairwise_distances_argmin(
        kmeans.cluster_centers_, np.array(segment_centroids)
    )
    cluster_to_segment = {
        cluster: segment_labels[seg] for cluster, seg in enumerate(nearest)
    }

    for col, cluster_id in zip(residuals, cluster_ids):
        assignments[col] = cluster_to_segment[int(cluster_id)]

    return assignments
