"""Validate the structure of config/ordinal_features.yaml.

Schema: top-level mapping from feature name (str) to a non-empty list
of ordered values (lowest to highest). An empty file or empty mapping
is acceptable during Week 1 — annotations are populated during EDA in
notebooks/01_eda.ipynb (deferred to a follow-up plan).
"""

from pathlib import Path

import yaml

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "ordinal_features.yaml"
)


def _load():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_yaml_file_exists():
    assert CONFIG_PATH.exists(), f"missing config file: {CONFIG_PATH}"


def test_yaml_parses_to_mapping_or_empty():
    data = _load()
    assert data is None or isinstance(data, dict)


def test_each_feature_maps_to_nonempty_list():
    data = _load() or {}
    for feature_name, value_sequence in data.items():
        assert isinstance(feature_name, str), (
            f"feature name must be a string, got {type(feature_name).__name__}"
        )
        assert isinstance(value_sequence, list), (
            f"{feature_name}: ordinal feature must map to a list of values"
        )
        assert len(value_sequence) > 0, (
            f"{feature_name}: ordinal value sequence cannot be empty"
        )
