import json
from pathlib import Path


def save_to_json(assignments: dict[str, str], path: Path | str) -> None:
    """Write assignments to path as JSON (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(assignments, f, indent=2, sort_keys=True)


def load_from_json(path: Path | str) -> dict[str, str]:
    """Inverse of save_to_json."""
    with open(path) as f:
        return json.load(f)
