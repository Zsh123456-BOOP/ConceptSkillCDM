"""Runnable Graph-IRT configurations used by dataset launchers.

These are starting configurations inherited from the pre-refactor experiments;
they are intentionally not called "best" until the clean architecture is tuned.
"""

from __future__ import annotations

from typing import Any, Dict

from src.config import DATASET_DEFAULTS


EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    dataset: dict(config)
    for dataset, config in DATASET_DEFAULTS.items()
}

# Keep the plateau scheduler ahead of early stopping so the learning rate can
# decay at least once on the large public benchmarks.
for _dataset in ("frcsub", "math2", "assist_12", "assist_15"):
    EXPERIMENT_CONFIGS[_dataset].update(epochs=45, early_stop_patience=10, patience=4)

DEFAULT_SEEDS = [42]
