"""Smoke checks for the auditable optimizer and robust early-stop controls."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main as main_module
from src.experiment_utils import _config_hash
from src.trainer import (
    _build_optimizer,
    _is_validation_improvement,
    _should_early_stop,
)


def main() -> None:
    args = main_module.parse_args(
        [
            "--optimizer",
            "adamw",
            "--epochs",
            "120",
            "--min_epochs",
            "20",
            "--patience",
            "4",
            "--early_stop_patience",
            "15",
            "--early_stop_min_delta",
            "0.00001",
        ]
    )
    main_module._validate_args(args)

    model = torch.nn.Linear(3, 1)
    optimizer = _build_optimizer(model, args)
    assert isinstance(optimizer, torch.optim.AdamW)

    stop_args = SimpleNamespace(min_epochs=20, early_stop_patience=15)
    assert not _should_early_stop(19, 15, stop_args)
    assert not _should_early_stop(20, 14, stop_args)
    assert _should_early_stop(20, 15, stop_args)

    assert not _is_validation_improvement(0.700005, 0.700000, 1e-5)
    assert _is_validation_improvement(0.70002, 0.700000, 1e-5)

    hash_args = SimpleNamespace(
        dataset_name="assist_09",
        epochs=120,
        min_stu_interactions=15,
        min_exer_interactions=0,
    )
    baseline_hash = _config_hash(hash_args)
    hash_args.epochs = 100
    assert _config_hash(hash_args) != baseline_hash
    hash_args.epochs = 120
    hash_args.min_stu_interactions = 0
    assert _config_hash(hash_args) != baseline_hash
    print("OK: robust training protocol checks passed.")


if __name__ == "__main__":
    main()
