"""Smoke checks for the auditable optimizer and robust early-stop controls."""

from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main as main_module
from src.experiment_utils import _config_hash
from src.dataset import create_dataloaders
from src.trainer import (
    _build_data_identity,
    _build_optimizer,
    _claim_test_seal,
    _is_validation_improvement,
    _validate_checkpoint_data_identity,
    run_inference,
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
    assert args.run_mode == "train"
    assert main_module.parse_args(["--run_mode", "train"]).run_mode == "train"
    assert main_module.parse_args(["--run_mode", "test"]).run_mode == "test"
    assert "max_test_batches" not in {
        action.dest for action in main_module.build_parser()._actions
    }

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
        enable_response_graph=True,
    )
    baseline_hash = _config_hash(hash_args)
    hash_args.epochs = 100
    assert _config_hash(hash_args) != baseline_hash
    hash_args.epochs = 120
    hash_args.min_stu_interactions = 0
    assert _config_hash(hash_args) != baseline_hash
    hash_args.min_stu_interactions = 15
    hash_args.enable_response_graph = False
    assert _config_hash(hash_args) != baseline_hash

    class _Logger:
        def info(self, *args, **kwargs):
            del args, kwargs

    with tempfile.TemporaryDirectory() as directory:
        train_path = os.path.join(directory, "train.csv")
        valid_path = os.path.join(directory, "valid.csv")
        frame = pd.DataFrame(
            {
                "stu_id": [1, 1, 2, 2],
                "exer_id": [10, 11, 10, 11],
                "cpt_seq": ["5", "6", "5", "6"],
                "label": [0.0, 1.0, 1.0, 0.0],
            }
        )
        frame.to_csv(train_path, index=False)
        frame.to_csv(valid_path, index=False)
        _, _, test_loader, info = create_dataloaders(
            train_file=train_path,
            val_file=valid_path,
            test_file=os.path.join(directory, "must_not_be_opened.csv"),
            batch_size=2,
            num_workers=0,
            min_stu_interactions=0,
            min_exer_interactions=0,
            load_test=False,
        )
        assert test_loader is None
        assert info["test_sealed_during_training"] is True
        assert "test_size" not in info and "test_seen_rows" not in info

        identity_args = SimpleNamespace(dataset_name="assist_09", data_dir=directory)
        identity = _build_data_identity(identity_args, train_path, valid_path)
        bound_dir, checked = _validate_checkpoint_data_identity(
            SimpleNamespace(explicit_arg_dests=[]),
            {"dataset_name": "assist_09", "data_dir": directory},
            {"data_identity": identity},
        )
        assert bound_dir == os.path.realpath(directory)
        assert checked == identity
        try:
            _validate_checkpoint_data_identity(
                SimpleNamespace(
                    explicit_arg_dests=["dataset_name"],
                    dataset_name="junyi",
                ),
                {"dataset_name": "assist_09", "data_dir": directory},
                {"data_identity": identity},
            )
        except RuntimeError as exc:
            assert "conflicts with checkpoint" in str(exc)
        else:
            raise AssertionError("explicitly conflicting dataset identity must be rejected")

    with tempfile.TemporaryDirectory() as directory:
        seal_path = os.path.join(directory, "test_seal.json")
        _claim_test_seal(seal_path, {"checkpoint_sha256": "abc"})
        try:
            _claim_test_seal(seal_path, {"checkpoint_sha256": "abc"})
        except RuntimeError as exc:
            assert "seal already exists" in str(exc)
        else:
            raise AssertionError("test seal claim must be exclusive")

    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "test_results.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        try:
            run_inference(
                SimpleNamespace(no_cuda=True, save_dir=directory),
                _Logger(),
            )
        except RuntimeError as exc:
            assert "evaluate test twice" in str(exc)
        else:
            raise AssertionError("sealed checkpoint must reject a second test evaluation")
    print("OK: robust training protocol checks passed.")


if __name__ == "__main__":
    main()
