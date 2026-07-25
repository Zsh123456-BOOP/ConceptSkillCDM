"""Smoke checks for the auditable optimizer and robust early-stop controls."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main as main_module
from src.config import EMA_DECAY, PAIRWISE_AUC_WEIGHT
from src.experiment_utils import _config_hash
from src.dataset import create_dataloaders
from src.model import GRAPH_IRT_ARCHITECTURE
from src.trainer import (
    _build_data_identity,
    _build_optimizer,
    _checkpoint_args,
    _clone_model_state,
    _claim_test_seal,
    _is_validation_improvement,
    _load_inference_selection_metadata,
    _materialize_checkpoint,
    _prediction_loss,
    _resolve_ema_decay,
    _resolve_pairwise_auc_weight,
    _select_residual_candidate,
    _sha256_file,
    _temporary_model_state,
    _training_evidence_kwargs,
    _update_ema_state,
    _validate_checkpoint_data_identity,
    run_inference,
    _should_early_stop,
)


def _write_selection_manifest(
    directory: str,
    *,
    selected_sha256: str,
    selected_source: str,
    residual_active: bool,
    accepted: bool,
    reason: str,
    deltas,
) -> None:
    manifest = {
        "schema_version": 1,
        "residual_mode": "relation_residual_v4",
        "accepted": accepted,
        "reason": reason,
        "deltas": deltas,
        "selected": {
            "source": selected_source,
            "sha256": selected_sha256,
            "residual_active": residual_active,
        },
    }
    (Path(directory) / "selection_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _assert_inference_selection_metadata_contract() -> None:
    loaded_args = {
        "model_variant": "gec_relation_residual",
        "gec_mode": "relation_residual_v4",
    }
    candidate_deltas = {
        "auc": 0.0002,
        "bce_loss": -0.0001,
        "rmse": -0.00005,
    }

    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        candidate_path = directory_path / "candidate_best.pth"
        selected_path = directory_path / "selected_model.pth"
        best_path = directory_path / "best_model.pth"
        candidate_path.write_bytes(b"candidate-checkpoint")
        selected_path.write_bytes(b"stale-selected-checkpoint")
        best_path.write_bytes(b"stale-best-checkpoint")
        _materialize_checkpoint(str(candidate_path), str(selected_path))
        _materialize_checkpoint(str(candidate_path), str(best_path))
        candidate_sha256 = _sha256_file(str(candidate_path))
        assert _sha256_file(str(selected_path)) == candidate_sha256
        assert _sha256_file(str(best_path)) == candidate_sha256

        _write_selection_manifest(
            directory,
            selected_sha256=candidate_sha256,
            selected_source="candidate",
            residual_active=True,
            accepted=True,
            reason="accepted",
            deltas=candidate_deltas,
        )
        candidate_metadata = _load_inference_selection_metadata(
            directory,
            checkpoint_sha256=_sha256_file(str(best_path)),
            loaded_args=loaded_args,
        )
        assert candidate_metadata == {
            "selection_manifest_present": True,
            "selected_source": "candidate",
            "residual_active": True,
            "selection_reason": "accepted",
            "selection_accepted": True,
            "candidate_parent_deltas": candidate_deltas,
            "effective_model_variant": "gec_relation_residual",
        }

        _write_selection_manifest(
            directory,
            selected_sha256="0" * 64,
            selected_source="candidate",
            residual_active=True,
            accepted=True,
            reason="accepted",
            deltas=candidate_deltas,
        )
        try:
            _load_inference_selection_metadata(
                directory,
                checkpoint_sha256=_sha256_file(str(best_path)),
                loaded_args=loaded_args,
            )
        except RuntimeError as exc:
            assert "selected SHA" in str(exc)
        else:
            raise AssertionError(
                "inference must reject a manifest whose selected SHA does not "
                "match best_model.pth"
            )

        parent_path = directory_path / "parent_fallback.pth"
        parent_path.write_bytes(b"parent-fallback-checkpoint")
        _materialize_checkpoint(str(parent_path), str(selected_path))
        _materialize_checkpoint(str(parent_path), str(best_path))
        parent_sha256 = _sha256_file(str(parent_path))
        assert _sha256_file(str(selected_path)) == parent_sha256
        assert _sha256_file(str(best_path)) == parent_sha256
        parent_deltas = {
            "auc": -0.0001,
            "bce_loss": 0.0002,
            "rmse": 0.0001,
        }
        _write_selection_manifest(
            directory,
            selected_sha256=parent_sha256,
            selected_source="parent",
            residual_active=False,
            accepted=False,
            reason="auc_gain_below_threshold",
            deltas=parent_deltas,
        )
        parent_metadata = _load_inference_selection_metadata(
            directory,
            checkpoint_sha256=_sha256_file(str(best_path)),
            loaded_args=loaded_args,
        )
        assert parent_metadata == {
            "selection_manifest_present": True,
            "selected_source": "parent",
            "residual_active": False,
            "selection_reason": "auc_gain_below_threshold",
            "selection_accepted": False,
            "candidate_parent_deltas": parent_deltas,
            "effective_model_variant": "full_fallback",
        }

        (directory_path / "selection_manifest.json").unlink()
        try:
            _load_inference_selection_metadata(
                directory,
                checkpoint_sha256=_sha256_file(str(best_path)),
                loaded_args=loaded_args,
            )
        except RuntimeError as exc:
            assert "requires selection_manifest.json" in str(exc)
        else:
            raise AssertionError(
                "relation_residual_v4 inference must reject a missing selection "
                "manifest"
            )

        legacy_metadata = _load_inference_selection_metadata(
            directory,
            checkpoint_sha256=_sha256_file(str(best_path)),
            loaded_args={"model_variant": "full", "gec_mode": "v1"},
        )
        assert legacy_metadata == {
            "selection_manifest_present": False,
            "selected_source": "checkpoint",
            "residual_active": False,
            "selection_reason": "legacy_checkpoint_without_manifest",
            "selection_accepted": None,
            "candidate_parent_deltas": None,
            "effective_model_variant": "full",
        }


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
    assert args.train_evidence_mode == "excluded"
    for mode in ("excluded", "neutralized", "self_included"):
        parsed_mode = main_module.parse_args(["--train_evidence_mode", mode])
        assert parsed_mode.train_evidence_mode == mode
    assert main_module.parse_args(["--run_mode", "train"]).run_mode == "train"
    assert main_module.parse_args(["--run_mode", "test"]).run_mode == "test"
    assert GRAPH_IRT_ARCHITECTURE == "graph_irt_v10"
    assert PAIRWISE_AUC_WEIGHT == 0.5
    assert EMA_DECAY == 0.9
    parser_dests = {action.dest for action in main_module.build_parser()._actions}
    assert "pairwise_auc_weight" not in parser_dests
    assert "ema_decay" not in parser_dests
    assert "max_test_batches" not in {
        action.dest for action in main_module.build_parser()._actions
    }

    # Production objective is pure BCE for every structural variant.
    for variant in (
        "full",
        "no_message_passing",
        "item_only",
        "exposure_only",
        "degree_random",
    ):
        variant_args = main_module.parse_args(["--model_variant", variant])
        main_module._apply_model_variant(variant_args)
        assert variant_args.pairwise_auc_weight == 0.0
        assert variant_args.ema_decay == 0.0
        assert variant_args.use_response_evidence
    no_evidence_args = main_module.parse_args(
        ["--model_variant", "no_response_evidence"]
    )
    main_module._apply_model_variant(no_evidence_args)
    assert not no_evidence_args.use_response_evidence
    assert no_evidence_args.pairwise_auc_weight == 0.0
    pairwise_args = main_module.parse_args(
        ["--model_variant", "pairwise_auc"]
    )
    main_module._apply_model_variant(pairwise_args)
    assert pairwise_args.pairwise_auc_weight == PAIRWISE_AUC_WEIGHT
    assert pairwise_args.ema_decay == 0.0
    ema_args = main_module.parse_args(["--model_variant", "ema_bce"])
    main_module._apply_model_variant(ema_args)
    assert ema_args.pairwise_auc_weight == 0.0
    assert ema_args.ema_decay == EMA_DECAY
    assert ema_args.use_response_evidence
    assert _resolve_ema_decay(ema_args) == EMA_DECAY
    residual_args = main_module.parse_args(
        [
            "--model_variant",
            "gec_residual",
            "--warm_start_checkpoint",
            "paired_full/best_model.pth",
        ]
    )
    main_module._apply_model_variant(residual_args)
    main_module._validate_args(residual_args)
    assert residual_args.gec_mode == "residual_v3"
    assert residual_args.evidence_anchor_mode == "full"
    relation_residual_args = main_module.parse_args(
        [
            "--model_variant",
            "gec_relation_residual",
            "--warm_start_checkpoint",
            "paired_full/best_model.pth",
        ]
    )
    main_module._apply_model_variant(relation_residual_args)
    main_module._validate_args(relation_residual_args)
    assert relation_residual_args.gec_mode == "relation_residual_v4"
    assert relation_residual_args.evidence_anchor_mode == "full"
    assert relation_residual_args.train_evidence_mode == "excluded"
    assert relation_residual_args.pairwise_auc_weight == 0.0
    assert relation_residual_args.ema_decay == 0.0
    assert relation_residual_args.use_response_evidence
    missing_relation_parent = main_module.parse_args(
        ["--model_variant", "gec_relation_residual"]
    )
    main_module._apply_model_variant(missing_relation_parent)
    try:
        main_module._validate_args(missing_relation_parent)
    except SystemExit as exc:
        assert "warm_start_checkpoint" in str(exc)
    else:
        raise AssertionError(
            "relation residual training must require a warm-start checkpoint"
        )
    assert "no_response_graph" not in main_module.MODEL_VARIANTS
    boundary_labels = torch.tensor([0.0, 1.0])
    assert set(_training_evidence_kwargs(boundary_labels, "excluded")) == {
        "outcome_to_exclude"
    }
    assert set(_training_evidence_kwargs(boundary_labels, "neutralized")) == {
        "outcome_to_neutralize"
    }
    assert _training_evidence_kwargs(boundary_labels, "self_included") == {}
    invalid_boundary_args = main_module.parse_args(
        [
            "--model_variant",
            "no_response_evidence",
            "--train_evidence_mode",
            "self_included",
        ]
    )
    main_module._apply_model_variant(invalid_boundary_args)
    try:
        main_module._validate_args(invalid_boundary_args)
    except SystemExit as exc:
        assert "requires response evidence" in str(exc)
    else:
        raise AssertionError(
            "a non-default training evidence mode must require response evidence"
        )

    logits = torch.tensor([2.0, -0.5, 1.0, -1.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    prediction, bce, pairwise, has_pairs = _prediction_loss(
        logits,
        labels,
        PAIRWISE_AUC_WEIGHT,
    )
    expected_bce = F.binary_cross_entropy_with_logits(logits, labels)
    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    expected_pairwise = F.softplus(
        -(positive.unsqueeze(1) - negative.unsqueeze(0))
    ).mean()
    expected_prediction = (
        (1.0 - PAIRWISE_AUC_WEIGHT) * expected_bce
        + PAIRWISE_AUC_WEIGHT * expected_pairwise
    )
    assert has_pairs
    assert torch.allclose(bce, expected_bce)
    assert torch.allclose(pairwise, expected_pairwise)
    assert torch.allclose(prediction, expected_prediction)
    prediction.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    single_logits = torch.tensor([0.3, -0.2, 1.1])
    single_labels = torch.ones(3)
    single_prediction, single_bce, single_pairwise, has_pairs = _prediction_loss(
        single_logits,
        single_labels,
        PAIRWISE_AUC_WEIGHT,
    )
    assert not has_pairs
    assert torch.equal(single_pairwise, single_bce)
    assert torch.allclose(single_prediction, single_bce)

    no_pair_prediction, no_pair_bce, _, _ = _prediction_loss(
        logits.detach(),
        labels,
        0.0,
    )
    assert torch.equal(no_pair_prediction, no_pair_bce)
    try:
        _resolve_pairwise_auc_weight(
            SimpleNamespace(model_variant="full", pairwise_auc_weight=0.25)
        )
    except ValueError as exc:
        assert "fixed by model_variant" in str(exc)
    else:
        raise AssertionError("pairwise weight must not become a free hyperparameter")

    checkpoint_pairwise = _checkpoint_args(
        SimpleNamespace(
            model_variant="pairwise_auc",
            pairwise_auc_weight=PAIRWISE_AUC_WEIGHT,
            disable_self_loop=False,
        )
    )
    assert checkpoint_pairwise["pairwise_auc_weight"] == PAIRWISE_AUC_WEIGHT
    checkpoint_bce = _checkpoint_args(
        SimpleNamespace(
            model_variant="full",
            pairwise_auc_weight=0.0,
            train_evidence_mode="neutralized",
            disable_self_loop=False,
        )
    )
    assert checkpoint_bce["pairwise_auc_weight"] == 0.0
    assert checkpoint_bce["train_evidence_mode"] == "neutralized"

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.register_buffer("counter", torch.tensor(1, dtype=torch.long))

    ema_model = _TinyModel()
    ema_state = _clone_model_state(ema_model)
    with torch.no_grad():
        ema_model.weight.fill_(3.0)
        ema_model.counter.fill_(2)
    _update_ema_state(ema_state, ema_model, EMA_DECAY)
    assert torch.allclose(ema_state["weight"], torch.tensor([1.2]))
    assert ema_state["counter"].item() == 2
    with _temporary_model_state(ema_model, ema_state):
        assert torch.allclose(ema_model.weight, torch.tensor([1.2]))
        assert ema_model.counter.item() == 2
    assert torch.allclose(ema_model.weight, torch.tensor([3.0]))
    assert ema_model.counter.item() == 2

    model = torch.nn.Linear(3, 1)
    optimizer = _build_optimizer(model, args)
    assert isinstance(optimizer, torch.optim.AdamW)

    stop_args = SimpleNamespace(min_epochs=20, early_stop_patience=15)
    assert not _should_early_stop(19, 15, stop_args)
    assert not _should_early_stop(20, 14, stop_args)
    assert _should_early_stop(20, 15, stop_args)

    assert not _is_validation_improvement(0.700005, 0.700000, 1e-5)
    assert _is_validation_improvement(0.70002, 0.700000, 1e-5)

    parent_metrics = {"auc": 0.8, "bce_loss": 0.4, "rmse": 0.35}
    accepted_candidate = _select_residual_candidate(
        parent_metrics,
        {"auc": 0.8002, "bce_loss": 0.40005, "rmse": 0.35005},
    )
    assert accepted_candidate["accepted"] is True
    assert accepted_candidate["reason"] == "accepted"
    assert set(accepted_candidate["deltas"]) == {"auc", "bce_loss", "rmse"}
    assert abs(accepted_candidate["deltas"]["auc"] - 0.0002) < 1e-12
    assert abs(accepted_candidate["deltas"]["bce_loss"] - 0.00005) < 1e-12
    assert abs(accepted_candidate["deltas"]["rmse"] - 0.00005) < 1e-12

    insufficient_auc = _select_residual_candidate(
        parent_metrics,
        {"auc": 0.80005, "bce_loss": 0.3999, "rmse": 0.3499},
    )
    assert insufficient_auc["accepted"] is False
    assert "auc" in insufficient_auc["reason"].lower()

    bce_regression = _select_residual_candidate(
        parent_metrics,
        {"auc": 0.8003, "bce_loss": 0.4002, "rmse": 0.35},
    )
    assert bce_regression["accepted"] is False
    assert "bce" in bce_regression["reason"].lower()

    rmse_regression = _select_residual_candidate(
        parent_metrics,
        {"auc": 0.8003, "bce_loss": 0.4, "rmse": 0.3502},
    )
    assert rmse_regression["accepted"] is False
    assert "rmse" in rmse_regression["reason"].lower()

    threshold_boundary = _select_residual_candidate(
        parent_metrics,
        {"auc": 0.8001, "bce_loss": 0.4001, "rmse": 0.3501},
    )
    assert threshold_boundary["accepted"] is True
    for invalid_candidate in (
        {"auc": float("nan"), "bce_loss": 0.4, "rmse": 0.35},
        {"auc": 0.8002, "bce_loss": float("inf"), "rmse": 0.35},
        {"auc": 0.8002, "bce_loss": 0.4},
    ):
        try:
            _select_residual_candidate(parent_metrics, invalid_candidate)
        except ValueError as exc:
            assert "finite" in str(exc)
        else:
            raise AssertionError(
                "residual selection must reject missing or non-finite metrics"
            )

    _assert_inference_selection_metadata_contract()

    hash_args = SimpleNamespace(
        dataset_name="assist_09",
        epochs=120,
        min_stu_interactions=15,
        min_exer_interactions=0,
        pairwise_auc_weight=PAIRWISE_AUC_WEIGHT,
        ema_decay=0.0,
    )
    baseline_hash = _config_hash(hash_args)
    hash_args.epochs = 100
    assert _config_hash(hash_args) != baseline_hash
    hash_args.epochs = 120
    hash_args.min_stu_interactions = 0
    assert _config_hash(hash_args) != baseline_hash
    hash_args.min_stu_interactions = 15
    hash_args.pairwise_auc_weight = 0.0
    assert _config_hash(hash_args) != baseline_hash
    hash_args.pairwise_auc_weight = PAIRWISE_AUC_WEIGHT
    hash_args.ema_decay = EMA_DECAY
    assert _config_hash(hash_args) != baseline_hash

    trainer_source = (Path(ROOT) / "src" / "trainer.py").read_text(encoding="utf-8")
    for removed_token in (
        "response_graph",
        "enable_response",
        "item_difficulty_delta",
    ):
        assert removed_token not in trainer_source, removed_token

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
