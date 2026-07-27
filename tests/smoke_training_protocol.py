"""Smoke checks for the auditable optimizer and robust early-stop controls."""

from __future__ import annotations

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
from src.evidence_completion import MEC_SCHEMA
from src.model import GRAPH_IRT_ARCHITECTURE
from src.trainer import (
    _build_data_identity,
    _build_optimizer,
    _checkpoint_args,
    _clone_model_state,
    _claim_test_seal,
    _collect_structural_switches,
    _is_validation_improvement,
    _prediction_loss,
    _resolve_ema_decay,
    _resolve_pairwise_auc_weight,
    _resolve_rate_evidence_mode,
    _temporary_model_state,
    _training_evidence_kwargs,
    _update_ema_state,
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
    assert "rate_evidence_mode" not in parser_dests
    assert "max_test_batches" not in {
        action.dest for action in main_module.build_parser()._actions
    }

    # Production objective is pure BCE for every structural variant.
    for variant in (
        "full",
        "no_message_passing",
        "lea_rate_single_gate",
        "mec",
        "mec_state_graph",
        "item_only",
        "exposure_only",
        "degree_random",
    ):
        variant_args = main_module.parse_args(["--model_variant", variant])
        main_module._apply_model_variant(variant_args)
        assert variant_args.pairwise_auc_weight == 0.0
        assert variant_args.ema_decay == 0.0
        assert variant_args.use_response_evidence
    lea_args = main_module.parse_args(
        ["--model_variant", "lea_rate_single_gate"]
    )
    main_module._apply_model_variant(lea_args)
    assert lea_args.evidence_anchor_mode == "direct_only"
    assert lea_args.rate_evidence_mode == "posterior_gap"
    assert not lea_args.disable_graph_module
    assert lea_args.graph_propagation_alpha == 0.20
    assert lea_args.warm_start_checkpoint is None
    main_module._validate_args(lea_args)
    mec_args = main_module.parse_args(["--model_variant", "mec"])
    main_module._apply_model_variant(mec_args)
    assert mec_args.evidence_anchor_mode == "mec"
    assert mec_args.disable_graph_module
    mec_state_args = main_module.parse_args(
        ["--model_variant", "mec_state_graph"]
    )
    main_module._apply_model_variant(mec_state_args)
    assert mec_state_args.evidence_anchor_mode == "mec"
    assert not mec_state_args.disable_graph_module
    assert mec_state_args.graph_propagation_alpha == 0.20
    no_graph_args = main_module.parse_args(
        ["--model_variant", "no_graph_calibration"]
    )
    main_module._apply_model_variant(no_graph_args)
    assert no_graph_args.evidence_anchor_mode == "direct_only"
    assert no_graph_args.rate_evidence_mode == "reliability_scaled"
    assert no_graph_args.disable_graph_module
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
    assert "no_response_graph" not in main_module.MODEL_VARIANTS
    assert not {
        "mec_lambda",
        "mec_mask_rate",
        "mec_hidden",
        "mec_pseudo_count_cap",
        "mec_temperature",
        "mec_consistency_weight",
    } & parser_dests
    missing_warm = main_module.parse_args(["--model_variant", "mec"])
    main_module._apply_model_variant(missing_warm)
    try:
        main_module._validate_args(missing_warm)
    except SystemExit as exc:
        assert "requires --warm_start_checkpoint" in str(exc)
    else:
        raise AssertionError("MEC training must require a matched baseline")
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
    assert checkpoint_bce["rate_evidence_mode"] == "reliability_scaled"
    checkpoint_lea = _checkpoint_args(lea_args)
    assert checkpoint_lea["rate_evidence_mode"] == "posterior_gap"
    legacy_switches = _collect_structural_switches(
        SimpleNamespace(model_variant="full")
    )
    assert legacy_switches["rate_evidence_mode"] == "reliability_scaled"
    assert _resolve_rate_evidence_mode({}) == "reliability_scaled"
    checkpoint_mec = _checkpoint_args(
        SimpleNamespace(
            model_variant="mec",
            pairwise_auc_weight=0.0,
            train_evidence_mode="excluded",
            evidence_anchor_mode="mec",
            disable_graph_module=True,
            disable_self_loop=False,
            warm_start_checkpoint="/tmp/source/best_model.pth",
        )
    )
    assert checkpoint_mec["disable_graph_module"] is True
    assert checkpoint_mec["evidence_anchor_mode"] == "mec"
    assert checkpoint_mec["mec_schema"] == MEC_SCHEMA
    assert checkpoint_mec["warm_start_checkpoint"] == os.path.realpath(
        "/tmp/source/best_model.pth"
    )

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

    hash_args = SimpleNamespace(
        dataset_name="assist_09",
        epochs=120,
        min_stu_interactions=15,
        min_exer_interactions=0,
        pairwise_auc_weight=PAIRWISE_AUC_WEIGHT,
        ema_decay=0.0,
    )
    baseline_hash = _config_hash(hash_args)
    explicit_legacy_hash_args = SimpleNamespace(
        **vars(hash_args),
        rate_evidence_mode="reliability_scaled",
    )
    assert _config_hash(explicit_legacy_hash_args) == baseline_hash
    posterior_hash_args = SimpleNamespace(
        **vars(hash_args),
        rate_evidence_mode="posterior_gap",
    )
    assert _config_hash(posterior_hash_args) != baseline_hash
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
    hash_args.mec_schema = "target_pseudocount_v2"
    legacy_mec_hash = _config_hash(hash_args)
    hash_args.mec_schema = MEC_SCHEMA
    assert _config_hash(hash_args) != legacy_mec_hash

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
