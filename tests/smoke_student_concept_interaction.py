"""Smoke checks for the label-free student-by-concept factorization."""

from __future__ import annotations

import math
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main as main_module
from src.model_graph import StudentKnowledgeEncoder
from src.experiment_utils import _config_hash
from src.model import CognitiveDiagnosisModel
from src.trainer import (
    _checkpoint_args,
    _collect_structural_switches,
    _model_kwargs,
    _runtime_facts,
)


def _encoder(
    mode: str,
    scale: float = 1.0,
    rank: int = 8,
    init_std: float = 0.1,
) -> StudentKnowledgeEncoder:
    encoder = StudentKnowledgeEncoder(
        num_students=2,
        num_concepts=2,
        knowledge_dim=2,
        num_gnn_layers=0,
        num_relation_heads=1,
        dropout=0.0,
        student_concept_interaction=mode,
        student_concept_interaction_scale=scale,
        student_concept_interaction_rank=rank,
        student_concept_interaction_init_std=init_std,
    )
    with torch.no_grad():
        encoder.student_global.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        )
        encoder.concept_emb.weight.copy_(
            torch.tensor([[2.0, 7.0], [11.0, 13.0]])
        )
    return encoder.eval()


def main() -> None:
    valid_args = main_module.parse_args(
        [
            "--student_concept_interaction",
            "low_rank",
            "--student_concept_interaction_rank",
            "4",
            "--student_concept_interaction_init_std",
            "0.05",
        ]
    )
    main_module._validate_args(valid_args)
    for invalid_args in (
        ["--student_concept_interaction_rank", "0"],
        ["--student_concept_interaction_init_std", "nan"],
        ["--student_concept_interaction_init_std", "1.1"],
        [
            "--student_concept_interaction",
            "low_rank",
            "--student_concept_interaction_init_std",
            "0",
        ],
    ):
        try:
            main_module._validate_args(main_module.parse_args(invalid_args))
        except SystemExit:
            pass
        else:
            raise AssertionError(f"expected validation failure for {invalid_args}")
    inactive_zero_args = main_module.parse_args(
        ["--student_concept_interaction", "none", "--student_concept_interaction_init_std", "0"]
    )
    main_module._validate_args(inactive_zero_args)
    try:
        _encoder("low_rank", init_std=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("low_rank encoder must reject zero init_std")

    student_ids = torch.tensor([0, 1])
    additive = _encoder("none").compose_initial_state(student_ids)
    concepts = torch.tensor([[2.0, 7.0], [11.0, 13.0]])
    students = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    expected_additive = concepts.unsqueeze(0) + students.unsqueeze(1)
    assert torch.allclose(additive, expected_additive)

    scale = 0.5
    factorized = _encoder("hadamard", scale).compose_initial_state(student_ids)
    expected_interaction = (
        scale
        * math.sqrt(2.0)
        * concepts.unsqueeze(0)
        * students.unsqueeze(1)
    )
    assert torch.allclose(factorized, expected_additive + expected_interaction)

    additive_contrast = additive[:, 1] - additive[:, 0]
    factorized_contrast = factorized[:, 1] - factorized[:, 0]
    assert torch.allclose(additive_contrast[0], additive_contrast[1])
    assert not torch.allclose(factorized_contrast[0], factorized_contrast[1])

    diagnostics = _encoder("hadamard", scale).get_interaction_diagnostics(student_ids)
    assert diagnostics["student_concept_interaction_rms"].item() > 0.0
    assert diagnostics["student_concept_interaction_ratio"].item() > 0.0

    none_encoder = _encoder("none")
    hadamard_encoder = _encoder("hadamard")
    old_state_keys = set(none_encoder.state_dict())
    assert set(hadamard_encoder.state_dict()) == old_state_keys
    assert not any("interaction_factor" in key for key in old_state_keys)
    assert "interaction_projection.weight" not in old_state_keys
    hadamard_encoder.load_state_dict(none_encoder.state_dict(), strict=True)

    low_rank = _encoder("low_rank", scale=0.5, rank=2, init_std=0.1)
    with torch.no_grad():
        low_rank.student_interaction_factor.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )
        low_rank.concept_interaction_factor.weight.copy_(
            torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        )
        low_rank.interaction_projection.weight.copy_(torch.eye(2))
    low_rank_state = low_rank.compose_initial_state(student_ids)
    expected_low_rank_interaction = 0.5 * (
        low_rank.student_interaction_factor(student_ids).unsqueeze(1)
        * low_rank.concept_interaction_factor.weight.unsqueeze(0)
    )
    assert low_rank_state.shape == (2, 2, 2)
    assert torch.allclose(low_rank_state, expected_additive + expected_low_rank_interaction)
    low_rank_keys = set(low_rank.state_dict())
    assert low_rank_keys - old_state_keys == {
        "student_interaction_factor.weight",
        "concept_interaction_factor.weight",
        "interaction_projection.weight",
    }
    low_rank_state.sum().backward()
    for parameter in (
        low_rank.student_interaction_factor.weight,
        low_rank.concept_interaction_factor.weight,
        low_rank.interaction_projection.weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum().item() > 0.0
    low_rank_diagnostics = low_rank.get_interaction_diagnostics(student_ids)
    assert low_rank_diagnostics["student_concept_interaction_rms"].item() > 0.0
    assert low_rank_diagnostics["student_concept_interaction_ratio"].item() > 0.0

    old_switches = _collect_structural_switches({})
    assert old_switches["student_concept_interaction"] == "none"
    assert old_switches["student_concept_interaction_scale"] == 1.0
    assert old_switches["student_concept_interaction_rank"] == 8
    assert old_switches["student_concept_interaction_init_std"] == 0.1

    hash_args = type("Args", (), {})()
    hash_args.student_concept_interaction = "none"
    hash_args.student_concept_interaction_scale = 1.0
    hash_args.student_concept_interaction_rank = 8
    hash_args.student_concept_interaction_init_std = 0.1
    baseline_hash = _config_hash(hash_args)
    hash_args.student_concept_interaction = "low_rank"
    assert _config_hash(hash_args) != baseline_hash
    hash_args.student_concept_interaction = "none"
    hash_args.student_concept_interaction_rank = 4
    assert _config_hash(hash_args) != baseline_hash
    hash_args.student_concept_interaction_rank = 8
    hash_args.student_concept_interaction_init_std = 0.2
    assert _config_hash(hash_args) != baseline_hash

    relation = torch.eye(2).unsqueeze(0)
    no_message = _encoder("hadamard", 1.0)
    no_message.propagation_alpha = 0.0
    state, initial = no_message(student_ids, relation, return_initial=True)
    assert torch.equal(state, initial)
    state.sum().backward()
    assert no_message.student_global.weight.grad.abs().sum().item() > 0.0
    assert no_message.concept_emb.weight.grad.abs().sum().item() > 0.0

    info = {
        "num_students": 2,
        "num_exercises": 2,
        "num_concepts": 2,
        "q_matrix": torch.eye(2),
        "item_prior_matrix": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "exposure_prior_matrix": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    }
    old_kwargs = _model_kwargs({}, info)
    assert old_kwargs["student_concept_interaction"] == "none"
    assert old_kwargs["student_concept_interaction_rank"] == 8
    assert old_kwargs["student_concept_interaction_init_std"] == 0.1
    old_model = CognitiveDiagnosisModel(**old_kwargs)
    hadamard_kwargs = dict(old_kwargs)
    hadamard_kwargs["student_concept_interaction"] = "hadamard"
    hadamard_model = CognitiveDiagnosisModel(**hadamard_kwargs)
    assert set(hadamard_model.state_dict()) == set(old_model.state_dict())
    hadamard_model.load_state_dict(old_model.state_dict(), strict=True)

    low_rank_kwargs = dict(old_kwargs)
    low_rank_kwargs.update(
        student_concept_interaction="low_rank",
        student_concept_interaction_scale=0.5,
        student_concept_interaction_rank=2,
        student_concept_interaction_init_std=0.05,
    )
    low_rank_model = CognitiveDiagnosisModel(**low_rank_kwargs)
    facts = _runtime_facts(low_rank_model)
    assert facts["student_concept_interaction"] == "low_rank"
    assert facts["student_concept_interaction_scale"] == 0.5
    assert facts["student_concept_interaction_rank"] == 2
    assert facts["student_concept_interaction_init_std"] == 0.05
    assert facts["num_parameters"] > _runtime_facts(old_model)["num_parameters"]

    checkpoint_args = _checkpoint_args(type("Args", (), low_rank_kwargs)())
    assert checkpoint_args["student_concept_interaction"] == "low_rank"
    assert checkpoint_args["student_concept_interaction_rank"] == 2
    assert checkpoint_args["student_concept_interaction_init_std"] == 0.05
    print("OK: student-concept interaction checks passed.")


if __name__ == "__main__":
    main()
