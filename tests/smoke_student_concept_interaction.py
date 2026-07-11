"""Smoke checks for the label-free student-by-concept factorization."""

from __future__ import annotations

import math
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model_graph import StudentKnowledgeEncoder
from src.experiment_utils import _config_hash
from src.model import CognitiveDiagnosisModel
from src.trainer import _collect_structural_switches, _model_kwargs


def _encoder(mode: str, scale: float = 1.0) -> StudentKnowledgeEncoder:
    encoder = StudentKnowledgeEncoder(
        num_students=2,
        num_concepts=2,
        knowledge_dim=2,
        num_gnn_layers=0,
        num_relation_heads=1,
        dropout=0.0,
        student_concept_interaction=mode,
        student_concept_interaction_scale=scale,
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

    old_switches = _collect_structural_switches({})
    assert old_switches["student_concept_interaction"] == "none"
    assert old_switches["student_concept_interaction_scale"] == 1.0

    hash_args = type("Args", (), {})()
    hash_args.student_concept_interaction = "none"
    hash_args.student_concept_interaction_scale = 1.0
    baseline_hash = _config_hash(hash_args)
    hash_args.student_concept_interaction = "hadamard"
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
    old_model = CognitiveDiagnosisModel(**old_kwargs)
    new_kwargs = dict(old_kwargs)
    new_kwargs["student_concept_interaction"] = "hadamard"
    new_model = CognitiveDiagnosisModel(**new_kwargs)
    new_model.load_state_dict(old_model.state_dict(), strict=True)
    print("OK: student-concept interaction checks passed.")


if __name__ == "__main__":
    main()
