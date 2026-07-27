"""Contracts for the minimal LEA posterior-gap rate-evidence candidate."""

import os
import sys

import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import (
    build_id_mappings,
    build_item_cooccurrence_prior,
    build_q_matrix,
    build_student_concept_response_stats,
    build_student_coexposure_prior,
)
from src.model import CognitiveDiagnosisModel, GRAPH_IRT_ARCHITECTURE


def _structure(dataframe: pd.DataFrame):
    students, exercises, concepts = build_id_mappings([dataframe])
    q_matrix = build_q_matrix([dataframe], exercises, concepts)
    item_prior, _ = build_item_cooccurrence_prior(q_matrix)
    exposure_prior, _ = build_student_coexposure_prior([dataframe], concepts)
    stats = build_student_concept_response_stats(
        dataframe,
        students,
        exercises,
        q_matrix,
    )
    return (
        students,
        exercises,
        concepts,
        q_matrix,
        item_prior,
        exposure_prior,
        stats,
    )


def _build_model(structure, rate_evidence_mode: str) -> CognitiveDiagnosisModel:
    students, exercises, concepts, q_matrix, item_prior, exposure_prior, stats = (
        structure
    )
    return CognitiveDiagnosisModel(
        num_students=len(students),
        num_exercises=len(exercises),
        num_concepts=len(concepts),
        q_matrix=q_matrix,
        item_prior_matrix=item_prior,
        exposure_prior_matrix=exposure_prior,
        response_evidence_stats=stats,
        use_response_evidence=True,
        evidence_anchor_mode="direct_only",
        evidence_state_injection=False,
        rate_evidence_mode=rate_evidence_mode,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        graph_propagation_alpha=0.20,
    )


def main() -> None:
    frame = pd.DataFrame(
        {
            "stu_id": [10, 10, 20, 20, 20],
            "exer_id": [100, 100, 100, 101, 102],
            "cpt_seq": ["1", "1", "1", "2", "1,2"],
            "label": [1, 0, 0, 1, 1],
        }
    )
    structure = _structure(frame)
    students, exercises, concepts, q_matrix, _, _, _ = structure

    torch.manual_seed(71)
    baseline = _build_model(structure, "reliability_scaled").eval()
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(71)
    candidate = _build_model(structure, "posterior_gap").eval()
    candidate_rng = torch.random.get_rng_state().clone()

    # The intervention is configuration-only: no parameter or RNG trajectory changes.
    assert torch.equal(baseline_rng, candidate_rng)
    assert tuple(baseline.state_dict()) == tuple(candidate.state_dict())
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
    assert sum(parameter.numel() for parameter in baseline.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    assert baseline.relation_learning is not None
    assert candidate.relation_learning is not None

    student_ids = torch.tensor([students[10]], dtype=torch.long)
    exercise_ids = torch.tensor([exercises[100]], dtype=torch.long)
    q_vector = torch.ones((1, len(concepts)), dtype=q_matrix.dtype)
    baseline_evidence, count, _, _, _ = baseline._build_response_evidence(
        student_ids,
        exercise_ids,
        q_vector,
        None,
        None,
    )
    candidate_evidence, candidate_count, _, _, _ = (
        candidate._build_response_evidence(
            student_ids,
            exercise_ids,
            q_vector,
            None,
            None,
        )
    )
    assert torch.equal(count, candidate_count)

    no_support = count == 0
    with_support = count > 0
    assert no_support.any() and with_support.any()
    assert torch.equal(
        candidate_evidence[..., 0][no_support],
        baseline_evidence[..., 0][no_support],
    )
    reliability = count / (count + 1.0)
    assert torch.allclose(
        candidate_evidence[..., 0][with_support],
        baseline_evidence[..., 0][with_support] / reliability[with_support],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(
        candidate_evidence[..., 1],
        baseline_evidence[..., 1],
    )

    logits, details = candidate(
        student_ids,
        exercise_ids,
        return_details=True,
        return_logits=True,
    )
    assert torch.equal(logits, details["irt_logit"])
    assert torch.equal(details["logits"], details["irt_logit"])
    assert details["knowledge_state_graph_delta"].item() > 0.0

    # Flipping a training target and excluding that same row remains isolated.
    flipped = frame.copy()
    flipped.loc[0, "label"] = 1 - flipped.loc[0, "label"]
    flipped_structure = _structure(flipped)
    torch.manual_seed(93)
    original_model = _build_model(structure, "posterior_gap").eval()
    torch.manual_seed(93)
    flipped_model = _build_model(flipped_structure, "posterior_gap").eval()
    original_logits, original_details = original_model(
        student_ids,
        exercise_ids,
        outcome_to_exclude=torch.tensor([frame.loc[0, "label"]]),
        return_details=True,
        return_logits=True,
    )
    flipped_logits, flipped_details = flipped_model(
        student_ids,
        exercise_ids,
        outcome_to_exclude=torch.tensor([flipped.loc[0, "label"]]),
        return_details=True,
        return_logits=True,
    )
    assert torch.allclose(original_logits, flipped_logits, atol=1e-7, rtol=0.0)
    assert torch.allclose(
        original_details["response_evidence"],
        flipped_details["response_evidence"],
        atol=1e-7,
        rtol=0.0,
    )
    assert GRAPH_IRT_ARCHITECTURE == "graph_irt_v10"
    print("OK: LEA posterior-gap single-gate contracts passed.")


if __name__ == "__main__":
    main()
