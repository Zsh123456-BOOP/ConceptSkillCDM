"""Prove train response evidence excludes the current target exactly."""

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
    return students, exercises, concepts, q_matrix, item_prior, exposure_prior


def main() -> None:
    frame = pd.DataFrame(
        {
            "stu_id": [10, 10, 10, 20, 20, 20],
            "exer_id": [100, 100, 102, 100, 102, 101],
            "cpt_seq": ["1", "1", "2,3", "1", "2,3", "1,2"],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "label": [0, 1, 0, 1, 1, 0],
        }
    )
    flipped = frame.copy()
    flipped.loc[0, "label"] = 1 - flipped.loc[0, "label"]

    first = _structure(frame)
    second = _structure(flipped)
    assert first[:3] == second[:3]
    for left, right in zip(first[3:6], second[3:6]):
        assert torch.allclose(left, right, atol=1e-7)

    students, exercises, concepts, q_matrix, item_prior, exposure_prior = first
    stats_a = build_student_concept_response_stats(
        frame,
        students,
        exercises,
        q_matrix,
    )
    stats_b = build_student_concept_response_stats(
        flipped,
        students,
        exercises,
        q_matrix,
    )
    assert float(stats_a["global_count"].item()) == len(frame)
    assert not torch.equal(
        stats_a["student_concept_correct"],
        stats_b["student_concept_correct"],
    )
    assert not torch.equal(
        stats_a["student_concept_residual_sum"],
        stats_b["student_concept_residual_sum"],
    )
    query_pair_key = students[10] * len(exercises) + exercises[100]
    query_position_a = torch.searchsorted(
        stats_a["student_item_keys"],
        torch.tensor(query_pair_key),
    )
    query_position_b = torch.searchsorted(
        stats_b["student_item_keys"],
        torch.tensor(query_pair_key),
    )
    assert torch.equal(
        stats_a["student_item_expected_correct"][query_position_a],
        stats_b["student_item_expected_correct"][query_position_b],
    )

    kwargs = dict(
        num_students=len(students),
        num_exercises=len(exercises),
        num_concepts=len(concepts),
        q_matrix=q_matrix,
        item_prior_matrix=item_prior,
        exposure_prior_matrix=exposure_prior,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_response_evidence=True,
    )
    student_ids = torch.tensor([students[10]], dtype=torch.long)
    exercise_ids = torch.tensor([exercises[100]], dtype=torch.long)
    torch.manual_seed(123)
    model_a = CognitiveDiagnosisModel(
        **kwargs,
        response_evidence_stats=stats_a,
    ).eval()
    torch.manual_seed(123)
    model_b = CognitiveDiagnosisModel(
        **kwargs,
        response_evidence_stats=stats_b,
    ).eval()

    # Flipping the current target changes the stored train total, but after
    # subtracting that same row its prediction is bitwise invariant.
    loo_a = model_a(
        student_ids,
        exercise_ids,
        outcome_to_exclude=torch.tensor([frame.loc[0, "label"]]),
        return_logits=True,
    )
    loo_b = model_b(
        student_ids,
        exercise_ids,
        outcome_to_exclude=torch.tensor([flipped.loc[0, "label"]]),
        return_logits=True,
    )
    assert torch.allclose(loo_a, loo_b, atol=1e-7, rtol=0.0)

    # At validation/test time the complete train evidence is intentionally
    # visible, so changing a train outcome must change the evidence path.
    assert not torch.equal(
        model_a(student_ids, exercise_ids, return_logits=True),
        model_b(student_ids, exercise_ids, return_logits=True),
    )
    structural_names = {
        name for name, _ in list(model_a.named_parameters()) + list(model_a.named_buffers())
    }
    assert any("response_evidence" in name for name in structural_names)
    assert any("response_student_concept" in name for name in structural_names)
    assert not any("item_difficulty" in name for name in structural_names)

    no_evidence_kwargs = dict(kwargs)
    no_evidence_kwargs["use_response_evidence"] = False
    no_evidence = CognitiveDiagnosisModel(**no_evidence_kwargs)
    no_evidence_names = {
        name
        for name, _ in list(no_evidence.named_parameters())
        + list(no_evidence.named_buffers())
    }
    assert not any("response_" in name for name in no_evidence_names)
    assert GRAPH_IRT_ARCHITECTURE == "graph_irt_v10"
    print("OK: train response evidence is target-safe under exact leave-one-out.")


if __name__ == "__main__":
    main()
