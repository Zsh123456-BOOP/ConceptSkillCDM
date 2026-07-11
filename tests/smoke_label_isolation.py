"""Prove that labels cannot enter Graph-IRT structure before optimization."""

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
    build_student_coexposure_prior,
)
from src.model import CognitiveDiagnosisModel


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
            "exer_id": [100, 101, 102, 100, 102, 101],
            "cpt_seq": ["1", "1,2", "2,3", "1", "2,3", "1,2"],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "label": [0, 1, 0, 1, 1, 0],
        }
    )
    flipped = frame.copy()
    flipped["label"] = 1 - flipped["label"]

    first = _structure(frame)
    second = _structure(flipped)
    assert first[:3] == second[:3]
    for left, right in zip(first[3:], second[3:]):
        assert torch.allclose(left, right, atol=1e-7)

    students, exercises, concepts, q_matrix, item_prior, exposure_prior = first
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
    )
    student_ids = torch.tensor([0, 1], dtype=torch.long)
    exercise_ids = torch.tensor([0, 2], dtype=torch.long)
    for interaction in ("none", "low_rank"):
        interaction_kwargs = dict(kwargs, student_concept_interaction=interaction)
        torch.manual_seed(123)
        model_a = CognitiveDiagnosisModel(**interaction_kwargs).eval()
        torch.manual_seed(123)
        model_b = CognitiveDiagnosisModel(**interaction_kwargs).eval()
        assert torch.equal(
            model_a(student_ids, exercise_ids, return_logits=True),
            model_b(student_ids, exercise_ids, return_logits=True),
        )
    print("OK: labels are isolated from graph construction and pre-training logits.")


if __name__ == "__main__":
    main()
