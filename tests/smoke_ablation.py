"""Smoke checks for the clean Graph-IRT ablation semantics."""

import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import CognitiveDiagnosisModel


def _build_model(*, propagation_alpha: float) -> CognitiveDiagnosisModel:
    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    item_prior = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    return CognitiveDiagnosisModel(
        num_students=5,
        num_exercises=4,
        num_concepts=4,
        q_matrix=q_matrix,
        item_prior_matrix=item_prior,
        exposure_prior_matrix=torch.zeros_like(item_prior),
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=2,
        dropout=0.0,
        graph_propagation_alpha=propagation_alpha,
    )


def main() -> None:
    torch.manual_seed(17)
    full = _build_model(propagation_alpha=0.35).eval()
    torch.manual_seed(17)
    no_message = _build_model(propagation_alpha=0.0).eval()

    student_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    exercise_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    _, full_details = full(
        student_ids, exercise_ids, return_details=True, return_logits=True
    )
    _, no_message_details = no_message(
        student_ids, exercise_ids, return_details=True, return_logits=True
    )

    assert torch.equal(
        no_message_details["knowledge_state"], no_message_details["initial_state"]
    ), "graph_propagation_alpha=0 must be an exact no-message-passing ablation"
    assert full_details["knowledge_state_graph_delta"].item() > 0.0

    relation = full_details["relation_matrices"]
    assert torch.isfinite(relation).all()
    assert (relation >= 0.0).all()
    assert torch.allclose(
        relation.sum(dim=-1), torch.ones_like(relation.sum(dim=-1)), atol=1e-6
    )
    print("OK: clean Graph-IRT ablation semantics passed.")


if __name__ == "__main__":
    main()
