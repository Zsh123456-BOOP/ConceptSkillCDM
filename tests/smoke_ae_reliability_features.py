import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import CognitiveDiagnosisModel


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = CognitiveDiagnosisModel(
        num_students=2,
        num_exercises=3,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=True,
        graph_topk=2,
        graph_prior_logit_scale=0.2,
        graph_query_readout_scale=0.02,
        graph_query_readout_2hop_scale=0.01,
        ae_logit_residual_scale=1.0,
        ae_posterior_prior_scale=1.0,
        personal_mastery_prior_scale=1.0,
        personal_mastery_count_smoothing=8.0,
    )
    model.initialize_ae_logit_priors(
        student_logits=torch.zeros(2),
        exercise_logits=torch.zeros(3),
        concept_logits=torch.zeros(3),
        student_concept_logits=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        scale=1.0,
        student_count_features=torch.zeros(2),
        exercise_count_features=torch.zeros(3),
        concept_count_features=torch.zeros(3),
        student_concept_count_features=torch.zeros(2, 3),
        student_concept_observed_counts=torch.tensor(
            [
                [20.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    logits, details = model(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
        return_details=True,
        return_logits=True,
    )
    rel = details.get("e_query_mastery_logit")
    _assert(rel is not None, "forward details must expose E query mastery contribution")
    _assert(
        float(rel[0].item()) > 0.0 and abs(float(rel[1].item())) < 1e-8,
        "observed student-concept counts must reliability-gate E query mastery evidence",
    )
    _assert(torch.isfinite(logits).all().item(), "logits must remain finite with reliability features")
    print("smoke_ae_reliability_features passed")


if __name__ == "__main__":
    main()
