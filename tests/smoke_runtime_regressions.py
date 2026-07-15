"""Runtime invariants for the single-path Graph-IRT model."""

import os
import sys

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import CognitiveDiagnosisModel


BANNED_TOKENS = (
    "personal",
    "posterior",
    "roadmap",
    "tutor",
    "calibration",
    "residual_logit",
    "ae_",
    "item_matching",
    "response_graph",
    "item_difficulty",
)


def _build_model() -> CognitiveDiagnosisModel:
    q_matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
    )
    prior = torch.tensor(
        [[0.0, 0.7, 0.3], [0.5, 0.0, 0.5], [0.6, 0.4, 0.0]]
    )
    return CognitiveDiagnosisModel(
        num_students=4,
        num_exercises=3,
        num_concepts=3,
        q_matrix=q_matrix,
        item_prior_matrix=prior,
        exposure_prior_matrix=torch.zeros_like(prior),
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        graph_propagation_alpha=0.25,
    )


def main() -> None:
    torch.manual_seed(9)
    model = _build_model()
    student_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    exercise_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

    logits, details = model(
        student_ids, exercise_ids, return_details=True, return_logits=True
    )
    probabilities = model(student_ids, exercise_ids, return_logits=False)

    assert torch.equal(logits, details["irt_logit"])
    assert torch.equal(details["logits"], details["irt_logit"])
    assert torch.allclose(probabilities, torch.sigmoid(logits), atol=0.0, rtol=0.0)
    assert torch.isfinite(logits).all()
    for key in details:
        assert not any(token in key.lower() for token in BANNED_TOKENS), key

    state_keys = tuple(model.state_dict())
    for key in state_keys:
        assert not any(token in key.lower() for token in BANNED_TOKENS), key
    assert not hasattr(model, "initialize_ae_logit_priors")

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    active_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert active_gradients
    assert all(torch.isfinite(gradient).all() for gradient in active_gradients)

    clone = _build_model().eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    expected = model(student_ids, exercise_ids, return_logits=True)
    actual = clone(student_ids, exercise_ids, return_logits=True)
    assert torch.equal(expected, actual)
    print("OK: single-path Graph-IRT runtime invariants passed.")


if __name__ == "__main__":
    main()
