"""Contracts for the per-concept, factorized-difficulty 2PL head."""

import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.prediction_head import CognitiveDiagnosisHead, ExerciseDifficultyEncoder


def main() -> None:
    torch.manual_seed(7)
    encoder = ExerciseDifficultyEncoder(num_exercises=3)
    b, a = encoder(torch.tensor([0, 1, 2]))
    assert tuple(b.shape) == (3,)
    assert tuple(a.shape) == (3,)
    assert (a > 0).all()

    head = CognitiveDiagnosisHead(knowledge_dim=4, num_concepts=3)
    state = torch.randn(2, 3, 4)
    item_state = torch.randn(2, 4, requires_grad=True)
    concept_basis = torch.randn(3, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    logits, details = head(
        knowledge_state=state,
        item_state=item_state,
        concept_basis=concept_basis,
        concept_mask=mask,
        b=torch.zeros(2),
        a=torch.ones(2),
        return_details=True,
    )
    assert tuple(logits.shape) == (2,)
    assert torch.equal(
        details["item_difficulty_delta"],
        torch.zeros_like(details["item_difficulty_delta"]),
    ), "zero projection must start exactly from scalar 2PL"
    expected = (details["concept_irt_logit"] * mask).sum(1) / mask.sum(1)
    assert torch.allclose(logits, expected, atol=0.0, rtol=0.0)

    logits.square().mean().backward()
    projection_grad = head.item_difficulty_projection.weight.grad
    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert projection_grad.abs().sum() > 0
    assert head.theta_proj.weight.grad is not None

    differentiated = CognitiveDiagnosisHead(knowledge_dim=2, num_concepts=2)
    with torch.no_grad():
        differentiated.theta_proj.weight.zero_()
        differentiated.theta_proj.bias.zero_()
        differentiated.item_difficulty_projection.weight.copy_(torch.eye(2))
    _, differentiated_details = differentiated(
        knowledge_state=torch.zeros(1, 2, 2),
        item_state=torch.tensor([[2.0, -1.0]]),
        concept_basis=torch.tensor([[2.0, -1.0], [-1.0, 2.0]]),
        concept_mask=torch.ones(1, 2),
        b=torch.zeros(1),
        a=torch.ones(1),
        return_details=True,
    )
    delta = differentiated_details["item_difficulty_delta"]
    assert not torch.equal(delta[:, 0], delta[:, 1])
    assert torch.isfinite(delta).all()
    print("OK: per-concept 2PL prediction-head contracts passed.")


if __name__ == "__main__":
    main()
