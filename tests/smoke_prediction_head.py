"""Contracts for the sole Q-masked scalar-difficulty 2PL head."""

import inspect
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
    assert set(dict(head.named_parameters())) == {
        "theta_proj.weight",
        "theta_proj.bias",
    }
    forward_parameters = set(inspect.signature(head.forward).parameters)
    assert "item_state" not in forward_parameters
    assert "concept_basis" not in forward_parameters

    state = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    difficulty = torch.tensor([0.25, -0.40], requires_grad=True)
    discrimination = torch.tensor([1.20, 0.70], requires_grad=True)
    logits, details = head(
        knowledge_state=state,
        concept_mask=mask,
        b=difficulty,
        a=discrimination,
        return_details=True,
    )
    assert tuple(logits.shape) == (2,)
    expected_theta_c = head.theta_proj(state).squeeze(-1)
    expected_theta_e = (expected_theta_c * mask).sum(1) / mask.sum(1)
    expected_logits = discrimination * (expected_theta_e - difficulty)
    assert torch.allclose(logits, expected_logits, atol=0.0, rtol=0.0)
    assert torch.allclose(details["theta_c"], expected_theta_c.detach())
    assert torch.allclose(details["theta_e"], expected_theta_e.detach())
    assert torch.equal(details["difficulty_e"], difficulty.detach())
    assert set(details) == {"theta_c", "theta_e", "difficulty_e", "irt_logit"}

    logits.square().mean().backward()
    assert head.theta_proj.weight.grad is not None
    assert head.theta_proj.weight.grad.abs().sum() > 0
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert difficulty.grad is not None and difficulty.grad.abs().sum() > 0
    assert discrimination.grad is not None and discrimination.grad.abs().sum() > 0

    # A deterministic example makes the architecture invariant explicit:
    # concepts are Q-pooled first and scalar b/a are applied exactly once.
    deterministic = CognitiveDiagnosisHead(knowledge_dim=2, num_concepts=3)
    with torch.no_grad():
        deterministic.theta_proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
        deterministic.theta_proj.bias.zero_()
    deterministic_state = torch.tensor(
        [[[1.0, 9.0], [3.0, 8.0], [100.0, 7.0]]]
    )
    deterministic_logit = deterministic(
        knowledge_state=deterministic_state,
        concept_mask=torch.tensor([[1.0, 1.0, 0.0]]),
        b=torch.tensor([0.5]),
        a=torch.tensor([2.0]),
    )
    assert torch.equal(deterministic_logit, torch.tensor([3.0]))
    print("OK: Q-masked scalar-difficulty 2PL contracts passed.")


if __name__ == "__main__":
    main()
