import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_prediction_head_module() -> None:
    from src.prediction_head import CognitiveDiagnosisHead, ExerciseDifficultyEncoder

    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    knowledge_state = torch.randn(3, 3, 8)
    concept_mask = q_matrix.clone()

    encoder = ExerciseDifficultyEncoder(num_exercises=3)
    b, a = encoder(torch.tensor([0, 1, 2], dtype=torch.long))
    _assert(tuple(b.shape) == (3,), f"unexpected difficulty shape: {tuple(b.shape)}")
    _assert(tuple(a.shape) == (3,), f"unexpected discrimination shape: {tuple(a.shape)}")
    _assert(float(a.min()) > 0.0, "discrimination should stay positive after softplus")

    head = CognitiveDiagnosisHead(knowledge_dim=8, num_concepts=3)
    _assert(
        not any("parametrizations" in name for name, _ in head.named_parameters()),
        "prediction head should use a plain interpretable linear theta projection.",
    )
    logits, details = head(
        knowledge_state=knowledge_state,
        concept_mask=concept_mask,
        b=b,
        a=a,
        return_details=True,
    )
    _assert(tuple(logits.shape) == (3,), f"unexpected logit shape: {tuple(logits.shape)}")
    for key in ("theta_c", "theta_e_base", "item_matching", "theta_e", "irt_logit"):
        _assert(key in details, f"missing prediction-head detail: {key}")
    _assert(
        torch.equal(details["item_matching"], torch.zeros_like(details["item_matching"])),
        "zero-initialized item directions must make the initial model exactly shared-2PL",
    )

    loss = logits.pow(2).mean()
    loss.backward()
    for name, param in head.named_parameters():
        if name == "item_matching_projection.weight":
            # The projection receives gradients after the initially-zero item
            # directions take their first optimizer step.
            continue
        _assert(param.grad is not None, f"missing grad for {name}")
        _assert(torch.isfinite(param.grad).all().item(), f"non-finite grad in {name}")
    _assert(
        head.concept_matching_direction.weight.grad.abs().sum().item() > 0.0,
        "Q-concept directions must leave zero initialization on the first step",
    )

    mean_head = CognitiveDiagnosisHead(
        knowledge_dim=2,
        num_concepts=3,
        enable_item_matching=False,
    )
    with torch.no_grad():
        mean_head.theta_proj.weight.zero_()
        mean_head.theta_proj.weight[0, 0] = 1.0
        mean_head.theta_proj.bias.zero_()
    known_state = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [0.0, 0.0], [5.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    known_mask = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mean_logits, mean_details = mean_head(
        knowledge_state=known_state,
        concept_mask=known_mask,
        b=torch.zeros(2),
        a=torch.ones(2),
        return_details=True,
    )
    _assert(torch.allclose(mean_logits[0], torch.tensor(2.0), atol=1e-6), "single-concept item should use its concept theta.")
    _assert(torch.allclose(mean_logits[1], torch.tensor(1.0), atol=1e-6), "multi-concept item should use the Q-mask mean theta.")
    _assert("theta_gap" not in mean_details, "prediction head should not keep the removed multi-concept gap path.")

    matching_head = CognitiveDiagnosisHead(knowledge_dim=2, num_concepts=3)
    with torch.no_grad():
        matching_head.theta_proj.weight.zero_()
        matching_head.theta_proj.bias.zero_()
        matching_head.item_matching_projection.weight.copy_(torch.eye(2))
        matching_head.concept_matching_direction.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        )
    repeated_state = known_state[:1].expand(2, -1, -1).clone()
    different_masks = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    _, matching_details = matching_head(
        knowledge_state=repeated_state,
        concept_mask=different_masks,
        b=torch.zeros(2),
        a=torch.ones(2),
        return_details=True,
    )
    _assert(
        not torch.equal(matching_details["theta_e"][0], matching_details["theta_e"][1]),
        "different Q-concept directions must read the same student state differently",
    )


def main() -> None:
    _check_prediction_head_module()
    print("OK: prediction head smoke checks passed.")


if __name__ == "__main__":
    main()
