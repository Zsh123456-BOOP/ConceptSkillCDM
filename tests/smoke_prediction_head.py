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

    head = CognitiveDiagnosisHead(knowledge_dim=8)
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
    for key in ("theta_c", "theta_e", "irt_logit"):
        _assert(key in details, f"missing prediction-head detail: {key}")

    loss = logits.pow(2).mean()
    loss.backward()
    for name, param in head.named_parameters():
        _assert(param.grad is not None, f"missing grad for {name}")
        _assert(torch.isfinite(param.grad).all().item(), f"non-finite grad in {name}")

    gap_head = CognitiveDiagnosisHead(knowledge_dim=2, concept_gap_scale=0.5)
    with torch.no_grad():
        gap_head.theta_proj.weight.zero_()
        gap_head.theta_proj.weight[0, 0] = 1.0
        gap_head.theta_proj.bias.zero_()
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
    gap_logits, gap_details = gap_head(
        knowledge_state=known_state,
        concept_mask=known_mask,
        b=torch.zeros(2),
        a=torch.ones(2),
        return_details=True,
    )
    _assert(torch.allclose(gap_details["theta_gap"][0], torch.tensor(0.0), atol=1e-6), "single-concept gap must be zero.")
    _assert(torch.allclose(gap_logits[0], torch.tensor(2.0), atol=1e-6), "single-concept item must stay unchanged.")
    _assert(gap_logits[1].item() < gap_details["theta_mean"][1].item(), "multi-concept gap should penalize dispersed theta.")
    _assert(torch.allclose(gap_details["concept_gap_scale"], torch.tensor(0.5), atol=1e-6), "gap scale should be exposed.")


def main() -> None:
    _check_prediction_head_module()
    print("OK: prediction head smoke checks passed.")


if __name__ == "__main__":
    main()
