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


def main() -> None:
    _check_prediction_head_module()
    print("OK: prediction head smoke checks passed.")


if __name__ == "__main__":
    main()
