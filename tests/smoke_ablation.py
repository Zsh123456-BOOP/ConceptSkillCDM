import os
import sys
from typing import Optional

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit("torch is not installed. Install requirements to run this smoke test.") from exc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import CognitiveDiagnosisModel


def _make_q_matrix(num_exercises: int, num_concepts: int) -> torch.Tensor:
    q = torch.zeros(num_exercises, num_concepts)
    for e in range(num_exercises):
        q[e, e % num_concepts] = 1.0
    extra = (torch.rand(num_exercises, num_concepts) < 0.2).float()
    q = torch.clamp(q + extra, 0.0, 1.0)
    return q


def _print_shape(name: str, tensor: Optional[torch.Tensor]) -> None:
    if tensor is None:
        print(f"  {name}: None")
    else:
        print(f"  {name}: {tuple(tensor.shape)}")


def run_case(name: str, **model_kwargs) -> None:
    torch.manual_seed(123)

    num_students = 10
    num_exercises = 12
    num_concepts = 6
    q_matrix = _make_q_matrix(num_exercises, num_concepts)

    model = CognitiveDiagnosisModel(
        num_students=num_students,
        num_exercises=num_exercises,
        num_concepts=num_concepts,
        q_matrix=q_matrix,
        knowledge_dim=16,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        **model_kwargs,
    )
    model.eval()

    student_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    exercise_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    concept_vector = q_matrix[exercise_ids]

    logits, details = model(
        student_ids,
        exercise_ids,
        concept_vector=concept_vector,
        return_details=True,
        return_logits=True,
    )

    print(f"\n[{name}]")
    _print_shape("logits", logits)
    _print_shape("relation_matrices", details.get("relation_matrices"))
    _print_shape("knowledge_state", details.get("knowledge_state"))
    _print_shape("student_repr", details.get("student_repr"))
    _print_shape("q_vector", details.get("q_vector"))
    _print_shape("irt_logit", details.get("irt_logit"))
    _print_shape("theta_c", details.get("theta_c"))
    _print_shape("alpha", details.get("alpha"))

    for removed_key in ("mf_logit", "residual_logit", "gate", "student_latent", "exercise_latent"):
        if removed_key in details:
            raise AssertionError(f"{removed_key} should not exist after deleting module B.")


def main() -> None:
    cases = [
        ("full", dict(use_concept_graph=True, use_personal_graph=True)),
        ("no_A", dict(use_concept_graph=False, use_personal_graph=True)),
        ("no_E", dict(use_concept_graph=True, use_personal_graph=False)),
    ]

    for name, kwargs in cases:
        run_case(name, **kwargs)


if __name__ == "__main__":
    main()
