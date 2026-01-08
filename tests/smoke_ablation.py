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
        skill_dim=8,
        exercise_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        num_prototypes=3,
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
    _print_shape("mf_logit", details.get("mf_logit"))
    _print_shape("gate", details.get("gate"))
    _print_shape("student_latent", details.get("student_latent"))
    _print_shape("exercise_latent", details.get("exercise_latent"))
    _print_shape("prototype_assign", details.get("prototype_assign"))


def main() -> None:
    cases = [
        ("full", dict(use_soft_prototype=True, use_mf_branch=True, use_concept_graph=True)),
        ("no_soft_proto", dict(use_soft_prototype=False, use_mf_branch=True, use_concept_graph=True)),
        ("no_skill", dict(use_soft_prototype=True, use_mf_branch=False, use_concept_graph=True)),
        ("no_concept_graph", dict(use_soft_prototype=True, use_mf_branch=True, use_concept_graph=False)),
    ]

    for name, kwargs in cases:
        run_case(name, **kwargs)


if __name__ == "__main__":
    main()
