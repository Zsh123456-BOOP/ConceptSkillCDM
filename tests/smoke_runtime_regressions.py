import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_model(*, ablate_module1: bool = False, use_personal_graph: bool = True):
    from src.model import CognitiveDiagnosisModel

    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    model = CognitiveDiagnosisModel(
        num_students=5,
        num_exercises=4,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=use_personal_graph,
        personal_rank=2,
        ablate_module1=ablate_module1,
        graph_tau_init=0.6,
        graph_topk=2,
        allow_self_loop=True,
    )
    model.eval()
    return model


def _check_per_head_personal_graph() -> None:
    model = _build_model(ablate_module1=False, use_personal_graph=True)
    student_ids = torch.tensor([0, 1], dtype=torch.long)
    exercise_ids = torch.tensor([0, 2], dtype=torch.long)

    with torch.no_grad():
        _, details = model(
            student_ids=student_ids,
            exercise_ids=exercise_ids,
            return_details=True,
            return_logits=True,
        )

    alpha = details.get("alpha")
    personal = details.get("personal_matrices")
    _assert(alpha is not None, "Expected per-head alpha details.")
    _assert(personal is not None, "Expected per-head personal matrices.")
    _assert(tuple(alpha.shape) == (2, 2, 1, 1), f"Expected alpha shape (2,2,1,1), got {tuple(alpha.shape)}.")
    _assert(
        tuple(personal.shape) == (2, 2, 3, 3),
        f"Expected personal_matrices shape (2,2,3,3), got {tuple(personal.shape)}.",
    )
    for key in ("personal_delta_pre_softmax_norm", "personal_delta_student_std", "alpha_head_std"):
        _assert(key in details, f"Expected detail key: {key}.")


def _check_removed_residual_artifacts() -> None:
    model = _build_model(ablate_module1=False, use_personal_graph=True)
    student_ids = torch.tensor([0, 1], dtype=torch.long)
    exercise_ids = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        _, details = model(
            student_ids=student_ids,
            exercise_ids=exercise_ids,
            return_details=True,
            return_logits=True,
        )

    for removed_key in ("mf_logit", "residual_logit", "gate", "gate_raw", "delta_logit"):
        _assert(removed_key not in details, f"Removed B detail should not appear: {removed_key}")
    _assert("irt_logit" in details, "Fixed prediction head should still expose irt_logit.")


def _check_get_student_diagnosis() -> None:
    for ablate_module1 in (False, True):
        model = _build_model(ablate_module1=ablate_module1, use_personal_graph=True)
        out = model.get_student_diagnosis(0)
        mastery = out.get("knowledge_mastery")
        student_repr = out.get("student_repr")
        _assert(mastery is not None, "knowledge_mastery should exist.")
        _assert(student_repr is not None, "student_repr should exist.")
        _assert(tuple(mastery.shape) == (3,), f"Unexpected mastery shape: {tuple(mastery.shape)}.")
        _assert(tuple(student_repr.shape) == (8,), f"Unexpected student_repr shape: {tuple(student_repr.shape)}.")
        _assert("skill_latent" not in out, "skill_latent should be removed with module B.")


def main() -> None:
    _check_per_head_personal_graph()
    _check_removed_residual_artifacts()
    _check_get_student_diagnosis()
    print("OK: runtime regression smoke checks passed.")


if __name__ == "__main__":
    main()
