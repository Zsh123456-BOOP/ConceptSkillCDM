"""Smoke checks for fixed, cross-fitted residual relation transport."""

import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model_cdm import CognitiveDiagnosisModel


NUM_STUDENTS = 6
NUM_EXERCISES = 3
NUM_CONCEPTS = 3


def _response_stats() -> dict:
    count = torch.tensor(
        [
            [0.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 1.0],
            [3.0, 1.0, 2.0],
            [1.0, 4.0, 3.0],
            [4.0, 2.0, 1.0],
        ]
    )
    correct = torch.floor(count * 0.6)
    residual_sum = torch.tensor(
        [
            [0.0, 0.4, -0.6],
            [0.3, -0.2, 0.8],
            [-0.4, 0.6, 0.2],
            [0.7, -0.3, 0.5],
            [-0.2, 0.9, -0.7],
            [0.8, -0.4, 0.3],
        ]
    )
    return {
        "student_concept_count": count,
        "student_concept_correct": correct,
        "student_concept_residual_sum": residual_sum,
        "student_item_keys": torch.arange(
            NUM_STUDENTS * NUM_EXERCISES,
            dtype=torch.long,
        ),
        "student_item_expected_correct": torch.full(
            (NUM_STUDENTS * NUM_EXERCISES,),
            0.5,
        ),
        "concept_count": count.sum(dim=0),
        "concept_correct": correct.sum(dim=0),
        "global_count": torch.tensor(18.0),
        "global_correct": torch.tensor(10.0),
    }


def _bundle() -> dict:
    full = torch.zeros(NUM_CONCEPTS, NUM_CONCEPTS)
    full[0, 1] = 1.0
    full[1, 0] = -0.5
    folds = torch.zeros(5, NUM_CONCEPTS, NUM_CONCEPTS)
    for fold in range(5):
        folds[fold, 0, 1] = float(fold + 1) / 5.0
        folds[fold, 1, 0] = -float(fold + 1) / 10.0
    return {
        "full_relation": full,
        "fold_relations": folds,
        "student_fold": torch.tensor([0, 1, 2, 3, 4, 0]),
    }


def _model(
    *,
    mode: str = "off",
    bundle: dict | None = None,
) -> CognitiveDiagnosisModel:
    q = torch.eye(NUM_CONCEPTS)
    return CognitiveDiagnosisModel(
        num_students=NUM_STUDENTS,
        num_exercises=NUM_EXERCISES,
        num_concepts=NUM_CONCEPTS,
        q_matrix=q,
        response_evidence_stats=_response_stats(),
        residual_relation_mode=mode,
        residual_relation_bundle=bundle,
        use_response_evidence=True,
        evidence_anchor_mode="direct_only",
        evidence_state_injection=True,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
    )


def _assert_transport_math() -> None:
    model = _model(mode="partial_topk", bundle=_bundle())
    evidence = torch.zeros(2, NUM_CONCEPTS, 2)
    evidence[:, :, 1] = torch.tensor([0.4, 0.8, 0.0])
    count = torch.tensor([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
    adjusted, transport = model._apply_residual_relation_transport(
        evidence,
        count,
        torch.tensor([0, 1]),
        scope="full",
    )
    expected_transport = torch.tensor([[0.8, -0.2, 0.0]]).repeat(2, 1)
    assert torch.allclose(transport, expected_transport)
    assert torch.allclose(adjusted[0, :, 1], torch.tensor([1.0, 0.6, 0.0]))
    assert torch.allclose(adjusted[1, :, 1], torch.tensor([0.6, 0.75, 0.0]))
    delta_low = adjusted[0, 1, 1] - evidence[0, 1, 1]
    delta_high = adjusted[1, 1, 1] - evidence[1, 1, 1]
    assert torch.allclose(delta_low.abs(), 4.0 * delta_high.abs())


def _assert_mixed_fold_transport() -> None:
    model = _model(mode="partial_topk", bundle=_bundle())
    evidence = torch.zeros(5, NUM_CONCEPTS, 2)
    evidence[:, 1, 1] = 0.1
    count = torch.zeros(5, NUM_CONCEPTS)
    students = torch.tensor([4, 0, 3, 1, 2])
    adjusted, transport = model._apply_residual_relation_transport(
        evidence,
        count,
        students,
        scope="student_oof",
    )
    expected = torch.tensor([0.1, 0.02, 0.08, 0.04, 0.06])
    assert torch.allclose(transport[:, 0], expected)
    assert torch.allclose(adjusted[:, 0, 1], expected)


def _assert_no_architecture_change() -> None:
    torch.manual_seed(17)
    off = _model(mode="off")
    torch.manual_seed(17)
    candidate = _model(mode="partial_topk", bundle=_bundle())
    assert tuple(off.state_dict()) == tuple(candidate.state_dict())
    assert tuple(dict(off.named_parameters())) == tuple(dict(candidate.named_parameters()))
    for key, value in off.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key

    explicit_off = _model(mode="off")
    explicit_off.load_state_dict(off.state_dict(), strict=True)
    off.eval()
    explicit_off.eval()
    students = torch.tensor([0, 1, 4, 5])
    exercises = torch.tensor([0, 1, 2, 0])
    baseline = off(students, exercises, return_logits=True)
    actual = explicit_off(students, exercises, return_logits=True)
    assert torch.equal(baseline, actual)

    candidate.load_state_dict(off.state_dict(), strict=True)
    candidate.eval()
    _, off_details = off(
        students,
        exercises,
        return_details=True,
        return_logits=True,
    )
    _, candidate_details = candidate(
        students,
        exercises,
        return_details=True,
        return_logits=True,
    )
    assert torch.equal(
        off_details["knowledge_state"],
        candidate_details["knowledge_state"],
    )
    assert torch.equal(
        off_details["response_evidence"],
        candidate_details["response_evidence"],
    )
    assert not torch.equal(
        candidate_details["response_evidence"],
        candidate_details["adjusted_response_evidence"],
    )
    diagnosis = candidate.get_student_diagnosis(0)
    assert torch.isfinite(diagnosis["knowledge_mastery"]).all()


def _assert_fail_closed() -> None:
    try:
        _model(mode="partial_topk", bundle=None)
    except ValueError as error:
        assert "bundle" in str(error)
    else:
        raise AssertionError("missing relation bundle was accepted")

    model = _model(mode="partial_topk", bundle=_bundle())
    students = torch.tensor([0, 1])
    exercises = torch.tensor([0, 1])
    labels = torch.tensor([1.0, 0.0])
    model.train()
    try:
        model(
            students,
            exercises,
            outcome_to_exclude=labels,
            return_logits=True,
        )
    except ValueError as error:
        assert "scope" in str(error)
    else:
        raise AssertionError("training query without student_oof scope was accepted")

    try:
        model(
            students,
            exercises,
            residual_relation_scope="student_oof",
            return_logits=True,
        )
    except ValueError as error:
        assert "outcome_to_exclude" in str(error)
    else:
        raise AssertionError("student_oof training without excluded outcomes was accepted")

    try:
        model(
            students,
            exercises,
            outcome_to_neutralize=labels,
            residual_relation_scope="student_oof",
            return_logits=True,
        )
    except ValueError as error:
        assert "only excluded" in str(error)
    else:
        raise AssertionError("neutralized relation training was accepted")

    model.eval()
    try:
        model(
            students,
            exercises,
            outcome_to_exclude=labels,
            residual_relation_scope="full",
            return_logits=True,
        )
    except ValueError as error:
        assert "student_oof" in str(error)
    else:
        raise AssertionError("label-adjusted query with full relation was accepted")

    try:
        model(
            students,
            exercises,
            residual_relation_scope="not_a_scope",
            return_logits=True,
        )
    except ValueError as error:
        assert "student_oof" in str(error) or "'full'" in str(error)
    else:
        raise AssertionError("invalid residual relation scope was accepted")


def main() -> None:
    _assert_transport_math()
    _assert_mixed_fold_transport()
    _assert_no_architecture_change()
    _assert_fail_closed()
    print("OK: residual relation transport invariants passed.")


if __name__ == "__main__":
    main()
