"""Focused invariants for reliability-routed GEC-v2."""

from __future__ import annotations

import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import (  # noqa: E402
    build_id_mappings,
    build_item_cooccurrence_prior,
    build_q_matrix,
    build_student_concept_response_stats,
    build_student_coexposure_prior,
)
from src.model import CognitiveDiagnosisModel  # noqa: E402
from src.model_graph import ReliabilityAwareEvidenceRouter  # noqa: E402


def _assert_router_math() -> None:
    router = ReliabilityAwareEvidenceRouter(num_heads=2)
    target = torch.tensor([[0.0, 1.0, 4.0]])
    support = torch.tensor([[2.0, 2.0, 2.0]])
    conflict = torch.zeros_like(target)
    gate = router.evidence_gate(target, support, conflict)
    assert gate[0, 0] > gate[0, 1] > gate[0, 2]

    increasing_support = router.evidence_gate(
        torch.zeros(1, 3),
        torch.tensor([[0.0, 1.0, 4.0]]),
        torch.zeros(1, 3),
    )
    assert increasing_support[0, 0] < increasing_support[0, 1] < increasing_support[0, 2]
    increasing_conflict = router.evidence_gate(
        torch.zeros(1, 3),
        torch.ones(1, 3),
        torch.tensor([[0.0, 1.0, 4.0]]),
    )
    assert (
        increasing_conflict[0, 0]
        > increasing_conflict[0, 1]
        > increasing_conflict[0, 2]
    )

    relation = torch.tensor(
        [
            [[0.8, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.6, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    offdiag = router.off_diagonal_relations(relation)
    assert torch.equal(
        torch.diagonal(offdiag, dim1=-2, dim2=-1),
        torch.zeros(2, 3),
    )
    assert torch.equal(offdiag[:, 1], torch.zeros(2, 3))
    assert torch.equal(offdiag[:, 2], torch.zeros(2, 3))

    count = torch.tensor([[0.0, 4.0, 0.0]])
    correct = torch.tensor([[0.0, 4.0, 0.0]])
    concept_rate = torch.full_like(count, 0.5)
    propagated, routed_gate, routes, details = router(
        relation,
        count=count,
        correct=correct,
        concept_rate=concept_rate,
        direct_probability_delta=torch.zeros_like(count),
        state_enabled=True,
        evidence_enabled=True,
    )
    assert propagated[0, 0] > 0.0
    assert routed_gate[0, 0] > 0.0
    assert propagated[0, 1].item() == 0.0
    assert propagated[0, 2].item() == 0.0
    assert routed_gate[0, 1].item() == 0.0
    assert routed_gate[0, 2].item() == 0.0
    assert torch.allclose(
        details["head_weights"][:, :, 0].sum(dim=1),
        torch.ones(1),
        atol=1e-6,
    )
    assert (routes.sum(dim=-1) <= 1.0 + 1e-7).all()
    assert routes[0, 1, 1].item() == 0.0

    single_relation = torch.ones(2, 1, 1)
    single_count = torch.zeros(1, 1)
    single_propagated, single_gate, single_routes, single_details = router(
        single_relation,
        count=single_count,
        correct=single_count,
        concept_rate=torch.full_like(single_count, 0.5),
        direct_probability_delta=single_count,
        state_enabled=True,
        evidence_enabled=True,
    )
    assert torch.equal(single_propagated, torch.zeros_like(single_propagated))
    assert torch.equal(single_gate, torch.zeros_like(single_gate))
    assert torch.equal(single_routes[..., 1], torch.zeros_like(single_routes[..., 1]))
    assert torch.isfinite(single_details["head_weights"]).all()


def _structure(frame: pd.DataFrame):
    students, exercises, concepts = build_id_mappings([frame])
    q_matrix = build_q_matrix([frame], exercises, concepts)
    item_prior, _ = build_item_cooccurrence_prior(q_matrix)
    exposure_prior, _ = build_student_coexposure_prior([frame], concepts)
    stats = build_student_concept_response_stats(
        frame,
        students,
        exercises,
        q_matrix,
    )
    return students, exercises, concepts, q_matrix, item_prior, exposure_prior, stats


def _assert_model_boundary() -> None:
    frame = pd.DataFrame(
        {
            "stu_id": [10, 10, 10, 20, 20, 20],
            "exer_id": [100, 100, 102, 100, 102, 101],
            "cpt_seq": ["1", "1", "2,3", "1", "2,3", "1,2"],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "label": [0, 1, 0, 1, 1, 0],
        }
    )
    flipped = frame.copy()
    flipped.loc[2, "label"] = 1
    first = _structure(frame)
    second = _structure(flipped)
    students, exercises, concepts, q_matrix, item_prior, exposure_prior, stats_a = first
    stats_b = second[-1]
    kwargs = dict(
        num_students=len(students),
        num_exercises=len(exercises),
        num_concepts=len(concepts),
        q_matrix=q_matrix,
        item_prior_matrix=item_prior,
        exposure_prior_matrix=exposure_prior,
        use_response_evidence=True,
        evidence_anchor_mode="full",
        evidence_state_injection=False,
        gec_mode="reliability_v2",
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=2,
        dropout=0.0,
        graph_propagation_alpha=0.2,
    )
    torch.manual_seed(71)
    model_a = CognitiveDiagnosisModel(
        **kwargs,
        response_evidence_stats=stats_a,
    ).eval()
    torch.manual_seed(71)
    model_b = CognitiveDiagnosisModel(
        **kwargs,
        response_evidence_stats=stats_b,
    ).eval()
    student_ids = torch.tensor([students[10]])
    exercise_ids = torch.tensor([exercises[102]])
    label_a = torch.tensor([float(frame.loc[2, "label"])])
    label_b = torch.tensor([float(flipped.loc[2, "label"])])
    logits_a, details_a = model_a(
        student_ids,
        exercise_ids,
        outcome_to_exclude=label_a,
        return_details=True,
        return_logits=True,
    )
    logits_b, details_b = model_b(
        student_ids,
        exercise_ids,
        outcome_to_exclude=label_b,
        return_details=True,
        return_logits=True,
    )
    assert torch.allclose(logits_a, logits_b, atol=1e-7, rtol=0.0)
    for key in (
        "response_evidence",
        "gec_propagated_logit",
        "gec_relation_weighted_support",
        "gec_conflict",
        "gec_evidence_gate",
        "gec_state_route",
        "gec_evidence_route",
    ):
        assert torch.allclose(details_a[key], details_b[key], atol=1e-7, rtol=0.0), key
    diagonal = torch.diagonal(
        details_a["gec_offdiag_relations"],
        dim1=-2,
        dim2=-1,
    )
    assert torch.equal(diagonal, torch.zeros_like(diagonal))

    train_model = model_a.train()
    train_logits = train_model(
        student_ids,
        exercise_ids,
        outcome_to_exclude=label_a,
        return_logits=True,
    )
    loss = F.binary_cross_entropy_with_logits(train_logits, label_a)
    _, training_details = train_model(
        student_ids,
        exercise_ids,
        outcome_to_exclude=label_a,
        return_details=True,
        return_logits=True,
    )
    regularization = train_model.get_regularization_components(
        training_details["relation_matrices"],
        details=training_details,
        base_loss=loss,
    )
    assert torch.isfinite(regularization["total"])
    assert "evidence_graph_entropy_norm" in training_details
    loss = loss + regularization["total"]
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in train_model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    evidence_graph_gradients = [
        parameter.grad
        for parameter in train_model.evidence_relation_learning.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert evidence_graph_gradients
    assert any(
        gradient.abs().sum().item() > 0.0
        for gradient in evidence_graph_gradients
    )

    clone = CognitiveDiagnosisModel(
        **kwargs,
        response_evidence_stats=stats_a,
    ).eval()
    clone.load_state_dict(model_a.state_dict(), strict=True)
    model_a.eval()
    assert torch.equal(
        model_a(student_ids, exercise_ids, return_logits=True),
        clone(student_ids, exercise_ids, return_logits=True),
    )


def _assert_sparse_shape_path() -> None:
    concepts = 196
    router = ReliabilityAwareEvidenceRouter(num_heads=2)
    relation = torch.zeros(2, concepts, concepts, requires_grad=True)
    with torch.no_grad():
        for head in range(2):
            indices = torch.arange(concepts)
            relation[head, indices, (indices + head + 1) % concepts] = 0.7
            relation[head, indices, (indices + head + 2) % concepts] = 0.3
    count = torch.randint(0, 4, (2, concepts)).float()
    correct = torch.minimum(count, torch.randint(0, 4, (2, concepts)).float())
    rate = torch.full_like(count, 0.5)
    propagated, gate, routes, _ = router(
        relation,
        count=count,
        correct=correct,
        concept_rate=rate,
        direct_probability_delta=torch.zeros_like(count),
        state_enabled=True,
        evidence_enabled=True,
    )
    assert propagated.shape == gate.shape == (2, concepts)
    assert routes.shape == (2, concepts, 2)
    (propagated.square().mean() + gate.mean() + routes.mean()).backward()
    assert relation.grad is not None and torch.isfinite(relation.grad).all()
    assert relation.grad.abs().sum().item() > 0.0


def main() -> None:
    _assert_router_math()
    _assert_model_boundary()
    _assert_sparse_shape_path()
    print("OK: reliability-routed GEC-v2 invariants passed.")


if __name__ == "__main__":
    main()
