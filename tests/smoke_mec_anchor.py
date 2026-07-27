"""Smoke checks for the graph-free masked evidence completion anchor."""

from __future__ import annotations

import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import build_student_concept_response_stats
from src.evidence_completion import MaskedEvidenceCompletion
from src.model import CognitiveDiagnosisModel


def _response_fixture():
    frame = pd.DataFrame(
        {
            "stu_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "exer_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "cpt_seq": ["0,1", "2", "3", "1", "0,1", "2", "3", "1"],
            "label": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        }
    )
    q_matrix = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    students = {0: 0, 1: 1}
    exercises = {value: value for value in range(4)}
    stats = build_student_concept_response_stats(
        frame,
        students,
        exercises,
        q_matrix,
    )
    return q_matrix, stats


def main() -> None:
    completion = MaskedEvidenceCompletion(
        num_concepts=4,
        global_response_count=100.0,
    )
    assert sum(parameter.numel() for parameter in completion.parameters()) == 401

    evidence = torch.tensor(
        [
            [[3.0, 0.8], [-2.0, -0.5], [1.0, 0.2], [-1.0, -0.3]],
            [[1.0, 0.1], [2.0, 0.4], [-3.0, -0.8], [0.5, 0.2]],
        ]
    )
    count = torch.tensor(
        [[4.0, 2.0, 3.0, 1.0], [1.0, 2.0, 4.0, 3.0]]
    )
    correct = torch.tensor(
        [[3.0, 0.0, 2.0, 0.0], [1.0, 2.0, 0.0, 2.0]]
    )
    q_mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )
    features = completion.build_features(evidence, count, correct, q_mask)
    changed_target = evidence.clone()
    changed_target[q_mask.bool()] *= -1.0
    changed_count = count.clone()
    changed_count[q_mask.bool()] += 7.0
    changed_correct = correct.clone()
    changed_correct[q_mask.bool()] = changed_count[q_mask.bool()]
    target_changed_features = completion.build_features(
        changed_target,
        changed_count,
        changed_correct,
        q_mask,
    )
    assert torch.equal(features, target_changed_features)

    no_source = completion(
        evidence,
        count,
        correct,
        torch.ones_like(q_mask),
    )
    assert torch.equal(no_source, torch.zeros_like(no_source))
    assert torch.equal(
        completion(evidence, count, correct, q_mask),
        torch.zeros_like(q_mask),
    ), "zero initialization must reproduce the matched baseline"

    with torch.no_grad():
        for layer in completion.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(0.1)
                layer.bias.zero_()
    changed_source = evidence.clone()
    changed_source[0, 2, 0] += 1.0
    anchor = completion(evidence, count, correct, q_mask)
    changed_anchor = completion(changed_source, count, correct, q_mask)
    assert not torch.equal(anchor, changed_anchor)
    assert torch.equal(anchor[q_mask == 0], torch.zeros_like(anchor[q_mask == 0]))

    q_matrix, stats = _response_fixture()
    common = dict(
        num_students=2,
        num_exercises=4,
        num_concepts=4,
        q_matrix=q_matrix,
        item_prior_matrix=torch.zeros(4, 4),
        exposure_prior_matrix=torch.zeros(4, 4),
        response_evidence_stats=stats,
        use_response_evidence=True,
        evidence_state_injection=False,
        disable_graph_module=True,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=2,
        dropout=0.0,
    )
    torch.manual_seed(17)
    baseline = CognitiveDiagnosisModel(
        **common,
        evidence_anchor_mode="direct_only",
    ).eval()
    baseline_rng_state = torch.random.get_rng_state().clone()
    torch.manual_seed(17)
    mec = CognitiveDiagnosisModel(
        **common,
        evidence_anchor_mode="mec",
    ).eval()
    mec_rng_state = torch.random.get_rng_state().clone()
    assert torch.equal(baseline_rng_state, mec_rng_state)
    assert baseline.relation_learning is None
    assert mec.relation_learning is None
    baseline_parameters = sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    mec_parameters = sum(parameter.numel() for parameter in mec.parameters())
    assert mec_parameters - baseline_parameters == 401 + q_matrix.size(1)

    student_ids = torch.tensor([0, 1], dtype=torch.long)
    exercise_ids = torch.tensor([0, 0], dtype=torch.long)
    labels = torch.tensor([1.0, 0.0])
    baseline_logits = baseline(
        student_ids,
        exercise_ids,
        outcome_to_exclude=labels,
        return_logits=True,
    )
    mec_logits, details = mec(
        student_ids,
        exercise_ids,
        outcome_to_exclude=labels,
        return_logits=True,
        return_details=True,
    )
    assert torch.equal(baseline_logits, mec_logits)
    assert torch.equal(details["logits"], details["irt_logit"])
    assert "mec_logit" not in details

    mec.eval()
    mec.zero_grad(set_to_none=True)
    train_logits = mec(
        student_ids,
        exercise_ids,
        outcome_to_exclude=labels,
        return_logits=True,
    )
    F.binary_cross_entropy_with_logits(train_logits, labels).backward()
    final = mec.evidence_completion.net[-1]
    assert final.weight.grad is not None
    assert float(final.weight.grad.abs().sum().item()) > 0.0

    graph_common = dict(common)
    graph_common["disable_graph_module"] = False
    try:
        CognitiveDiagnosisModel(
            **graph_common,
            evidence_anchor_mode="mec",
        )
    except ValueError as exc:
        assert "graph-free" in str(exc)
    else:
        raise AssertionError("MEC must reject a live graph module")
    print("OK: MEC is graph-free, Q-masked, lightweight, and single-logit.")


if __name__ == "__main__":
    main()
