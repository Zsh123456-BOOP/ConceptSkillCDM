"""Deterministic contracts for support-normalized GEC evidence transport."""

import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import CognitiveDiagnosisModel


def _response_stats() -> dict:
    count = torch.tensor(
        [
            [0.0, 3.0, 0.0],
            [1.0, 2.0, 1.0],
        ]
    )
    correct = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return {
        "student_concept_count": count,
        "student_concept_correct": correct,
        "student_concept_residual_sum": torch.zeros_like(count),
        "student_item_keys": torch.tensor([0, 1, 2, 3]),
        "student_item_expected_correct": torch.full((4,), 0.5),
        "concept_count": count.sum(dim=0),
        "concept_correct": correct.sum(dim=0),
        "global_count": torch.tensor(4.0),
        "global_correct": torch.tensor(2.0),
    }


def main() -> None:
    relation = torch.tensor(
        [
            [0.80, 0.02, 0.18],
            [0.30, 0.40, 0.30],
            [0.20, 0.30, 0.50],
        ]
    )

    # No off-diagonal source evidence must produce an exact finite zero even
    # when a high-value self-loop is present.
    no_source = CognitiveDiagnosisModel._support_normalized_propagation(
        torch.tensor([[4.0, 0.0, 0.0]]),
        torch.tensor([[5.0, 0.0, 0.0]]),
        relation,
    )
    assert torch.equal(no_source[0, 0], torch.tensor(0.0))
    assert torch.isfinite(no_source).all()

    # With a scarce target and one observed off-diagonal source, unavailable
    # graph neighbours must not dilute that source and self evidence is ignored.
    single_source = CognitiveDiagnosisModel._support_normalized_propagation(
        torch.tensor([[-4.0, 0.75, 0.0]]),
        torch.tensor([[0.0, 3.0, 0.0]]),
        relation,
    )
    assert torch.allclose(single_source[0, 0], torch.tensor(0.75), atol=1e-7)

    # The normalization correction returns smoothly toward the legacy weighted
    # sum as direct target evidence accumulates.
    target_counts = torch.tensor([0.0, 1.0, 4.0, 20.0])
    evidence = torch.tensor([[0.0, 0.75, 0.0]]).repeat(4, 1)
    counts = torch.stack(
        (target_counts, torch.full_like(target_counts, 3.0), torch.zeros_like(target_counts)),
        dim=1,
    )
    corrected = CognitiveDiagnosisModel._support_normalized_propagation(
        evidence,
        counts,
        relation,
    )[:, 0]
    assert torch.all(corrected[:-1] > corrected[1:])
    assert torch.all((corrected >= 0.0) & (corrected <= 0.75))

    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    model = CognitiveDiagnosisModel(
        num_students=2,
        num_exercises=2,
        num_concepts=3,
        q_matrix=q_matrix,
        item_prior_matrix=relation,
        exposure_prior_matrix=relation,
        response_evidence_stats=_response_stats(),
        use_response_evidence=True,
        evidence_propagation_mode="support_normalized",
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
    ).eval()
    logits, details = model(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        return_logits=True,
        return_details=True,
    )
    assert details["evidence_anchor"].shape == (2, 3, 3)
    assert torch.isfinite(details["evidence_anchor"]).all()
    assert torch.equal(logits, details["irt_logit"])
    assert torch.equal(details["logits"], details["irt_logit"])
    probabilities = model(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        return_logits=False,
    )
    assert torch.allclose(probabilities, torch.sigmoid(logits), atol=0.0, rtol=0.0)
    diagnosis = model.get_student_diagnosis(0)
    assert torch.isfinite(diagnosis["knowledge_mastery"]).all()

    # Omitting the new switch is exactly the explicit legacy mode, preserving
    # old command and checkpoint reconstruction semantics.
    legacy_kwargs = {
        "num_students": 2,
        "num_exercises": 2,
        "num_concepts": 3,
        "q_matrix": q_matrix,
        "item_prior_matrix": relation,
        "exposure_prior_matrix": relation,
        "response_evidence_stats": _response_stats(),
        "use_response_evidence": True,
        "knowledge_dim": 8,
        "num_relation_heads": 2,
        "num_gnn_layers": 1,
        "dropout": 0.0,
    }
    torch.manual_seed(19)
    implicit_legacy = CognitiveDiagnosisModel(**legacy_kwargs).eval()
    torch.manual_seed(19)
    explicit_legacy = CognitiveDiagnosisModel(
        **legacy_kwargs,
        evidence_propagation_mode="legacy",
    ).eval()
    explicit_legacy.load_state_dict(implicit_legacy.state_dict(), strict=True)
    implicit_logits = implicit_legacy(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        return_logits=True,
    )
    explicit_logits = explicit_legacy(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        return_logits=True,
    )
    assert torch.equal(implicit_logits, explicit_logits)
    print("OK: support-normalized evidence propagation contracts passed.")


if __name__ == "__main__":
    main()
