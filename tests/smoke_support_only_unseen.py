"""Smoke test for the support_only_unseen Path-2 method change.

Verifies, on a single set of weights:
  1. Off-equivalence: turning the flag on while NO query row is unseen yields
     identical logits to the flag-off baseline (the substitution is gated).
  2. Load-bearing: when some query rows ARE unseen, the flag changes predictions
     and they stay finite -> the LCRF support aggregation now drives those rows.
  3. The constructor accepts the new kwargs.

Run on a torch-enabled host:  python tests/smoke_support_only_unseen.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.smoke_interpretable_ae import _assert, _build_small_model


def _forward_logits(model) -> torch.Tensor:
    with torch.no_grad():
        logits = model(
            student_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            exercise_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            return_details=False,
            return_logits=True,
        )
    return logits


def _check_off_equivalence_and_load_bearing() -> None:
    model = _build_small_model()  # observed_counts all = 4.0 (no unseen rows)
    logits_off = _forward_logits(model)
    _assert(torch.isfinite(logits_off).all().item(), "baseline logits should be finite.")

    # (1) flag on but no unseen rows -> must be identical to off.
    model.support_only_unseen = True
    model.support_only_unseen_strength = 1.0
    logits_on_nounseen = _forward_logits(model)
    _assert(
        torch.allclose(logits_off, logits_on_nounseen, atol=1e-6),
        "support_only_unseen with no unseen rows must equal the baseline.",
    )

    # (2) introduce unseen (student, concept) by zeroing observed counts, so some
    # query rows become direct-unseen and the substitution activates.
    model.ae_student_concept_observed_count.zero_()
    logits_on_unseen = _forward_logits(model)
    _assert(torch.isfinite(logits_on_unseen).all().item(), "unseen-substituted logits should be finite.")
    _assert(
        not torch.allclose(logits_off, logits_on_unseen, atol=1e-6),
        "with unseen rows, support_only_unseen must change predictions (support is load-bearing).",
    )


def _check_constructor_accepts_kwargs() -> None:
    from src.model import CognitiveDiagnosisModel

    q_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    model = CognitiveDiagnosisModel(
        num_students=3,
        num_exercises=2,
        num_concepts=2,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=True,
        graph_prior_logit_scale=0.5,
        graph_topk=1,
        support_only_unseen=True,
        support_only_unseen_strength=0.5,
    )
    _assert(model.support_only_unseen is True, "constructor should store support_only_unseen.")
    _assert(abs(model.support_only_unseen_strength - 0.5) < 1e-8, "constructor should store strength.")


def main() -> None:
    _check_constructor_accepts_kwargs()
    _check_off_equivalence_and_load_bearing()
    print("OK: support_only_unseen smoke checks passed.")


if __name__ == "__main__":
    main()
