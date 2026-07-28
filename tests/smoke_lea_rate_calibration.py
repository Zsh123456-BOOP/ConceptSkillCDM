"""Contracts for the LEA rate-evidence calibration candidates."""

import os
import sys

import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import (
    build_id_mappings,
    build_item_cooccurrence_prior,
    build_q_matrix,
    build_student_concept_response_stats,
    build_student_coexposure_prior,
)
from src.model import CognitiveDiagnosisModel, GRAPH_IRT_ARCHITECTURE


def _structure(dataframe: pd.DataFrame):
    students, exercises, concepts = build_id_mappings([dataframe])
    q_matrix = build_q_matrix([dataframe], exercises, concepts)
    item_prior, _ = build_item_cooccurrence_prior(q_matrix)
    exposure_prior, _ = build_student_coexposure_prior([dataframe], concepts)
    stats = build_student_concept_response_stats(
        dataframe,
        students,
        exercises,
        q_matrix,
    )
    return (
        students,
        exercises,
        concepts,
        q_matrix,
        item_prior,
        exposure_prior,
        stats,
    )


def _build_model(
    structure,
    rate_evidence_mode: str,
    evidence_anchor_mode: str = "direct_only",
) -> CognitiveDiagnosisModel:
    students, exercises, concepts, q_matrix, item_prior, exposure_prior, stats = (
        structure
    )
    return CognitiveDiagnosisModel(
        num_students=len(students),
        num_exercises=len(exercises),
        num_concepts=len(concepts),
        q_matrix=q_matrix,
        item_prior_matrix=item_prior,
        exposure_prior_matrix=exposure_prior,
        response_evidence_stats=stats,
        use_response_evidence=True,
        evidence_anchor_mode=evidence_anchor_mode,
        evidence_state_injection=False,
        rate_evidence_mode=rate_evidence_mode,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        graph_propagation_alpha=0.20,
    )


def main() -> None:
    frame = pd.DataFrame(
        {
            "stu_id": [10, 10, 20, 20, 20],
            "exer_id": [100, 100, 100, 101, 102],
            "cpt_seq": ["1", "1", "1", "2", "1,2"],
            "label": [1, 0, 0, 1, 1],
        }
    )
    structure = _structure(frame)
    students, exercises, concepts, q_matrix, _, _, _ = structure

    torch.manual_seed(71)
    baseline = _build_model(structure, "reliability_scaled").eval()
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(71)
    candidate = _build_model(structure, "posterior_gap").eval()
    candidate_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(71)
    bounded = _build_model(structure, "bounded_low_count_v1").eval()
    bounded_rng = torch.random.get_rng_state().clone()

    # Both interventions reuse the same parameter structure and RNG trajectory.
    assert torch.equal(baseline_rng, candidate_rng)
    assert torch.equal(baseline_rng, bounded_rng)
    assert tuple(baseline.state_dict()) == tuple(candidate.state_dict())
    assert tuple(baseline.state_dict()) == tuple(bounded.state_dict())
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
        if key != "anchor_gate":
            assert torch.equal(value, bounded.state_dict()[key]), key
    assert sum(parameter.numel() for parameter in baseline.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    assert sum(parameter.numel() for parameter in baseline.parameters()) == sum(
        parameter.numel() for parameter in bounded.parameters()
    )
    assert baseline.relation_learning is not None
    assert candidate.relation_learning is not None
    assert bounded.relation_learning is not None
    try:
        _build_model(
            structure,
            "bounded_low_count_v1",
            evidence_anchor_mode="full",
        )
    except ValueError as exc:
        assert "requires evidence_anchor_mode='direct_only'" in str(exc)
    else:
        raise AssertionError("bounded rate calibration must stay direct-only")
    bounded_reload = _build_model(structure, "bounded_low_count_v1").eval()
    incompatible = bounded_reload.load_state_dict(
        bounded.state_dict(),
        strict=True,
    )
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

    student_ids = torch.tensor([students[10]], dtype=torch.long)
    exercise_ids = torch.tensor([exercises[100]], dtype=torch.long)
    q_vector = torch.ones((1, len(concepts)), dtype=q_matrix.dtype)
    baseline_evidence, count, _, _, _ = baseline._build_response_evidence(
        student_ids,
        exercise_ids,
        q_vector,
        None,
        None,
    )
    candidate_evidence, candidate_count, _, _, _ = (
        candidate._build_response_evidence(
            student_ids,
            exercise_ids,
            q_vector,
            None,
            None,
        )
    )
    bounded_evidence, bounded_count, _, _, _ = bounded._build_response_evidence(
        student_ids,
        exercise_ids,
        q_vector,
        None,
        None,
    )
    assert torch.equal(count, candidate_count)
    assert torch.equal(count, bounded_count)

    no_support = count == 0
    with_support = count > 0
    assert no_support.any() and with_support.any()
    assert torch.equal(
        candidate_evidence[..., 0][no_support],
        baseline_evidence[..., 0][no_support],
    )
    reliability = count / (count + 1.0)
    assert torch.allclose(
        candidate_evidence[..., 0][with_support],
        baseline_evidence[..., 0][with_support] / reliability[with_support],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(
        candidate_evidence[..., 1],
        baseline_evidence[..., 1],
    )
    assert torch.equal(bounded_evidence, candidate_evidence)

    # The bounded gate is the only rate-channel count transform: it stays
    # between the legacy reliability and the raw posterior gap, explicitly
    # abstains at n=0, and concentrates its correction at n=1.
    synthetic_count = torch.tensor([[0.0, 1.0, 2.0, 3.0, 8.0]])
    synthetic_evidence = torch.zeros((1, 5, 2))
    synthetic_evidence[..., 0] = 1.0
    synthetic_relations = torch.eye(5).unsqueeze(0).repeat(2, 1, 1)
    bounded_anchor = bounded._compose_evidence_anchor(
        synthetic_relations,
        synthetic_evidence,
        synthetic_count,
    )
    assert bounded_anchor is not None
    bounded_scale = bounded_anchor[..., 0]
    reliability_scale = synthetic_count / (synthetic_count + 1.0)
    alpha = bounded.anchor_gate[0, 0]
    decay = torch.nn.functional.softplus(bounded.anchor_gate[0, 1])
    low_count_gate = torch.sigmoid(
        alpha - decay * torch.log1p(synthetic_count)
    )
    expected_scale = reliability_scale + (
        1.0 - reliability_scale
    ) * low_count_gate
    expected_scale = expected_scale * (synthetic_count > 0)
    assert torch.allclose(bounded_scale, expected_scale, atol=1e-7, rtol=0.0)
    assert bounded_scale[0, 0].item() == 0.0
    assert torch.all(
        bounded_scale[synthetic_count > 0]
        >= reliability_scale[synthetic_count > 0]
    )
    assert torch.all(bounded_scale <= 1.0)
    assert torch.allclose(
        low_count_gate[0, 1:3],
        torch.tensor([0.8, 0.1]),
        atol=1e-6,
        rtol=0.0,
    )
    correction = (
        expected_scale - reliability_scale
    )[synthetic_count > 0]
    assert torch.all(correction[1:] < correction[:-1])
    bounded.zero_grad(set_to_none=True)
    bounded_anchor.sum().backward()
    assert torch.isfinite(bounded.anchor_gate.grad).all()

    logits, details = bounded(
        student_ids,
        exercise_ids,
        return_details=True,
        return_logits=True,
    )
    assert torch.equal(logits, details["irt_logit"])
    assert torch.equal(details["logits"], details["irt_logit"])
    assert details["knowledge_state_graph_delta"].item() > 0.0

    # Flipping a training target and excluding that same row remains isolated.
    flipped = frame.copy()
    flipped.loc[0, "label"] = 1 - flipped.loc[0, "label"]
    flipped_structure = _structure(flipped)
    torch.manual_seed(93)
    original_model = _build_model(structure, "bounded_low_count_v1").eval()
    torch.manual_seed(93)
    flipped_model = _build_model(
        flipped_structure,
        "bounded_low_count_v1",
    ).eval()
    original_logits, original_details = original_model(
        student_ids,
        exercise_ids,
        outcome_to_exclude=torch.tensor([frame.loc[0, "label"]]),
        return_details=True,
        return_logits=True,
    )
    flipped_logits, flipped_details = flipped_model(
        student_ids,
        exercise_ids,
        outcome_to_exclude=torch.tensor([flipped.loc[0, "label"]]),
        return_details=True,
        return_logits=True,
    )
    assert torch.allclose(original_logits, flipped_logits, atol=1e-7, rtol=0.0)
    assert torch.allclose(
        original_details["response_evidence"],
        flipped_details["response_evidence"],
        atol=1e-7,
        rtol=0.0,
    )
    assert GRAPH_IRT_ARCHITECTURE == "graph_irt_v10"
    print("OK: LEA rate calibration contracts passed.")


if __name__ == "__main__":
    main()
