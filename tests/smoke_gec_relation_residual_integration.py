"""Integration checks for the v1-compatible relation residual contract."""

import os
import sys
import tempfile
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn.functional as F


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
from src.trainer import (
    _build_optimizer,
    _load_residual_warm_start,
    _parameter_sha256,
)


RESIDUAL_MODE = "relation_residual_v4"
RESIDUAL_PREFIX = "evidence_residual."


def _inputs(frame: pd.DataFrame):
    students, exercises, concepts = build_id_mappings([frame])
    q_matrix = build_q_matrix([frame], exercises, concepts)
    item_prior, _, item_support = build_item_cooccurrence_prior(
        q_matrix,
        return_support=True,
    )
    exposure_prior, _, exposure_support = build_student_coexposure_prior(
        [frame],
        concepts,
        return_support=True,
    )
    stats = build_student_concept_response_stats(
        frame,
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
        item_support,
        exposure_support,
        stats,
    )


def _model(inputs, *, gec_mode: str) -> CognitiveDiagnosisModel:
    (
        students,
        exercises,
        concepts,
        q_matrix,
        item_prior,
        exposure_prior,
        item_support,
        exposure_support,
        stats,
    ) = inputs
    return CognitiveDiagnosisModel(
        num_students=len(students),
        num_exercises=len(exercises),
        num_concepts=len(concepts),
        q_matrix=q_matrix,
        item_prior_matrix=item_prior,
        exposure_prior_matrix=exposure_prior,
        item_support_matrix=item_support,
        exposure_support_matrix=exposure_support,
        response_evidence_stats=stats,
        use_response_evidence=True,
        evidence_anchor_mode="full",
        evidence_state_injection=True,
        anchor_multihead_prop=True,
        gec_mode=gec_mode,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
    )


def _paired_args(checkpoint_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name="demo",
        seed=42,
        train_evidence_mode="excluded",
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        graph_topk=None,
        disable_self_loop=False,
        gnn_residual_weight=0.5,
        graph_identity_residual=0.0,
        graph_propagation_alpha=0.2,
        graph_prior_strength_init=1.0,
        graph_tau_init=1.0,
        graph_dropout=-1.0,
        evidence_state_injection=True,
        anchor_multihead_prop=True,
        prediction_head="irt2pl",
        graph_prior_mode="evidence",
        min_stu_interactions=0,
        min_exer_interactions=0,
        gec_mode=RESIDUAL_MODE,
        warm_start_checkpoint=checkpoint_path,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer="adam",
    )


def _adapter_parameter_names(model: CognitiveDiagnosisModel):
    return {
        f"{RESIDUAL_PREFIX}{name}"
        for name, _ in model.evidence_residual.named_parameters()
    }


def _assert_zero_initialized_exact_fallback(inputs, student_ids, exercise_ids):
    torch.manual_seed(123)
    parent = _model(inputs, gec_mode="v1").eval()
    torch.manual_seed(123)
    residual = _model(inputs, gec_mode=RESIDUAL_MODE).eval()

    parent_keys = set(parent.state_dict())
    residual_keys = set(residual.state_dict())
    expected_missing = _adapter_parameter_names(residual)
    assert residual_keys - parent_keys == expected_missing
    assert not parent_keys - residual_keys
    assert len(expected_missing) == 5
    assert sum(
        parameter.numel()
        for parameter in residual.evidence_residual.parameters()
    ) == 21

    incompatible = residual.load_state_dict(parent.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == expected_missing
    assert not incompatible.unexpected_keys

    parent_logits = parent(
        student_ids,
        exercise_ids,
        return_logits=True,
    )
    residual_logits, details = residual(
        student_ids,
        exercise_ids,
        return_details=True,
        return_logits=True,
    )
    assert torch.equal(parent_logits, residual_logits)
    assert torch.equal(
        details["theta_adjustment"],
        torch.zeros_like(details["theta_adjustment"]),
    )
    assert residual.evidence_residual.parameter_summary()["num_parameters"] == 21
    return parent


def _assert_warm_start_freezes_parent(
    inputs,
    parent,
    student_ids,
    exercise_ids,
) -> None:
    parent_args = vars(_paired_args("unused")).copy()
    parent_args.update({"model_variant": "full", "gec_mode": "v1"})
    data_identity = {
        "schema": "graph_irt_data_v1",
        "dataset_name": "demo",
        "train_sha256": "train",
        "valid_sha256": "valid",
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "best_model.pth")
        torch.save(
            {
                "architecture": GRAPH_IRT_ARCHITECTURE,
                "model_state_dict": parent.state_dict(),
                "args": parent_args,
                "info_dict": {"data_identity": data_identity},
                "val_auc": 0.75,
            },
            checkpoint_path,
        )
        args = _paired_args(checkpoint_path)
        residual = _model(inputs, gec_mode=RESIDUAL_MODE)
        _load_residual_warm_start(
            residual,
            args,
            {"data_identity": data_identity},
        )

        expected_trainable = _adapter_parameter_names(residual)
        trainable = {
            name
            for name, parameter in residual.named_parameters()
            if parameter.requires_grad
        }
        assert trainable == expected_trainable
        assert len(trainable) == 5
        assert sum(
            parameter.numel()
            for parameter in residual.parameters()
            if parameter.requires_grad
        ) == 21

        frozen_hash = _parameter_sha256(
            residual,
            excluded_prefix=RESIDUAL_PREFIX,
        )
        assert frozen_hash == args.residual_frozen_parameter_sha256
        optimizer = _build_optimizer(residual, args)
        optimized = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        adapter_parameters = list(residual.evidence_residual.parameters())
        assert len(optimized) == len(adapter_parameters)
        assert {id(parameter) for parameter in optimized} == {
            id(parameter) for parameter in adapter_parameters
        }
        assert sum(parameter.numel() for parameter in optimized) == 21

        residual.eval()
        labels = torch.tensor([0.0, 1.0])
        logits = residual(
            student_ids,
            exercise_ids,
            outcome_to_exclude=labels,
            return_logits=True,
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert residual.evidence_residual.rho.grad is not None
        assert torch.isfinite(residual.evidence_residual.rho.grad)
        assert residual.evidence_residual.rho.grad.abs().item() > 0.0
        for name, parameter in residual.named_parameters():
            if not name.startswith(RESIDUAL_PREFIX):
                assert not parameter.requires_grad
                assert parameter.grad is None
        optimizer.step()

        assert _parameter_sha256(
            residual,
            excluded_prefix=RESIDUAL_PREFIX,
        ) == frozen_hash
        assert all(
            torch.isfinite(parameter).all()
            for parameter in residual.evidence_residual.parameters()
        )


def _assert_leave_one_out_flip_invariance(frame: pd.DataFrame) -> None:
    flipped = frame.copy()
    flipped.loc[0, "label"] = 1.0
    inputs_a = _inputs(frame)
    inputs_b = _inputs(flipped)
    students_a, exercises_a = inputs_a[:2]
    students_b, exercises_b = inputs_b[:2]

    torch.manual_seed(456)
    model_a = _model(inputs_a, gec_mode=RESIDUAL_MODE).eval()
    torch.manual_seed(456)
    model_b = _model(inputs_b, gec_mode=RESIDUAL_MODE).eval()
    with torch.no_grad():
        for model in (model_a, model_b):
            model.evidence_residual.rho.fill_(0.8)
            model.evidence_residual.quality_output_weight.copy_(
                torch.tensor(
                    [[0.20, -0.10, 0.15], [-0.10, 0.25, -0.20]],
                    dtype=torch.float32,
                )
            )
            model.evidence_residual.quality_output_bias.copy_(
                torch.tensor([0.05, -0.05], dtype=torch.float32)
            )

    query_student_a = torch.tensor([students_a[10]], dtype=torch.long)
    query_exercise_a = torch.tensor([exercises_a[100]], dtype=torch.long)
    query_student_b = torch.tensor([students_b[10]], dtype=torch.long)
    query_exercise_b = torch.tensor([exercises_b[100]], dtype=torch.long)
    q_a = model_a.q_matrix[query_exercise_a]
    q_b = model_b.q_matrix[query_exercise_b]
    components_a = model_a._build_response_components(
        query_student_a,
        query_exercise_a,
        q_a,
        outcome_to_exclude=torch.tensor([0.0]),
    )
    components_b = model_b._build_response_components(
        query_student_b,
        query_exercise_b,
        q_b,
        outcome_to_exclude=torch.tensor([1.0]),
    )
    for name in ("evidence", "count", "correct", "concept_rate"):
        value_a = getattr(components_a, name)
        value_b = getattr(components_b, name)
        assert value_a is not None and value_b is not None
        assert torch.allclose(value_a, value_b, atol=1e-7, rtol=0.0), name

    relations_a = model_a.relation_learning()
    relations_b = model_b.relation_learning()
    assert torch.equal(relations_a, relations_b)
    adjustment_a = model_a._compute_evidence_residual(
        relations_a,
        components_a,
    )
    adjustment_b = model_b._compute_evidence_residual(
        relations_b,
        components_b,
    )
    assert torch.allclose(adjustment_a, adjustment_b, atol=1e-7, rtol=0.0)

    logits_a = model_a(
        query_student_a,
        query_exercise_a,
        outcome_to_exclude=torch.tensor([0.0]),
        return_logits=True,
    )
    logits_b = model_b(
        query_student_b,
        query_exercise_b,
        outcome_to_exclude=torch.tensor([1.0]),
        return_logits=True,
    )
    assert torch.allclose(logits_a, logits_b, atol=1e-7, rtol=0.0)


def main() -> None:
    frame = pd.DataFrame(
        {
            "stu_id": [10, 10, 10, 20, 20, 20],
            "exer_id": [100, 100, 102, 100, 102, 101],
            "cpt_seq": ["1", "1", "2,3", "1", "2,3", "1,2"],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "label": [0, 1, 0, 1, 1, 0],
        }
    )
    inputs = _inputs(frame)
    students, exercises = inputs[:2]
    student_ids = torch.tensor(
        [students[10], students[20]],
        dtype=torch.long,
    )
    exercise_ids = torch.tensor(
        [exercises[100], exercises[102]],
        dtype=torch.long,
    )

    parent = _assert_zero_initialized_exact_fallback(
        inputs,
        student_ids,
        exercise_ids,
    )
    _assert_warm_start_freezes_parent(
        inputs,
        parent,
        student_ids,
        exercise_ids,
    )
    _assert_leave_one_out_flip_invariance(frame)
    print(
        "OK: relation residual exactly preserves v1 at initialization, "
        "warm-starts 21 adapter parameters, freezes the parent, and remains "
        "leave-one-out flip invariant."
    )


if __name__ == "__main__":
    main()
