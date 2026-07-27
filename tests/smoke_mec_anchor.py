"""Smoke checks for target-conditioned rate correction."""

from __future__ import annotations

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

from src.dataset import build_student_concept_response_stats
from src.evidence_completion import MaskedEvidenceCompletion
from src.model import CognitiveDiagnosisModel
from src.trainer import _load_and_freeze_mec_warm_start


def _relation() -> torch.Tensor:
    # Rows are targets and columns are sources; deliberately asymmetric.
    return torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )


def _response_fixture(first_label: float = 1.0):
    frame = pd.DataFrame(
        {
            "stu_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "exer_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "cpt_seq": ["0,1", "2", "3", "1", "0,1", "2", "3", "1"],
            "label": [
                first_label,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ],
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
    stats = build_student_concept_response_stats(
        frame,
        {0: 0, 1: 1},
        {value: value for value in range(4)},
        q_matrix,
    )
    return q_matrix, stats


def _module_inputs():
    evidence = torch.tensor(
        [
            [[3.0, 0.8], [-2.0, -0.5], [2.0, 0.4], [-1.0, -0.3]],
            [[1.0, 0.1], [2.0, 0.4], [-3.0, -0.8], [0.5, 0.2]],
        ]
    )
    count = torch.tensor(
        [[4.0, 2.0, 3.0, 1.0], [1.0, 2.0, 4.0, 3.0]]
    )
    prior = torch.tensor(
        [[0.55, 0.45, 0.60, 0.40], [0.55, 0.45, 0.60, 0.40]]
    )
    global_rate = torch.tensor([[0.50], [0.50]])
    q_mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )
    return evidence, count, prior, global_rate, q_mask


def _activate(module: MaskedEvidenceCompletion) -> None:
    with torch.no_grad():
        for layer in module.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(0.1)
                layer.bias.zero_()


def main() -> None:
    completion = MaskedEvidenceCompletion(relation_matrix=_relation())
    assert sum(parameter.numel() for parameter in completion.parameters()) == 89
    assert completion.FEATURE_NAMES == (
        "related_rate",
        "related_residual",
        "rate_gap",
        "residual_gap",
        "related_rate_std",
        "related_support",
        "related_agreement",
        "target_log_count",
        "target_prior_gap",
    )
    assert "relation_matrix" not in dict(completion.named_parameters())

    evidence, count, prior, global_rate, q_mask = _module_inputs()
    completed, initial = completion(
        evidence,
        count,
        prior,
        global_rate,
        q_mask,
    )
    assert torch.equal(completed, evidence[..., 0])
    assert torch.equal(
        initial["mec_rate_delta"],
        torch.zeros_like(initial["mec_rate_delta"]),
    )
    assert bool(
        (
            (initial["mec_completion_weight"] >= 0.0)
            & (initial["mec_completion_weight"] <= 0.5)
        ).all()
    )

    # Every current-Q concept is excluded from the related-evidence pool.
    changed_q_evidence = evidence.clone()
    changed_q_count = count.clone()
    changed_q_evidence[q_mask.bool()] *= -1.0
    changed_q_count[q_mask.bool()] += 10.0
    features, support, agreement = completion.build_features(
        evidence,
        count,
        prior,
        global_rate,
        q_mask,
    )
    changed_features, changed_support, changed_agreement = (
        completion.build_features(
            changed_q_evidence,
            changed_q_count,
            prior,
            global_rate,
            q_mask,
        )
    )
    pooled_feature_indices = (0, 1, 4, 5, 6, 8)
    for index in pooled_feature_indices:
        assert torch.equal(
            features[..., index][q_mask.bool()],
            changed_features[..., index][q_mask.bool()],
        )
    assert not torch.equal(
        features[..., 2][q_mask.bool()],
        changed_features[..., 2][q_mask.bool()],
    )
    assert not torch.equal(
        features[..., 3][q_mask.bool()],
        changed_features[..., 3][q_mask.bool()],
    )
    assert not torch.equal(
        features[..., 7][q_mask.bool()],
        changed_features[..., 7][q_mask.bool()],
    )
    assert torch.equal(support[q_mask.bool()], changed_support[q_mask.bool()])
    assert torch.equal(
        agreement[q_mask.bool()],
        changed_agreement[q_mask.bool()],
    )

    _activate(completion)
    completed, details = completion(
        evidence,
        count,
        prior,
        global_rate,
        q_mask,
    )
    assert not torch.equal(completed[q_mask.bool()], evidence[..., 0][q_mask.bool()])
    assert torch.equal(
        completed[~q_mask.bool()],
        evidence[..., 0][~q_mask.bool()],
    )
    assert bool(
        (
            (details["mec_completion_weight"] >= 0.0)
            & (details["mec_completion_weight"] <= 0.5)
        ).all()
    )
    assert torch.equal(
        details["mec_applicable"],
        (details["mec_completion_weight"] > 0.0).to(evidence.dtype),
    )
    assert torch.equal(
        details["mec_abstain"],
        q_mask * (1.0 - details["mec_applicable"]),
    )
    assert bool(
        (
            details["mec_rate_delta"].abs()
            <= details["mec_completion_weight"] + 1e-7
        ).all()
    )
    assert torch.allclose(
        details["mec_rate_delta"],
        completed - evidence[..., 0],
        atol=1e-7,
        rtol=0.0,
    )

    # Diagnostics expose the effective post-clamp change, not a raw proposal.
    boundary_evidence = evidence.clone()
    boundary_evidence[0, 0, 0] = 4.0
    boundary_completed, boundary_details = completion(
        boundary_evidence,
        count,
        prior,
        global_rate,
        q_mask,
    )
    assert boundary_completed[0, 0].item() == 4.0
    assert boundary_details["mec_rate_delta"][0, 0].item() == 0.0

    # R[0,2] is active while R[0,3] is zero: direction and locality matter.
    source_two_changed = evidence.clone()
    source_two_changed[0, 2, 0] += 1.0
    _, source_two_details = completion(
        source_two_changed,
        count,
        prior,
        global_rate,
        q_mask,
    )
    source_three_changed = evidence.clone()
    source_three_changed[0, 3, 0] += 1.0
    _, source_three_details = completion(
        source_three_changed,
        count,
        prior,
        global_rate,
        q_mask,
    )
    assert not torch.equal(
        details["mec_rate_delta"][0, 0],
        source_two_details["mec_rate_delta"][0, 0],
    )
    assert torch.equal(
        details["mec_rate_delta"][0, 0],
        source_three_details["mec_rate_delta"][0, 0],
    )
    assert not torch.equal(
        details["mec_rate_delta"][0, 0],
        details["mec_rate_delta"][0, 1],
    )

    no_source_completed, no_source_details = completion(
        evidence,
        count,
        prior,
        global_rate,
        torch.ones_like(q_mask),
    )
    assert torch.equal(
        no_source_details["mec_completion_weight"],
        torch.zeros_like(q_mask),
    )
    assert torch.equal(no_source_details["mec_abstain"], torch.ones_like(q_mask))
    assert torch.equal(no_source_completed, evidence[..., 0])
    single = MaskedEvidenceCompletion(relation_matrix=torch.ones(1, 1))
    _activate(single)
    single_completed, single_details = single(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1),
        torch.full((1, 1), 0.5),
        torch.full((1, 1), 0.5),
        torch.ones(1, 1),
    )
    assert torch.equal(single_completed, torch.zeros(1, 1))
    assert torch.equal(
        single_details["mec_completion_weight"],
        torch.zeros(1, 1),
    )
    assert torch.equal(single_details["mec_abstain"], torch.ones(1, 1))

    # Related evidence alone cannot activate MEC when every Q count is zero.
    all_q_zero_count = count[:1].clone()
    all_q_zero_count[q_mask[:1].bool()] = 0.0
    all_q_zero_completed, all_q_zero_details = completion(
        evidence[:1],
        all_q_zero_count,
        prior[:1],
        global_rate[:1],
        q_mask[:1],
    )
    assert bool(
        (
            all_q_zero_details["mec_related_support"][q_mask[:1].bool()]
            > 0
        ).all()
    )
    assert torch.equal(
        all_q_zero_details["mec_completion_weight"],
        torch.zeros_like(all_q_zero_count),
    )
    assert torch.equal(
        all_q_zero_details["mec_rate_delta"],
        torch.zeros_like(all_q_zero_count),
    )
    assert torch.equal(all_q_zero_completed, evidence[:1, :, 0])

    # The direct correction is explicitly reduced as target evidence grows.
    dominance = MaskedEvidenceCompletion(relation_matrix=_relation())
    _activate(dominance)
    completion_weights = []
    for target_count in (0.0, 10.0, 100.0):
        local_count = count[:1].clone()
        local_count[0, 0] = target_count
        local_evidence = evidence[:1].clone()
        local_evidence[0, 0, 0] = 0.0
        _, local_details = dominance(
            local_evidence,
            local_count,
            prior[:1],
            global_rate[:1],
            q_mask[:1],
        )
        completion_weights.append(
            float(local_details["mec_completion_weight"][0, 0].item())
        )
    assert completion_weights[0] > completion_weights[1] > completion_weights[2]

    q_matrix, stats = _response_fixture()
    common = dict(
        num_students=2,
        num_exercises=4,
        num_concepts=4,
        q_matrix=q_matrix,
        item_prior_matrix=_relation(),
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
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(17)
    mec = CognitiveDiagnosisModel(
        **common,
        evidence_anchor_mode="mec",
    ).eval()
    mec_rng = torch.random.get_rng_state().clone()
    assert torch.equal(baseline_rng, mec_rng)
    assert mec.diagnosis_head.evidence_anchor_channels == 2
    assert (
        sum(parameter.numel() for parameter in mec.parameters())
        - sum(parameter.numel() for parameter in baseline.parameters())
        == 89
    )
    assert set(mec.state_dict()) - set(baseline.state_dict()) == {
        "evidence_completion.relation_matrix",
        "evidence_completion.net.0.weight",
        "evidence_completion.net.0.bias",
        "evidence_completion.net.2.weight",
        "evidence_completion.net.2.bias",
    }

    student_ids = torch.tensor([0, 1], dtype=torch.long)
    exercise_ids = torch.tensor([0, 0], dtype=torch.long)
    labels = torch.tensor([1.0, 0.0])
    baseline_logits = baseline(
        student_ids,
        exercise_ids,
        outcome_to_exclude=labels,
        return_logits=True,
    )
    mec_logits, model_details = mec(
        student_ids,
        exercise_ids,
        outcome_to_exclude=labels,
        return_logits=True,
        return_details=True,
    )
    assert torch.equal(baseline_logits, mec_logits)
    assert model_details["evidence_anchor"].size(-1) == 2
    assert torch.equal(model_details["logits"], model_details["irt_logit"])

    mec.zero_grad(set_to_none=True)
    train_logits = mec(
        student_ids,
        exercise_ids,
        outcome_to_exclude=labels,
        return_logits=True,
    )
    F.binary_cross_entropy_with_logits(train_logits, labels).backward()
    assert mec.evidence_completion.net[-1].weight.grad is not None
    assert float(
        mec.evidence_completion.net[-1].weight.grad.abs().sum().item()
    ) > 0.0

    # Learned state-graph weights do not enter the fixed MEC relation.
    graph_common = dict(common)
    graph_common["disable_graph_module"] = False
    graph_common["graph_propagation_alpha"] = 0.2
    torch.manual_seed(23)
    graph_mec = CognitiveDiagnosisModel(
        **graph_common,
        evidence_anchor_mode="mec",
    ).eval()
    _activate(graph_mec.evidence_completion)
    _, before = graph_mec(
        student_ids,
        exercise_ids,
        outcome_to_exclude=labels,
        return_logits=True,
        return_details=True,
    )
    with torch.no_grad():
        for parameter in graph_mec.relation_learning.parameters():
            parameter.add_(0.25)
    _, after = graph_mec(
        student_ids,
        exercise_ids,
        outcome_to_exclude=labels,
        return_logits=True,
        return_details=True,
    )
    assert torch.equal(
        before["mec_completion_weight"],
        after["mec_completion_weight"],
    )
    assert torch.equal(
        before["mec_rate_delta"],
        after["mec_rate_delta"],
    )

    # Flipping the current label cannot enter its own correction query.
    _, flipped_stats = _response_fixture(first_label=0.0)
    flipped_common = dict(common)
    flipped_common["response_evidence_stats"] = flipped_stats
    torch.manual_seed(31)
    original_model = CognitiveDiagnosisModel(
        **common,
        evidence_anchor_mode="mec",
    ).eval()
    torch.manual_seed(31)
    flipped_model = CognitiveDiagnosisModel(
        **flipped_common,
        evidence_anchor_mode="mec",
    ).eval()
    _activate(original_model.evidence_completion)
    _activate(flipped_model.evidence_completion)
    original_logit, original_details = original_model(
        torch.tensor([0]),
        torch.tensor([0]),
        outcome_to_exclude=torch.tensor([1.0]),
        return_logits=True,
        return_details=True,
    )
    flipped_logit, flipped_details = flipped_model(
        torch.tensor([0]),
        torch.tensor([0]),
        outcome_to_exclude=torch.tensor([0.0]),
        return_logits=True,
        return_details=True,
    )
    assert torch.allclose(original_logit, flipped_logit, atol=1e-7, rtol=0.0)
    assert torch.equal(
        original_details["mec_completion_weight"],
        flipped_details["mec_completion_weight"],
    )
    assert torch.allclose(
        original_details["mec_rate_delta"],
        flipped_details["mec_rate_delta"],
        atol=1e-7,
        rtol=0.0,
    )

    # The stage-2 loader inherits every shared tensor and freezes all but MEC.
    source_args = dict(
        dataset_name="smoke",
        seed=42,
        train_evidence_mode="excluded",
        model_variant="no_graph_calibration",
        evidence_anchor_mode="direct_only",
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=2,
        dropout=0.0,
        graph_topk=None,
        disable_self_loop=False,
        allow_self_loop=True,
        graph_identity_residual=0.0,
        graph_dropout=-1.0,
        graph_tau_init=1.0,
        graph_propagation_alpha=0.0,
        graph_prior_strength_init=1.0,
        gnn_residual_weight=0.5,
        prediction_head="irt2pl",
        evidence_state_injection=False,
        anchor_multihead_prop=True,
        disable_graph_module=True,
        graph_prior_mode="evidence",
        min_stu_interactions=0,
        min_exer_interactions=0,
    )
    warm_info = {
        "num_students": 2,
        "num_exercises": 4,
        "num_concepts": 4,
        "q_matrix": q_matrix,
        "item_prior_matrix": _relation(),
        "exposure_prior_matrix": torch.zeros(4, 4),
        "data_identity": {
            "schema": "graph_irt_data_v1",
            "dataset_name": "smoke",
            "data_dir": "/tmp/smoke",
            "train_sha256": "train",
            "valid_sha256": "valid",
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        source_path = os.path.join(directory, "best_model.pth")
        torch.save(
            {
                "architecture": baseline.architecture,
                "epoch": 3,
                "model_state_dict": baseline.state_dict(),
                "args": source_args,
                "info_dict": warm_info,
                "val_metrics": {
                    "auc": 0.75,
                    "bce_loss": 0.50,
                    "rmse": 0.40,
                },
                "val_auc": 0.75,
            },
            source_path,
        )
        warm_candidate = CognitiveDiagnosisModel(
            **common,
            evidence_anchor_mode="mec",
        )
        candidate_args = SimpleNamespace(
            **{
                **source_args,
                "model_variant": "mec",
                "evidence_anchor_mode": "mec",
                "warm_start_checkpoint": source_path,
            }
        )
        provenance = _load_and_freeze_mec_warm_start(
            warm_candidate,
            candidate_args,
            warm_info,
        )
        assert provenance["source_epoch"] == 3
        trainable = {
            name
            for name, parameter in warm_candidate.named_parameters()
            if parameter.requires_grad
        }
        assert trainable
        assert all(name.startswith("evidence_completion.") for name in trainable)
        before_state = {
            key: value.detach().clone()
            for key, value in warm_candidate.state_dict().items()
        }
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in warm_candidate.parameters()
                if parameter.requires_grad
            ],
            lr=0.003,
        )
        optimizer.zero_grad(set_to_none=True)
        warm_candidate.eval()
        warm_logits = warm_candidate(
            student_ids,
            exercise_ids,
            outcome_to_exclude=labels,
            return_logits=True,
        )
        F.binary_cross_entropy_with_logits(warm_logits, labels).backward()
        optimizer.step()
        after_state = warm_candidate.state_dict()
        for key, before_value in before_state.items():
            if not key.startswith("evidence_completion."):
                assert torch.equal(before_value, after_state[key]), key
    print(
        "OK: MEC-v3 is target-conditioned, Q-isolated, bounded, "
        "baseline-preserving, and remains a single two-channel 2PL path."
    )


if __name__ == "__main__":
    main()
