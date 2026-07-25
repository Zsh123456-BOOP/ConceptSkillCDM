"""Smoke checks for the raw-support relation-quality residual."""

import math
import os
import sys
from unittest import mock

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evidence_residual import RelationQualitySignedResidual


def _raw_supports() -> tuple[torch.Tensor, torch.Tensor]:
    item_support = torch.tensor(
        [
            [50.0, 1.0, 9.0],
            [4.0, 40.0, 2.0],
            [7.0, 3.0, 30.0],
        ],
        dtype=torch.float32,
    )
    exposure_support = torch.tensor(
        [
            [80.0, 8.0, 2.0],
            [3.0, 70.0, 6.0],
            [1.0, 5.0, 60.0],
        ],
        dtype=torch.float32,
    )
    return item_support, exposure_support


def _module(
    *,
    heads: int = 2,
    enabled: bool = True,
    max_abs_adjustment: float = 0.20,
) -> RelationQualitySignedResidual:
    item_support, exposure_support = _raw_supports()
    return RelationQualitySignedResidual(
        num_relation_heads=heads,
        item_support_matrix=item_support,
        exposure_support_matrix=exposure_support,
        max_abs_adjustment=max_abs_adjustment,
        enabled=enabled,
    )


def _small_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    relations = torch.tensor(
        [
            [
                [0.75, 0.20, 0.05],
                [0.10, 0.80, 0.10],
                [0.15, 0.25, 0.60],
            ],
            [
                [0.70, 0.05, 0.25],
                [0.20, 0.70, 0.10],
                [0.10, 0.30, 0.60],
            ],
        ],
        dtype=torch.float32,
    )
    evidence = torch.tensor(
        [
            [[200.0, 0.0], [20.0, 0.5], [-10.0, -0.25]],
            [[50.0, -0.5], [-20.0, -0.2], [15.0, 0.4]],
        ],
        dtype=torch.float32,
    )
    count = torch.tensor(
        [[0.0, 4.0, 2.0], [1.0, 5.0, 3.0]],
        dtype=torch.float32,
    )
    return relations, evidence, count


def _assert_zero_initialization_and_details() -> None:
    module = _module()
    relations, evidence, count = _small_inputs()
    adjustment, details = module(
        relations,
        response_evidence=evidence,
        count=count,
        return_details=True,
    )

    assert sum(parameter.numel() for parameter in module.parameters()) == 21
    assert torch.equal(
        module.quality_hidden_weight,
        torch.eye(3, dtype=module.quality_hidden_weight.dtype),
    )
    assert torch.equal(
        module.quality_hidden_bias,
        torch.zeros_like(module.quality_hidden_bias),
    )
    assert torch.equal(
        module.quality_output_weight,
        torch.zeros_like(module.quality_output_weight),
    )
    assert torch.equal(
        module.quality_output_bias,
        torch.zeros_like(module.quality_output_bias),
    )
    assert module.rho.item() == 0.0
    assert torch.equal(adjustment, torch.zeros_like(adjustment))
    assert torch.equal(details["adjustment"], adjustment)
    expected_quality = (
        0.5 * (1.0 - torch.eye(3))
    ).unsqueeze(0).repeat(2, 1, 1)
    assert torch.equal(details["quality_positive"], expected_quality)
    assert torch.equal(details["quality_negative"], expected_quality)
    assert details["support"].shape == (2, 3)
    assert details["gate"].shape == (2, 3)
    assert details["positive_message"].shape == (2, 3)
    assert details["negative_message"].shape == (2, 3)
    assert details["alpha"].item() == 0.0
    assert details["enabled"].item() == 1.0
    assert {
        "adjustment",
        "quality_positive",
        "quality_negative",
        "support",
        "gate",
        "positive_message",
        "negative_message",
        "alpha",
        "enabled",
    }.issubset(details)

    state = module.state_dict()
    assert "item_support_matrix" not in state
    assert "exposure_support_matrix" not in state
    summary = module.parameter_summary()
    assert summary["enabled"] is True
    assert summary["num_relation_heads"] == 2
    assert summary["num_concepts"] == 3
    assert summary["num_parameters"] == 21
    assert summary["num_trainable_parameters"] == 21
    assert summary["all_parameters_finite"] is True
    assert summary["effective_alpha"] == 0.0
    assert all(
        math.isfinite(value)
        for value in summary.values()
        if isinstance(value, float)
    )


def _assert_raw_support_changes_quality() -> None:
    item_support_a, exposure_support_a = _raw_supports()
    item_support_b = item_support_a.clone()
    exposure_support_b = exposure_support_a.clone()
    item_support_b[0, 1] = 9.0
    exposure_support_b[0, 1] = 1.0
    module_a = RelationQualitySignedResidual(
        2,
        item_support_a,
        exposure_support_a,
    )
    module_b = RelationQualitySignedResidual(
        2,
        item_support_b,
        exposure_support_b,
    )
    with torch.no_grad():
        learned_output = torch.tensor(
            [[1.25, -0.50, 0.25], [-0.25, 1.00, -0.50]],
            dtype=torch.float32,
        )
        module_a.quality_output_weight.copy_(learned_output)
        module_b.quality_output_weight.copy_(learned_output)

    relations, evidence, count = _small_inputs()
    _, details_a = module_a(
        relations,
        response_evidence=evidence,
        count=count,
        return_details=True,
    )
    _, details_b = module_b(
        relations,
        response_evidence=evidence,
        count=count,
        return_details=True,
    )
    assert not torch.equal(
        details_a["quality_positive"][:, 0, 1],
        details_b["quality_positive"][:, 0, 1],
    )
    assert not torch.equal(
        details_a["quality_negative"][:, 0, 1],
        details_b["quality_negative"][:, 0, 1],
    )


def _assert_only_residual_channel_is_used() -> None:
    module = _module()
    with torch.no_grad():
        module.rho.fill_(1.0)
        module.quality_output_weight.copy_(
            torch.tensor(
                [[0.25, -0.10, 0.30], [-0.20, 0.35, -0.15]],
                dtype=torch.float32,
            )
        )
    relations, evidence, count = _small_inputs()
    changed_rate = evidence.clone()
    changed_rate[..., 0] = torch.tensor(
        [[-1.0e6, 5.0e5, 7.0e5], [8.0e5, -9.0e5, 3.0e5]]
    )
    original = module(
        relations,
        response_evidence=evidence,
        count=count,
    )
    changed = module(
        relations,
        response_evidence=changed_rate,
        count=count,
    )
    assert torch.equal(original, changed)


def _two_concept_module() -> RelationQualitySignedResidual:
    raw_item = torch.tensor([[100.0, 8.0], [4.0, 100.0]])
    raw_exposure = torch.tensor([[100.0, 12.0], [6.0, 100.0]])
    return RelationQualitySignedResidual(
        1,
        raw_item,
        raw_exposure,
    )


def _assert_positive_negative_directions() -> None:
    relations = torch.tensor(
        [[[0.8, 0.2], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    count = torch.tensor([[0.0, 5.0]], dtype=torch.float32)
    module = _two_concept_module()
    with torch.no_grad():
        module.rho.fill_(2.0)

    positive = module(
        relations,
        response_evidence=torch.tensor(
            [[[999.0, 0.0], [-999.0, 0.8]]],
            dtype=torch.float32,
        ),
        count=count,
    )
    negative = module(
        relations,
        response_evidence=torch.tensor(
            [[[-999.0, 0.0], [999.0, -0.8]]],
            dtype=torch.float32,
        ),
        count=count,
    )
    assert positive[0, 0].item() > 0.0
    assert negative[0, 0].item() < 0.0
    assert positive[0, 1].item() == 0.0
    assert negative[0, 1].item() == 0.0


def _assert_gate_formula_and_bound() -> None:
    count = torch.tensor([0.0, 1.0, 4.0, 9.0])
    support = torch.tensor([0.0, 0.25, 1.0, 4.0])
    gate = RelationQualitySignedResidual.analytic_gate(count, support)
    expected = torch.rsqrt(count + 1.0) * (
        support / (support + 0.25)
    )
    assert torch.equal(gate, expected)

    module = _module(max_abs_adjustment=0.20)
    with torch.no_grad():
        module.rho.fill_(100.0)
        module.quality_output_weight.fill_(20.0)
        module.quality_output_bias.copy_(torch.tensor([20.0, -20.0]))
    relations, evidence, count_matrix = _small_inputs()
    adjustment, details = module(
        relations,
        response_evidence=evidence * 1.0e5,
        count=count_matrix,
        return_details=True,
    )
    assert torch.isfinite(adjustment).all()
    assert float(adjustment.abs().max()) <= 0.20
    for value in details.values():
        assert torch.isfinite(value).all()


def _assert_enabled_no_edges_and_self_edges() -> None:
    disabled = _module(enabled=False)
    relations, evidence, count = _small_inputs()
    with torch.no_grad():
        disabled.rho.fill_(2.0)
    adjustment, details = disabled(
        relations,
        response_evidence=evidence,
        count=count,
        return_details=True,
    )
    assert torch.equal(adjustment, torch.zeros_like(adjustment))
    assert details["enabled"].item() == 0.0
    assert disabled.parameter_summary()["enabled"] is False

    module = _module()
    with torch.no_grad():
        module.rho.fill_(2.0)
    for no_edges in (
        torch.zeros((2, 3, 3), dtype=torch.float32),
        torch.eye(3, dtype=torch.float32).repeat(2, 1, 1),
    ):
        output, no_edge_details = module(
            no_edges,
            response_evidence=evidence,
            count=count,
            return_details=True,
        )
        assert torch.equal(output, torch.zeros_like(output))
        assert torch.equal(
            no_edge_details["support"],
            torch.zeros_like(no_edge_details["support"]),
        )
        assert torch.equal(
            no_edge_details["gate"],
            torch.zeros_like(no_edge_details["gate"]),
        )

    one = RelationQualitySignedResidual(
        1,
        torch.tensor([[100.0]]),
        torch.tensor([[100.0]]),
    )
    with torch.no_grad():
        one.rho.fill_(2.0)
    single = one(
        torch.ones((1, 1, 1), dtype=torch.float32),
        response_evidence=torch.tensor([[[100.0, -0.5]]]),
        count=torch.tensor([[4.0]]),
    )
    assert torch.equal(single, torch.zeros_like(single))


def _assert_relation_detached_and_quality_trainable() -> None:
    module = _module()
    with torch.no_grad():
        module.rho.fill_(0.8)
        module.quality_output_weight.copy_(
            torch.tensor(
                [[0.30, -0.20, 0.25], [-0.15, 0.35, -0.10]],
                dtype=torch.float32,
            )
        )
    relations, evidence, count = _small_inputs()
    relations = relations.clone().requires_grad_(True)
    adjustment = module(
        relations,
        response_evidence=evidence,
        count=count,
    )
    adjustment.square().sum().backward()

    assert relations.grad is None
    quality_parameters = (
        module.quality_hidden_weight,
        module.quality_hidden_bias,
        module.quality_output_weight,
        module.quality_output_bias,
    )
    for parameter in quality_parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert (
        sum(parameter.grad.abs().sum().item() for parameter in quality_parameters)
        > 0.0
    )
    assert module.rho.grad is not None
    assert torch.isfinite(module.rho.grad)


def _assert_large_sparse_forward_backward() -> None:
    concepts = 256
    batch_size = 4
    heads = 2
    indices = torch.arange(concepts)
    relations = torch.zeros(
        (heads, concepts, concepts),
        dtype=torch.float32,
    )
    relations[:, indices, indices] = 0.75
    relations[0, indices, (indices + 1) % concepts] = 0.25
    relations[1, indices, (indices + 3) % concepts] = 0.25
    relations.requires_grad_(True)

    raw_item = torch.zeros((concepts, concepts), dtype=torch.float32)
    raw_exposure = torch.zeros_like(raw_item)
    raw_item[indices, (indices + 1) % concepts] = 11.0
    raw_item[indices, (indices + 3) % concepts] = 5.0
    raw_exposure[indices, (indices + 1) % concepts] = 7.0
    raw_exposure[indices, (indices + 3) % concepts] = 13.0
    module = RelationQualitySignedResidual(
        heads,
        raw_item,
        raw_exposure,
    )
    with torch.no_grad():
        module.rho.fill_(0.8)
        module.quality_output_weight.copy_(
            torch.tensor(
                [[0.20, -0.10, 0.15], [-0.10, 0.25, -0.20]],
                dtype=torch.float32,
            )
        )

    grid = torch.arange(batch_size * concepts).reshape(batch_size, concepts)
    count = (grid % 7 + 1).float()
    sign = torch.where(
        grid % 2 == 0,
        torch.ones_like(grid),
        -torch.ones_like(grid),
    ).float()
    evidence = torch.stack(
        (
            1000.0 * sign,
            0.5 * torch.roll(sign, shifts=1, dims=1),
        ),
        dim=-1,
    )

    sparse_calls = 0
    original_sparse_mm = torch.sparse.mm

    def counted_sparse_mm(*args, **kwargs):
        nonlocal sparse_calls
        sparse_calls += 1
        return original_sparse_mm(*args, **kwargs)

    with mock.patch.object(torch.sparse, "mm", side_effect=counted_sparse_mm):
        output = module(
            relations,
            response_evidence=evidence,
            count=count,
        )
        output.square().mean().backward()

    assert sparse_calls == 3 * heads
    assert output.shape == (batch_size, concepts)
    assert torch.isfinite(output).all()
    assert float(output.abs().max()) <= module.max_abs_adjustment
    assert relations.grad is None
    assert module.quality_output_weight.grad is not None
    assert module.quality_output_weight.grad.abs().sum().item() > 0.0
    for parameter in module.parameters():
        assert torch.isfinite(parameter).all()
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def main() -> None:
    _assert_zero_initialization_and_details()
    _assert_raw_support_changes_quality()
    _assert_only_residual_channel_is_used()
    _assert_positive_negative_directions()
    _assert_gate_formula_and_bound()
    _assert_enabled_no_edges_and_self_edges()
    _assert_relation_detached_and_quality_trainable()
    _assert_large_sparse_forward_backward()
    print(
        "OK: 21-parameter raw-support relation-quality residual is exactly "
        "zero-initialized, signed, bounded, graph-detached, and sparse."
    )


if __name__ == "__main__":
    main()
