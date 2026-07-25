"""Smoke checks for the bounded sparse-evidence theta residual."""

import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evidence_residual import SparseEvidenceThetaResidual


def _small_inputs():
    relations = torch.tensor(
        [
            [
                [0.80, 0.15, 0.05],
                [0.20, 0.60, 0.20],
                [0.10, 0.30, 0.60],
            ],
            [
                [0.70, 0.10, 0.20],
                [0.25, 0.70, 0.05],
                [0.15, 0.25, 0.60],
            ],
        ],
        dtype=torch.float32,
    )
    count = torch.tensor(
        [[0.0, 3.0, 5.0], [1.0, 2.0, 4.0]],
        dtype=torch.float32,
    )
    correct = torch.tensor(
        [[0.0, 3.0, 1.0], [1.0, 0.0, 3.0]],
        dtype=torch.float32,
    )
    concept_rate = torch.tensor(
        [[0.45, 0.55, 0.50], [0.45, 0.55, 0.50]],
        dtype=torch.float32,
    )
    return relations, count, correct, concept_rate


def _assert_zero_initialization() -> None:
    module = SparseEvidenceThetaResidual()
    relations, count, correct, concept_rate = _small_inputs()
    adjustment, details = module(
        relations,
        count=count,
        correct=correct,
        concept_rate=concept_rate,
        return_details=True,
    )
    assert module.rho.item() == 0.0
    assert details["alpha"].item() == 0.0
    assert torch.equal(adjustment, torch.zeros_like(adjustment))


def _assert_gate_monotonicity() -> None:
    constant = torch.full((4,), 2.0)
    by_count = SparseEvidenceThetaResidual.reliability_gate(
        torch.tensor([0.0, 1.0, 3.0, 8.0]),
        constant,
        torch.zeros(4),
    )
    assert bool((by_count[:-1] >= by_count[1:]).all())

    by_support = SparseEvidenceThetaResidual.reliability_gate(
        torch.zeros(4),
        torch.tensor([0.0, 0.5, 2.0, 8.0]),
        torch.zeros(4),
    )
    assert bool((by_support[:-1] <= by_support[1:]).all())

    by_conflict = SparseEvidenceThetaResidual.reliability_gate(
        torch.zeros(4),
        constant,
        torch.tensor([0.0, 0.5, 2.0, 8.0]),
    )
    assert bool((by_conflict[:-1] >= by_conflict[1:]).all())
    assert by_support[0].item() == 0.0


def _assert_absent_relations_are_exactly_zero() -> None:
    module = SparseEvidenceThetaResidual()
    with torch.no_grad():
        module.rho.fill_(2.0)
    _, count, correct, concept_rate = _small_inputs()

    no_support = torch.zeros((2, 3, 3))
    output = module(
        no_support,
        count=count,
        correct=correct,
        concept_rate=concept_rate,
    )
    assert torch.equal(output, torch.zeros_like(output))

    pure_self = torch.eye(3).repeat(2, 1, 1)
    output = module(
        pure_self,
        count=count,
        correct=correct,
        concept_rate=concept_rate,
    )
    assert torch.equal(output, torch.zeros_like(output))

    single = module(
        torch.ones((2, 1, 1)),
        count=torch.tensor([[3.0]]),
        correct=torch.tensor([[2.0]]),
        concept_rate=torch.tensor([[0.5]]),
    )
    assert torch.equal(single, torch.zeros_like(single))


def _assert_head_permutation_and_bound() -> None:
    module = SparseEvidenceThetaResidual()
    with torch.no_grad():
        module.rho.fill_(10.0)
    relations, count, correct, concept_rate = _small_inputs()
    expected = module(
        relations,
        count=count,
        correct=correct,
        concept_rate=concept_rate,
    )
    permuted = module(
        relations.flip(0),
        count=count,
        correct=correct,
        concept_rate=concept_rate,
    )
    assert torch.allclose(expected, permuted, atol=1e-7, rtol=0.0)
    assert float(expected.abs().max()) <= module.max_abs_adjustment + 1e-7


def _assert_relation_is_gradient_isolated() -> None:
    module = SparseEvidenceThetaResidual()
    with torch.no_grad():
        module.rho.fill_(0.75)
    relations, count, correct, concept_rate = _small_inputs()
    relations = relations.clone().requires_grad_(True)
    adjustment = module(
        relations,
        count=count,
        correct=correct,
        concept_rate=concept_rate,
    )
    loss = adjustment.square().sum()
    loss.backward()
    assert relations.grad is None
    assert module.rho.grad is not None
    assert torch.isfinite(module.rho.grad)
    assert module.rho.grad.abs().item() > 0.0


def _assert_large_concept_forward_backward() -> None:
    concepts = 256
    batch_size = 4
    heads = 2
    relations = torch.zeros((heads, concepts, concepts), dtype=torch.float32)
    indices = torch.arange(concepts)
    relations[:, indices, indices] = 0.75
    relations[0, indices, (indices + 1) % concepts] = 0.25
    relations[1, indices, (indices + 3) % concepts] = 0.25
    relations.requires_grad_(True)

    count = (torch.arange(batch_size * concepts).reshape(batch_size, concepts) % 7 + 1).float()
    high = (torch.arange(concepts) % 2 == 0).float().unsqueeze(0)
    rates = 0.25 + 0.50 * high
    correct = count * rates
    concept_rate = torch.full_like(count, 0.50)

    module = SparseEvidenceThetaResidual()
    with torch.no_grad():
        module.rho.fill_(0.50)
    output = module(
        relations,
        count=count,
        correct=correct,
        concept_rate=concept_rate,
    )
    assert output.shape == (batch_size, concepts)
    assert torch.isfinite(output).all()
    assert float(output.abs().max()) <= module.max_abs_adjustment + 1e-7
    output.square().mean().backward()
    assert relations.grad is None
    assert module.rho.grad is not None and torch.isfinite(module.rho.grad)


def main() -> None:
    _assert_zero_initialization()
    _assert_gate_monotonicity()
    _assert_absent_relations_are_exactly_zero()
    _assert_head_permutation_and_bound()
    _assert_relation_is_gradient_isolated()
    _assert_large_concept_forward_backward()
    print(
        "OK: sparse evidence residual is zero-initialized, monotone, bounded, "
        "head-order invariant, and graph-gradient isolated."
    )


if __name__ == "__main__":
    main()
