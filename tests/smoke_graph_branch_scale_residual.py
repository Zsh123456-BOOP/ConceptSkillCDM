"""Smoke checks for the bounded graph-branch scale residual."""

import math
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evidence_residual import GraphBranchScaleResidual


def _inputs():
    state_delta = torch.tensor(
        [[0.20, -0.10], [0.40, 0.30]],
        dtype=torch.float32,
    )
    evidence_anchor = torch.tensor(
        [
            [
                [9.0, -9.0, 0.50, -0.25, 0.10],
                [8.0, -8.0, -0.20, 0.30, 0.40],
            ],
            [
                [7.0, -7.0, 0.10, 0.20, -0.30],
                [6.0, -6.0, -0.40, -0.10, 0.25],
            ],
        ],
        dtype=torch.float32,
    )
    anchor_weights = torch.tensor(
        [
            [2.0, -2.0, 0.40, -0.20, 0.50],
            [3.0, -3.0, -0.30, 0.60, 0.20],
        ],
        dtype=torch.float32,
    )
    return state_delta, evidence_anchor, anchor_weights


def _full_graph_contribution(
    state_delta: torch.Tensor,
    evidence_anchor: torch.Tensor,
    anchor_weights: torch.Tensor,
) -> torch.Tensor:
    return state_delta + (
        evidence_anchor[..., 2:] * anchor_weights[:, 2:].unsqueeze(0)
    ).sum(dim=-1)


def _assert_zero_is_exact_full_identity() -> None:
    module = GraphBranchScaleResidual(3)
    state_delta, evidence_anchor, anchor_weights = _inputs()
    adjustment, details = module(
        state_delta,
        evidence_anchor,
        anchor_weights,
        return_details=True,
    )

    assert torch.equal(adjustment, torch.zeros_like(state_delta))
    assert torch.equal(details["state_route"], torch.tensor(0.0))
    assert torch.equal(
        details["propagation_route"],
        torch.zeros(3),
    )
    assert torch.equal(details["adjustment"], adjustment)
    assert torch.equal(
        details["propagation_channel_contribution"],
        evidence_anchor[..., 2:] * anchor_weights[:, 2:].unsqueeze(0),
    )
    summary = module.parameter_summary()
    assert summary["enabled"] is True
    assert summary["num_parameters"] == 4
    assert summary["num_trainable_parameters"] == 4
    assert summary["all_parameters_finite"] is True
    assert summary["effective_state_route"] == 0.0
    assert summary["effective_propagation_route_min"] == 0.0
    assert summary["effective_propagation_route_max"] == 0.0
    assert summary["route_l2"] == 0.0
    assert torch.equal(details["route"], torch.zeros(4))


def _assert_negative_routes_remove_full_branches() -> None:
    module = GraphBranchScaleResidual(3)
    with torch.no_grad():
        module.state_route_raw.fill_(-100.0)
        module.propagation_route_raw.fill_(-100.0)
    state_delta, evidence_anchor, anchor_weights = _inputs()
    full_graph = _full_graph_contribution(
        state_delta,
        evidence_anchor,
        anchor_weights,
    )
    adjustment, details = module(
        state_delta,
        evidence_anchor,
        anchor_weights,
        return_details=True,
    )

    assert details["state_route"].item() == -1.0
    assert torch.equal(
        details["propagation_route"],
        -torch.ones(3),
    )
    assert torch.allclose(
        adjustment,
        -full_graph,
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        full_graph + adjustment,
        torch.zeros_like(full_graph),
        atol=1e-7,
        rtol=0.0,
    )


def _assert_positive_routes_double_full_branches() -> None:
    module = GraphBranchScaleResidual(3)
    with torch.no_grad():
        module.state_route_raw.fill_(100.0)
        module.propagation_route_raw.fill_(100.0)
    state_delta, evidence_anchor, anchor_weights = _inputs()
    full_graph = _full_graph_contribution(
        state_delta,
        evidence_anchor,
        anchor_weights,
    )
    adjustment, details = module(
        state_delta,
        evidence_anchor,
        anchor_weights,
        return_details=True,
    )

    assert details["state_route"].item() == 1.0
    assert torch.equal(details["propagation_route"], torch.ones(3))
    assert torch.allclose(
        adjustment,
        full_graph,
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        full_graph + adjustment,
        2.0 * full_graph,
        atol=1e-7,
        rtol=0.0,
    )


def _assert_independent_gradients_and_bounds() -> None:
    module = GraphBranchScaleResidual(3)
    state_delta, evidence_anchor, anchor_weights = _inputs()
    state_delta = state_delta.requires_grad_(True)
    evidence_anchor = evidence_anchor.requires_grad_(True)
    anchor_weights = anchor_weights.requires_grad_(True)
    adjustment = module(
        state_delta,
        evidence_anchor,
        anchor_weights,
    )
    adjustment.sum().backward()

    assert module.state_route_raw.grad is not None
    assert module.state_route_raw.grad.abs().item() > 0.0
    assert module.propagation_route_raw.grad is not None
    assert bool((module.propagation_route_raw.grad.abs() > 0.0).all())
    assert state_delta.grad is None
    assert evidence_anchor.grad is None
    assert anchor_weights.grad is None

    with torch.no_grad():
        module.state_route_raw.fill_(1.0e6)
        module.propagation_route_raw.copy_(
            torch.tensor([-1.0e6, 1.0e6, 0.25])
        )
    _, details = module(
        state_delta.detach(),
        evidence_anchor.detach(),
        anchor_weights.detach(),
        return_details=True,
    )
    routes = torch.cat(
        (
            details["state_route"].reshape(1),
            details["propagation_route"],
        )
    )
    assert torch.isfinite(routes).all()
    assert float(routes.abs().max()) <= 1.0
    assert routes[0].item() == 1.0
    assert routes[1].item() == -1.0
    assert routes[2].item() == 1.0
    assert math.isclose(
        routes[3].item(),
        math.tanh(0.25),
        rel_tol=0.0,
        abs_tol=1e-7,
    )


def _assert_disabled_and_invalid_inputs() -> None:
    module = GraphBranchScaleResidual(3, enabled=False)
    with torch.no_grad():
        module.state_route_raw.fill_(0.5)
        module.propagation_route_raw.fill_(-0.5)
    state_delta, evidence_anchor, anchor_weights = _inputs()
    adjustment, details = module(
        state_delta,
        evidence_anchor,
        anchor_weights,
        return_details=True,
    )
    assert torch.equal(adjustment, torch.zeros_like(adjustment))
    assert details["enabled"].item() == 0.0
    assert module.parameter_summary()["enabled"] is False

    invalid_calls = (
        lambda: GraphBranchScaleResidual(0),
        lambda: GraphBranchScaleResidual(1.5),
        lambda: GraphBranchScaleResidual(3)(
            state_delta.unsqueeze(-1),
            evidence_anchor,
            anchor_weights,
        ),
        lambda: GraphBranchScaleResidual(3)(
            state_delta,
            evidence_anchor[..., :-1],
            anchor_weights,
        ),
        lambda: GraphBranchScaleResidual(3)(
            state_delta,
            evidence_anchor,
            anchor_weights[:, :-1],
        ),
        lambda: GraphBranchScaleResidual(3)(
            state_delta,
            evidence_anchor.clone().masked_fill(
                torch.ones_like(evidence_anchor, dtype=torch.bool),
                float("nan"),
            ),
            anchor_weights,
        ),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid branch-scale input was accepted")


def main() -> None:
    _assert_zero_is_exact_full_identity()
    _assert_negative_routes_remove_full_branches()
    _assert_positive_routes_double_full_branches()
    _assert_independent_gradients_and_bounds()
    _assert_disabled_and_invalid_inputs()
    print(
        "OK: graph branch scale residual preserves Full at zero, removes or "
        "amplifies each frozen branch, isolates gradients, and bounds routes."
    )


if __name__ == "__main__":
    main()
