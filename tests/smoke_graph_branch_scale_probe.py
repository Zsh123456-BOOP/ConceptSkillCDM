#!/usr/bin/env python
"""Smoke checks for the frozen graph-branch scale algebra."""

import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.probe_graph_branch_scales import compose_scaled_logits


def main() -> None:
    base = torch.tensor([0.3, -0.2, 1.1])
    state = torch.tensor([0.1, 0.4, -0.3])
    propagation = torch.tensor([-0.2, 0.5, 0.2])

    full = compose_scaled_logits(
        base,
        state,
        propagation,
        state_scale=1.0,
        propagation_scale=1.0,
    )
    assert torch.equal(full, base)

    no_state = compose_scaled_logits(
        base,
        state,
        propagation,
        state_scale=0.0,
        propagation_scale=1.0,
    )
    assert torch.allclose(no_state, base - state)

    no_propagation = compose_scaled_logits(
        base,
        state,
        propagation,
        state_scale=1.0,
        propagation_scale=0.0,
    )
    assert torch.allclose(no_propagation, base - propagation)

    half_both = compose_scaled_logits(
        base,
        state,
        propagation,
        state_scale=0.5,
        propagation_scale=0.5,
    )
    assert torch.allclose(
        half_both,
        base - 0.5 * state - 0.5 * propagation,
    )

    try:
        compose_scaled_logits(
            base,
            state[:2],
            propagation,
            state_scale=1.0,
            propagation_scale=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must be rejected")

    print("OK: frozen graph branch scales preserve Full and remove each branch exactly.")


if __name__ == "__main__":
    main()
