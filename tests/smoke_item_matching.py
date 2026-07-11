"""Contracts for the Q-aware low-rank item-matching readout."""

from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiment_utils import _config_hash
from src.config import ITEM_MATCHING_RANK
from src.model import CognitiveDiagnosisModel, GRAPH_IRT_ARCHITECTURE
from src.trainer import (
    _checkpoint_args,
    _collect_structural_switches,
    _model_kwargs,
    _require_graph_irt_checkpoint,
    _runtime_facts,
)


def _model(enable_item_matching: bool) -> CognitiveDiagnosisModel:
    q = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    prior = torch.tensor(
        [
            [0.0, 0.7, 0.3],
            [0.5, 0.0, 0.5],
            [0.6, 0.4, 0.0],
        ]
    )
    return CognitiveDiagnosisModel(
        num_students=4,
        num_exercises=3,
        num_concepts=3,
        q_matrix=q,
        item_prior_matrix=prior,
        exposure_prior_matrix=torch.zeros_like(prior),
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        graph_dropout=0.0,
        graph_propagation_alpha=0.25,
        enable_item_matching=enable_item_matching,
    )


def _assert_batch_independent(model: CognitiveDiagnosisModel) -> None:
    model.eval()
    with torch.no_grad():
        single = model(
            torch.tensor([2]),
            torch.tensor([1]),
            return_logits=True,
        )[0]
        mixed = model(
            torch.tensor([0, 2, 3]),
            torch.tensor([0, 1, 2]),
            return_logits=True,
        )[1]
        permuted = model(
            torch.tensor([3, 0, 2]),
            torch.tensor([2, 0, 1]),
            return_logits=True,
        )[2]
    assert torch.equal(single, mixed)
    assert torch.equal(single, permuted)


def main() -> None:
    torch.manual_seed(20260711)
    full = _model(True)
    torch.manual_seed(20260711)
    ablated = _model(False)

    # Both variants keep identical state keys and common initialization.  The
    # ablation freezes and bypasses only the item-conditioned direction.
    assert set(full.state_dict()) == set(ablated.state_dict())
    for key, value in full.state_dict().items():
        assert torch.equal(value, ablated.state_dict()[key]), key
    assert full.diagnosis_head.enable_item_matching
    assert not ablated.diagnosis_head.enable_item_matching
    assert all(
        not parameter.requires_grad
        for name, parameter in ablated.diagnosis_head.named_parameters()
        if name.startswith("item_matching_")
    )

    students = torch.tensor([0, 1, 2, 3])
    exercises = torch.tensor([0, 1, 2, 0])
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
    full.eval()
    ablated.eval()
    with torch.no_grad():
        initial_full = full(students, exercises, return_logits=True)
        initial_ablated = ablated(students, exercises, return_logits=True)
    assert torch.equal(initial_full, initial_ablated)

    # The zero Q-concept factor must receive a first-step gradient and become active.
    full.train()
    optimizer = torch.optim.SGD(full.parameters(), lr=0.1)
    first_logits = full(students, exercises, return_logits=True)
    loss = F.binary_cross_entropy_with_logits(first_logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    direction = full.diagnosis_head.concept_matching_direction.weight
    assert direction.grad is not None
    assert torch.isfinite(direction.grad).all()
    assert direction.grad.abs().sum().item() > 0.0
    optimizer.step()
    assert direction.detach().abs().sum().item() > 0.0

    optimizer.zero_grad(set_to_none=True)
    second_loss = F.binary_cross_entropy_with_logits(
        full(students, exercises, return_logits=True),
        labels,
    )
    second_loss.backward()
    projection = full.diagnosis_head.item_matching_projection.weight
    assert projection.grad is not None
    assert torch.isfinite(projection.grad).all()
    assert projection.grad.abs().sum().item() > 0.0
    _assert_batch_independent(full)

    info = {
        "num_students": 4,
        "num_exercises": 3,
        "num_concepts": 3,
        "q_matrix": full.q_matrix.detach().cpu(),
        "item_prior_matrix": full.item_prior_matrix.detach().cpu(),
        "exposure_prior_matrix": torch.zeros(3, 3),
    }
    args = {
        "model_variant": "full",
        "enable_item_matching": True,
        "item_matching_rank": ITEM_MATCHING_RANK,
        "knowledge_dim": 8,
        "num_relation_heads": 2,
        "num_gnn_layers": 1,
        "dropout": 0.0,
        "graph_dropout": 0.0,
        "graph_propagation_alpha": 0.25,
    }
    switches = _collect_structural_switches(args)
    assert switches["architecture"] == GRAPH_IRT_ARCHITECTURE
    assert switches["enable_item_matching"] is True
    facts = _runtime_facts(full)
    assert facts["item_matching_enabled"] is True
    assert facts["item_matching_rank"] == ITEM_MATCHING_RANK

    hash_args = type(
        "Args",
        (),
        {
            "enable_item_matching": True,
            "item_matching_rank": ITEM_MATCHING_RANK,
        },
    )()
    full_hash = _config_hash(hash_args)
    hash_args.enable_item_matching = False
    assert _config_hash(hash_args) != full_hash
    hash_args.enable_item_matching = True
    hash_args.item_matching_rank = ITEM_MATCHING_RANK + 1
    assert _config_hash(hash_args) != full_hash

    checkpoint_args = _checkpoint_args(type("Args", (), args)())
    assert checkpoint_args["enable_item_matching"] is True
    full.eval()
    expected = full(students, exercises, return_logits=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "roundtrip.pth")
        torch.save(
            {
                "architecture": GRAPH_IRT_ARCHITECTURE,
                "args": checkpoint_args,
                "model_state_dict": full.state_dict(),
            },
            path,
        )
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        rebuilt = CognitiveDiagnosisModel(**_model_kwargs(loaded["args"], info)).eval()
        rebuilt.load_state_dict(loaded["model_state_dict"], strict=True)
        actual = rebuilt(students, exercises, return_logits=True)
    assert torch.equal(expected, actual)

    no_item_args = dict(args)
    no_item_args.update(
        model_variant="no_item_matching",
        enable_item_matching=False,
    )
    no_item_checkpoint_args = _checkpoint_args(type("Args", (), no_item_args)())
    ablated.eval()
    expected_no_item = ablated(students, exercises, return_logits=True)
    rebuilt_no_item = CognitiveDiagnosisModel(
        **_model_kwargs(no_item_checkpoint_args, info)
    ).eval()
    rebuilt_no_item.load_state_dict(ablated.state_dict(), strict=True)
    actual_no_item = rebuilt_no_item(students, exercises, return_logits=True)
    assert torch.equal(expected_no_item, actual_no_item)
    assert not rebuilt_no_item.diagnosis_head.enable_item_matching

    for legacy_architecture in (
        "graph_irt_v2",
        "graph_irt_v3",
        "graph_irt_v4",
    ):
        try:
            _require_graph_irt_checkpoint(
                {"architecture": legacy_architecture},
                "legacy.pth",
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"v5 runtime must reject a {legacy_architecture} checkpoint"
            )
    print("OK: Q-aware item-matching contracts passed.")


if __name__ == "__main__":
    main()
