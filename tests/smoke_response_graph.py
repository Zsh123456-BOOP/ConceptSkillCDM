"""Contracts for query-safe train-only student-item collaboration."""

import os
import sys
from unittest import mock

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import build_id_mappings, build_student_item_interaction_graph
from src.model import CognitiveDiagnosisModel, GRAPH_IRT_ARCHITECTURE
from src.response_graph import ResponseGraphEncoder
from src.trainer import _require_graph_irt_checkpoint


def _support() -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.long),
        torch.ones(3),
        size=(2, 2),
    ).coalesce()


def _different_support() -> torch.Tensor:
    """Same shape/edge count as ``_support``, but one edge is different."""
    return torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long),
        torch.ones(3),
        size=(2, 2),
    ).coalesce()


def _model(
    enable_response_graph: bool,
    response_graph: torch.Tensor | None = None,
) -> CognitiveDiagnosisModel:
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prior = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    return CognitiveDiagnosisModel(
        num_students=2,
        num_exercises=2,
        num_concepts=2,
        q_matrix=q,
        response_graph_matrix=_support() if response_graph is None else response_graph,
        item_prior_matrix=prior,
        exposure_prior_matrix=torch.zeros_like(prior),
        knowledge_dim=4,
        num_relation_heads=1,
        num_gnn_layers=0,
        dropout=0.0,
        graph_propagation_alpha=0.0,
        enable_response_graph=enable_response_graph,
    )


def _naive_query_contexts(
    support: torch.Tensor,
    student_base: torch.Tensor,
    item_base: torch.Tensor,
    student_ids: torch.Tensor,
    item_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small differentiable reference implementation for exact edge exclusion."""
    dense_support = support.to_dense().bool()
    student_contexts = []
    item_contexts = []
    for student_id, item_id in zip(student_ids.tolist(), item_ids.tolist()):
        student_neighbours = dense_support[student_id].nonzero(as_tuple=False).flatten()
        item_neighbours = dense_support[:, item_id].nonzero(as_tuple=False).flatten()
        if bool(dense_support[student_id, item_id]):
            student_neighbours = student_neighbours[student_neighbours != item_id]
            item_neighbours = item_neighbours[item_neighbours != student_id]

        if student_neighbours.numel():
            student_context = 0.5 * (
                student_base[student_id] + item_base[student_neighbours].mean(dim=0)
            )
        else:
            student_context = student_base[student_id]
        if item_neighbours.numel():
            item_context = 0.5 * (
                item_base[item_id] + student_base[item_neighbours].mean(dim=0)
            )
        else:
            item_context = item_base[item_id]
        student_contexts.append(student_context)
        item_contexts.append(item_context)
    return torch.stack(student_contexts), torch.stack(item_contexts)


def main() -> None:
    frame = pd.DataFrame(
        {
            "stu_id": [20, 10, 10, 10],
            "exer_id": [100, 101, 100, 100],
            "cpt_seq": ["1", "2", "1", "1"],
            "label": [0, 1, 0, 1],
        }
    )
    flipped = frame.sample(frac=1.0, random_state=9).reset_index(drop=True)
    flipped["label"] = 1 - flipped["label"]
    maps = build_id_mappings([frame])[:2]
    graph_a, stats_a = build_student_item_interaction_graph([frame], *maps)
    graph_b, stats_b = build_student_item_interaction_graph([flipped], *maps)
    assert graph_a.layout == torch.sparse_coo
    assert torch.equal(graph_a.indices(), graph_b.indices())
    assert torch.equal(graph_a.values(), graph_b.values())
    assert stats_a == stats_b
    assert graph_a._nnz() == 3, "duplicate pairs must collapse to binary support"

    encoder = ResponseGraphEncoder(_support())
    student_base = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    item_base = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    student_context, item_context = encoder(
        student_base,
        item_base,
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
    )
    assert torch.allclose(student_context[0], torch.tensor([5.0, 2.0]))
    assert torch.equal(
        student_context[1], student_base[1]
    ), "degree-one query must fall back to base"
    assert torch.allclose(item_context[0], torch.tensor([1.0, 5.0]))
    assert torch.allclose(item_context[1], torch.tensor([6.0, 0.0]))

    single_student, single_item = encoder(
        student_base,
        item_base,
        torch.tensor([0]),
        torch.tensor([0]),
    )
    assert torch.equal(single_student[0], student_context[0])
    assert torch.equal(single_item[0], item_context[0])

    # The optimized path must visit only CSR rows requested by the batch. A
    # regression to full-graph sparse.mm should fail this test immediately.
    assert all(buffer.layout == torch.strided for buffer in encoder.buffers())
    with mock.patch("torch.sparse.mm", side_effect=AssertionError("full-graph SpMM called")):
        no_spmm_student, no_spmm_item = encoder(
            student_base,
            item_base,
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 1, 1]),
        )
    assert tuple(no_spmm_student.shape) == (3, 2)
    assert tuple(no_spmm_item.shape) == (3, 2)

    # Compare both values and dense-embedding gradients against a transparent
    # query-by-query implementation, including duplicates, present/absent
    # edges, and the degree-one fallback.
    query_students = torch.tensor([0, 0, 1, 1, 0])
    query_items = torch.tensor([0, 1, 0, 1, 0])
    fast_students = torch.tensor(
        [[0.2, -0.4, 0.7], [0.1, 0.6, -0.3]],
        requires_grad=True,
    )
    fast_items = torch.tensor(
        [[0.5, -0.2, 0.9], [-0.8, 0.3, 0.4]],
        requires_grad=True,
    )
    ref_students = fast_students.detach().clone().requires_grad_(True)
    ref_items = fast_items.detach().clone().requires_grad_(True)
    fast_student_context, fast_item_context = encoder(
        fast_students,
        fast_items,
        query_students,
        query_items,
    )
    ref_student_context, ref_item_context = _naive_query_contexts(
        _support(),
        ref_students,
        ref_items,
        query_students,
        query_items,
    )
    assert torch.allclose(fast_student_context, ref_student_context, atol=1e-7, rtol=1e-7)
    assert torch.allclose(fast_item_context, ref_item_context, atol=1e-7, rtol=1e-7)
    student_weights = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10.0
    item_weights = student_weights.flip(0)
    fast_loss = (fast_student_context * student_weights).sum() + (
        fast_item_context * item_weights
    ).sum()
    ref_loss = (ref_student_context * student_weights).sum() + (
        ref_item_context * item_weights
    ).sum()
    fast_loss.backward()
    ref_loss.backward()
    assert torch.allclose(fast_students.grad, ref_students.grad, atol=1e-7, rtol=1e-7)
    assert torch.allclose(fast_items.grad, ref_items.grad, atol=1e-7, rtol=1e-7)

    torch.manual_seed(13)
    full = _model(True).eval()
    torch.manual_seed(13)
    ablated = _model(False).eval()
    assert tuple(full.state_dict()) == tuple(ablated.state_dict())
    for key, value in full.state_dict().items():
        other = ablated.state_dict()[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), key
        else:
            assert value == other, key
    graph_state_key = "response_graph_encoder._extra_state"
    assert graph_state_key in full.state_dict()
    graph_state = full.state_dict()[graph_state_key]
    assert set(graph_state) == {
        "schema",
        "num_students",
        "num_items",
        "num_edges",
        "fingerprint",
    }
    assert len(graph_state["fingerprint"]) == 64
    assert not any(
        token in key
        for key in full.state_dict()
        for token in ("crow_indices", "col_indices", "edge_keys", "support")
    ), "the checkpoint must not duplicate the complete response graph"

    students = torch.tensor([0, 1])
    items = torch.tensor([0, 0])
    full_logits, full_details = full(students, items, return_details=True, return_logits=True)
    ablated_logits, ablated_details = ablated(
        students, items, return_details=True, return_logits=True
    )
    assert full_details["response_student_delta"] > 0
    assert full_details["response_item_delta"] > 0
    assert ablated_details["response_student_delta"] == 0
    assert ablated_details["response_item_delta"] == 0
    assert not torch.equal(full_logits, ablated_logits)

    train_model = _model(True)
    logits = train_model(students, items, return_logits=True)
    F.binary_cross_entropy_with_logits(logits, torch.tensor([1.0, 0.0])).backward()
    assert train_model.response_item_embedding.weight.grad is not None
    assert train_model.response_item_embedding.weight.grad.abs().sum() > 0
    assert train_model.knowledge_encoder.student_global.weight.grad is not None

    clone = _model(True).eval()
    clone.load_state_dict(full.state_dict(), strict=True)
    assert torch.equal(
        clone(students, items, return_logits=True),
        full(students, items, return_logits=True),
    )
    mismatched_graph_model = _model(True, response_graph=_different_support()).eval()
    try:
        mismatched_graph_model.load_state_dict(full.state_dict(), strict=True)
    except RuntimeError as exc:
        assert "graph schema/fingerprint mismatch" in str(exc).lower()
    else:
        raise AssertionError("strict load must reject a same-shaped graph with different edges")
    assert GRAPH_IRT_ARCHITECTURE == "graph_irt_v6"
    _require_graph_irt_checkpoint({"architecture": "graph_irt_v6"}, "v6.pth")
    try:
        _require_graph_irt_checkpoint({"architecture": "graph_irt_v5"}, "v5.pth")
    except RuntimeError:
        pass
    else:
        raise AssertionError("v6 code must reject v5 checkpoints")
    print("OK: response-graph contracts passed.")


if __name__ == "__main__":
    main()
