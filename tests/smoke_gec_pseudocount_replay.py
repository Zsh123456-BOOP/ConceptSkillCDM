"""Smoke contracts for validation-only GEC pseudo-count replay."""

import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.gec_pseudocount_replay import (
    _degree_matched_random_graph,
    build_fixed_graphs,
    fuse_pseudocount,
)


def main() -> None:
    prior = torch.tensor(
        [
            [0.0, 0.7, 0.2, 0.1],
            [0.4, 0.0, 0.4, 0.2],
            [0.5, 0.3, 0.0, 0.2],
            [0.2, 0.3, 0.5, 0.0],
        ]
    )
    graphs, stats = build_fixed_graphs(prior, topk=2, random_seed=42)
    real = graphs["exposure"]
    random_graph = graphs["degree_random"]
    assert torch.equal((real > 0).sum(1), (random_graph > 0).sum(1))
    assert torch.equal((real > 0).sum(0), (random_graph > 0).sum(0))
    assert torch.equal(torch.diagonal(real), torch.zeros(4))
    assert torch.equal(torch.diagonal(random_graph), torch.zeros(4))
    assert torch.allclose(real.sum(1), torch.ones(4))
    assert torch.allclose(random_graph.sum(1), torch.ones(4))
    assert stats["graph_edges"] == 8.0

    repeated, repeated_swaps = _degree_matched_random_graph(real, 42)
    assert torch.equal(random_graph, repeated)
    assert repeated_swaps == stats["random_successful_swaps"]
    for row in range(4):
        expected = torch.sort(real[row, real[row] > 0]).values
        actual = torch.sort(random_graph[row, random_graph[row] > 0]).values
        assert torch.equal(expected, actual)

    count = torch.tensor([[0.0, 2.0, 4.0, 1.0]])
    correct = torch.tensor([[0.0, 2.0, 1.0, 0.0]])
    concept_rate = torch.tensor([0.55, 0.60, 0.45, 0.50])
    posterior = (correct + concept_rate) / (count + 1.0)
    rate = (
        (torch.logit(posterior) - torch.logit(concept_rate))
        * count
        / (count + 1.0)
    )
    query_mask = torch.tensor([[True, False, False, False]])

    unchanged, zero_support = fuse_pseudocount(
        rate_evidence=rate,
        correct=correct,
        count=count,
        concept_rate=concept_rate,
        query_mask=query_mask,
        graph=torch.zeros_like(real),
    )
    assert torch.equal(unchanged, rate)
    assert torch.equal(zero_support, torch.zeros_like(count))

    fused, support = fuse_pseudocount(
        rate_evidence=rate,
        correct=correct,
        count=count,
        concept_rate=concept_rate,
        query_mask=query_mask,
        graph=real,
    )
    assert torch.isfinite(fused).all()
    assert (support >= 0.0).all() and (support <= 1.0 + 1e-6).all()
    assert support[0, 0] > 0.0
    assert fused[0, 0] != rate[0, 0]

    # Every concept attached to the current item is excluded as a graph source.
    only_query_source = torch.zeros_like(real)
    only_query_source[1, 0] = 1.0
    excluded, excluded_support = fuse_pseudocount(
        rate_evidence=rate,
        correct=correct,
        count=count,
        concept_rate=concept_rate,
        query_mask=query_mask,
        graph=only_query_source,
    )
    assert excluded_support[0, 1] == 0.0
    assert excluded[0, 1] == rate[0, 1]

    print("OK: GEC pseudo-count replay contracts passed.")


if __name__ == "__main__":
    main()
