"""Smoke checks for permutation-invariant, label-free graph priors."""

import os
import sys
import tempfile

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import (
    build_student_coexposure_prior,
    create_dataloaders,
    make_degree_random_prior,
)
from src.model import CognitiveDiagnosisModel


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_coexposure_is_order_and_label_invariant() -> None:
    frame = pd.DataFrame(
        [
            {"stu_id": 1, "cpt_seq": "100", "label": 1},
            {"stu_id": 1, "cpt_seq": "101,102", "label": 0},
            {"stu_id": 2, "cpt_seq": "100,101", "label": 1},
            {"stu_id": 3, "cpt_seq": "102", "label": 0},
            {"stu_id": 3, "cpt_seq": "103", "label": 1},
        ]
    )
    concept_map = {100: 0, 101: 1, 102: 2, 103: 3}
    prior, stats = build_student_coexposure_prior([frame], concept_map)
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled["label"] = 1 - shuffled["label"]
    contrast, contrast_stats = build_student_coexposure_prior([shuffled], concept_map)

    _assert(torch.allclose(prior, contrast, atol=1e-7), "co-exposure prior must ignore row order and labels")
    for key in stats:
        _assert(abs(stats[key] - contrast_stats[key]) < 1e-6, f"co-exposure diagnostic changed: {key}")
    _assert(torch.all(torch.diag(prior) == 0.0).item(), "co-exposure must not create self evidence")
    nonempty = prior.sum(dim=-1) > 0
    _assert(torch.allclose(prior[nonempty].sum(dim=-1), torch.ones(int(nonempty.sum()))), "nonempty rows must sum to one")
    _assert(stats["exposure_student_count"] == 3.0, "all train students should be counted")


def _check_loader_builds_label_free_exposure_graph() -> None:
    train = pd.DataFrame(
        [
            {"stu_id": 1, "exer_id": 10, "cpt_seq": "100", "label": 1},
            {"stu_id": 1, "exer_id": 11, "cpt_seq": "101", "label": 0},
            {"stu_id": 1, "exer_id": 12, "cpt_seq": "102", "label": 1},
            {"stu_id": 2, "exer_id": 10, "cpt_seq": "100", "label": 0},
            {"stu_id": 2, "exer_id": 11, "cpt_seq": "101", "label": 1},
            {"stu_id": 2, "exer_id": 12, "cpt_seq": "102", "label": 0},
        ]
    )
    valid = pd.DataFrame(
        [
            {"stu_id": 1, "exer_id": 10, "cpt_seq": "100", "label": 1},
            {"stu_id": 1, "exer_id": 11, "cpt_seq": "101", "label": 0},
            {"stu_id": 1, "exer_id": 99, "cpt_seq": "999", "label": 0},
        ]
    )
    test = pd.DataFrame(
        [
            {"stu_id": 2, "exer_id": 11, "cpt_seq": "101", "label": 1},
            {"stu_id": 2, "exer_id": 10, "cpt_seq": "100", "label": 0},
            {"stu_id": 999, "exer_id": 11, "cpt_seq": "101", "label": 0},
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        paths = {}
        for name, frame in (("train", train), ("valid", valid), ("test", test)):
            path = os.path.join(directory, f"{name}.csv")
            frame.to_csv(path, index=False)
            paths[name] = path
        _, _, _, info = create_dataloaders(
            train_file=paths["train"],
            val_file=paths["valid"],
            test_file=paths["test"],
            batch_size=2,
            num_workers=0,
            min_stu_interactions=0,
            min_exer_interactions=0,
        )

    exposure = info["exposure_prior_matrix"]
    _assert(tuple(exposure.shape) == (3, 3), "exposure prior should cover train concepts")
    _assert(torch.allclose(exposure.sum(dim=-1), torch.ones(3)), "exposure rows should sum to one")
    _assert(info["item_prior_matrix"].sum().item() == 0.0, "single-concept items must not fake item edges")
    _assert(info["val_seen_rows"] == 2 and info["test_seen_rows"] == 2, "seen filtering should be explicit")


def _check_relation_sources_are_active_and_ablatable() -> None:
    q_matrix = torch.eye(4, dtype=torch.float32)
    item_prior = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    )
    exposure_prior = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    model = CognitiveDiagnosisModel(
        num_students=3,
        num_exercises=4,
        num_concepts=4,
        q_matrix=q_matrix,
        item_prior_matrix=item_prior,
        exposure_prior_matrix=exposure_prior,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        graph_dropout=0.0,
        graph_prior_strength_init=1.0,
        graph_topk=3,
    ).eval()
    relation = model.relation_learning()
    diagnostics = model.relation_learning._last_support_diagnostics
    _assert(relation[0, 0, 1].item() > 0.0, "item support should survive source-balanced top-k")
    _assert(relation[0, 0, 2].item() > 0.0, "co-exposure support should survive source-balanced top-k")
    _assert(diagnostics["support_item_survival_rate"].item() > 0.0, "item survival should be tracked")
    _assert(diagnostics["support_exposure_survival_rate"].item() > 0.0, "exposure survival should be tracked")

    item_beta = F.softplus(model.relation_learning.prior_strength_raw.detach())
    exposure_beta = F.softplus(model.relation_learning.exposure_prior_strength_raw.detach())
    _assert(item_beta[0].item() > exposure_beta[0].item(), "head 0 should start item-heavy")
    _assert(exposure_beta[1].item() > item_beta[1].item(), "head 1 should start exposure-heavy")


def _check_degree_random_preserves_row_degree() -> None:
    item = torch.tensor(
        [[0.0, 0.8, 0.2, 0.0], [0.3, 0.0, 0.0, 0.7], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    exposure = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.4, 0.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    evidence_support = ((item + exposure) > 0.0) & (~torch.eye(4, dtype=torch.bool))
    randomized, stats = make_degree_random_prior(item, exposure, seed=7)
    _assert(
        torch.equal((randomized > 0.0).sum(dim=-1), evidence_support.sum(dim=-1)),
        "degree_random must preserve row-wise evidence degree",
    )
    _assert(torch.all(torch.diag(randomized) == 0.0).item(), "degree_random must not add self evidence")
    _assert(stats["degree_random_edge_count"] == float(evidence_support.sum().item()), "edge count mismatch")


def main() -> None:
    _check_coexposure_is_order_and_label_invariant()
    _check_loader_builds_label_free_exposure_graph()
    _check_relation_sources_are_active_and_ablatable()
    _check_degree_random_preserves_row_degree()
    print("OK: graph-prior semantics passed.")


if __name__ == "__main__":
    main()
