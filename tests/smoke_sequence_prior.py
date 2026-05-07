import os
import sys
import tempfile

import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_train_only_sequence_prior_handles_single_concept_items() -> None:
    from src.dataset import create_dataloaders

    train_df = pd.DataFrame(
        [
            {"stu_id": 1, "exer_id": 10, "cpt_seq": "100", "label": 1},
            {"stu_id": 1, "exer_id": 11, "cpt_seq": "101", "label": 0},
            {"stu_id": 1, "exer_id": 12, "cpt_seq": "102", "label": 1},
            {"stu_id": 2, "exer_id": 13, "cpt_seq": "100", "label": 0},
            {"stu_id": 2, "exer_id": 14, "cpt_seq": "101", "label": 1},
            {"stu_id": 2, "exer_id": 15, "cpt_seq": "102", "label": 0},
        ]
    )
    val_df = pd.DataFrame(
        [
            {"stu_id": 1, "exer_id": 10, "cpt_seq": "100", "label": 1},
            {"stu_id": 1, "exer_id": 99, "cpt_seq": "999", "label": 0},
        ]
    )
    test_df = pd.DataFrame(
        [
            {"stu_id": 2, "exer_id": 11, "cpt_seq": "101", "label": 1},
            {"stu_id": 999, "exer_id": 11, "cpt_seq": "101", "label": 0},
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.csv")
        val_path = os.path.join(tmpdir, "valid.csv")
        test_path = os.path.join(tmpdir, "test.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        _, _, _, info = create_dataloaders(
            train_file=train_path,
            val_file=val_path,
            test_file=test_path,
            batch_size=2,
            num_workers=0,
            min_stu_interactions=0,
            min_exer_interactions=0,
            min_poison_count=0,
        )

    seq_prior = info["sequence_prior_matrix"]
    item_prior = info["item_prior_matrix"]
    stats = info["graph_prior_stats"]

    _assert(tuple(seq_prior.shape) == (3, 3), "sequence prior should cover train-seen concepts only.")
    _assert(torch.allclose(seq_prior.sum(dim=-1), torch.ones(3), atol=1e-6), "sequence prior rows should sum to 1.")
    _assert(seq_prior[1, 0].item() > seq_prior[1, 2].item(), "incoming row 101 should prefer previous concept 100.")
    _assert(seq_prior[2, 1].item() > seq_prior[2, 0].item(), "incoming row 102 should prefer previous concept 101.")
    _assert(abs(item_prior[0, 1].item() - item_prior[0, 2].item()) < 1e-6, "single-concept items should leave row-0 item prior uninformative.")
    _assert(stats["seq_raw_transition_mass"] > 0.0, "sequence stats should record raw train-only transitions.")
    _assert(stats["seq_student_weighted_mass"] > 0.0, "sequence stats should record reliability-weighted transition mass.")
    _assert(
        stats["seq_student_weighted_mass"] < stats["seq_student_count"],
        "student reliability should down-weight short/noisy trajectories.",
    )
    _assert(stats["item_observed_edge_count"] == 0.0, "single-concept train data should not fake item co-occurrence edges.")


def _check_sequence_prior_drives_relation_learning_support() -> None:
    from src.model import CognitiveDiagnosisModel

    q_matrix = torch.eye(3, dtype=torch.float32)
    item_prior = torch.tensor(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    sequence_prior = torch.tensor(
        [
            [0.0, 0.10, 0.90],
            [0.90, 0.0, 0.10],
            [0.10, 0.90, 0.0],
        ],
        dtype=torch.float32,
    )
    model = CognitiveDiagnosisModel(
        num_students=3,
        num_exercises=3,
        num_concepts=3,
        q_matrix=q_matrix,
        item_prior_matrix=item_prior,
        sequence_prior_matrix=sequence_prior,
        knowledge_dim=8,
        num_relation_heads=1,
        num_gnn_layers=1,
        dropout=0.0,
        graph_dropout=0.0,
        graph_prior_logit_scale=1.0,
        graph_topk=1,
        use_concept_graph=True,
        use_personal_graph=False,
    )
    model.eval()
    relation, _ = model.structure_module.relation_learning()
    row1 = relation[0, 1]
    row2 = relation[0, 2]
    _assert(row1[0].item() > row1[2].item(), "A row 1 should prefer incoming sequence support 0->1.")
    _assert(row2[1].item() > row2[0].item(), "A row 2 should prefer incoming sequence support 1->2.")
    diag = model.structure_module.relation_learning._last_support_diagnostics
    _assert(diag["support_seq_survival_rate"].item() > 0.0, "support diagnostics should track sequence source survival.")
    _assert(diag["support_final_size_mean"].item() >= 1.0, "support diagnostics should track final support size.")


def main() -> None:
    _check_train_only_sequence_prior_handles_single_concept_items()
    _check_sequence_prior_drives_relation_learning_support()
    print("smoke_sequence_prior passed")


if __name__ == "__main__":
    main()
