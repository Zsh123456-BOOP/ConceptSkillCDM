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
    _assert(seq_prior[0].sum().item() == 0.0, "concepts without observed incoming transitions should not fake support.")
    _assert(torch.allclose(seq_prior[1:].sum(dim=-1), torch.ones(2), atol=1e-6), "observed sequence prior rows should sum to 1.")
    _assert(seq_prior[1, 0].item() > seq_prior[1, 2].item(), "incoming row 101 should prefer previous concept 100.")
    _assert(seq_prior[2, 1].item() > seq_prior[2, 0].item(), "incoming row 102 should prefer previous concept 101.")
    _assert(item_prior.sum().item() == 0.0, "single-concept items should not fake item co-occurrence support.")
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
    item_prior = torch.zeros(3, 3, dtype=torch.float32)
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
        graph_topk=2,
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


def _check_item_support_is_not_pruned_by_dense_sequence_prior() -> None:
    from src.model import CognitiveDiagnosisModel

    q_matrix = torch.eye(4, dtype=torch.float32)
    item_prior = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sequence_prior = torch.tensor(
        [
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.9, 0.1],
            [0.1, 0.9, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = CognitiveDiagnosisModel(
        num_students=3,
        num_exercises=4,
        num_concepts=4,
        q_matrix=q_matrix,
        item_prior_matrix=item_prior,
        sequence_prior_matrix=sequence_prior,
        knowledge_dim=8,
        num_relation_heads=1,
        num_gnn_layers=1,
        dropout=0.0,
        graph_dropout=0.0,
        graph_prior_logit_scale=1.0,
        graph_topk=2,
        use_concept_graph=True,
        use_personal_graph=False,
    )
    model.eval()
    relation, _ = model.structure_module.relation_learning()
    _assert(relation[0, 0, 1].item() > 0.0, "source-balanced top-k should preserve observed item support.")
    diag = model.structure_module.relation_learning._last_support_diagnostics
    _assert(diag["support_item_survival_rate"].item() > 0.0, "item source survival should be tracked and nonzero.")


def _check_relation_heads_start_with_source_roles() -> None:
    from src.model import CognitiveDiagnosisModel
    import torch.nn.functional as F

    q_matrix = torch.eye(4, dtype=torch.float32)
    item_prior = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sequence_prior = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = CognitiveDiagnosisModel(
        num_students=3,
        num_exercises=4,
        num_concepts=4,
        q_matrix=q_matrix,
        item_prior_matrix=item_prior,
        sequence_prior_matrix=sequence_prior,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        graph_dropout=0.0,
        graph_prior_logit_scale=1.0,
        graph_topk=3,
        use_concept_graph=True,
        use_personal_graph=False,
    )
    relation_learning = model.structure_module.relation_learning
    item_beta = F.softplus(relation_learning.prior_strength_raw.detach())
    seq_beta = F.softplus(relation_learning.sequence_prior_strength_raw.detach())
    _assert(item_beta[0].item() > seq_beta[0].item(), "head 0 should start item-heavy.")
    _assert(seq_beta[1].item() > item_beta[1].item(), "head 1 should start sequence-heavy.")
    _assert(item_beta.std(unbiased=False).item() > 0.0, "item beta should expose head source diversity.")
    _assert(seq_beta.std(unbiased=False).item() > 0.0, "sequence beta should expose head source diversity.")


def _check_support_control_priors_keep_expected_support() -> None:
    from src.dataset import make_degree_random_prior, make_support_uniform_prior

    item_prior = torch.tensor(
        [
            [0.0, 0.8, 0.2, 0.0],
            [0.3, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sequence_prior = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.4, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    evidence_support = ((item_prior + sequence_prior) > 0.0) & (~torch.eye(4, dtype=torch.bool))

    support_uniform, support_stats = make_support_uniform_prior(item_prior, sequence_prior)
    _assert(
        torch.equal(support_uniform > 0.0, evidence_support),
        "support_uniform must keep exactly the train-evidence union support.",
    )
    row0 = support_uniform[0]
    _assert(torch.allclose(row0[row0 > 0], torch.full((3,), 1.0 / 3.0)), "support_uniform row should be uniform.")
    _assert(support_stats["support_uniform_edge_count"] == float(evidence_support.sum().item()), "support stats should count evidence edges.")

    degree_random, random_stats = make_degree_random_prior(item_prior, sequence_prior, seed=7)
    _assert(
        torch.equal((degree_random > 0.0).sum(dim=-1), evidence_support.sum(dim=-1)),
        "degree_random must preserve row-wise evidence support degree.",
    )
    _assert(torch.all(torch.diag(degree_random) == 0.0).item(), "degree_random must not create off-protocol self evidence.")
    _assert(random_stats["degree_random_edge_count"] == float(evidence_support.sum().item()), "random stats should count sampled edges.")


def main() -> None:
    _check_train_only_sequence_prior_handles_single_concept_items()
    _check_sequence_prior_drives_relation_learning_support()
    _check_item_support_is_not_pruned_by_dense_sequence_prior()
    _check_relation_heads_start_with_source_roles()
    _check_support_control_priors_keep_expected_support()
    print("smoke_sequence_prior passed")


if __name__ == "__main__":
    main()
