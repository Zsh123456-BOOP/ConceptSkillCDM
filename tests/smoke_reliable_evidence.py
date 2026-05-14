import pandas as pd
import torch

from src.reliable_evidence import (
    assign_student_quantile_groups,
    build_group_concept_logits,
    build_response_transition_priors,
)


def test_response_transition_prior_uses_incoming_direction_and_correctness_split():
    train_df = pd.DataFrame(
        {
            "stu_id": [1, 1, 1],
            "exer_id": [10, 11, 12],
            "cpt_seq": [0, 1, 2],
            "label": [1, 1, 0],
        }
    )

    result = build_response_transition_priors(
        train_df,
        {0: 0, 1: 1, 2: 2},
        max_hops=1,
        decay=1.0,
        student_reliability_lambda=0.0,
    )

    right = result["right_prior"]
    wrong = result["wrong_prior"]

    assert right[1, 0] > 0.99
    assert wrong[2, 1] > 0.99
    assert right[0].sum().item() == 0.0
    assert wrong[1].sum().item() == 0.0
    assert torch.diag(right).sum().item() == 0.0
    assert torch.diag(wrong).sum().item() == 0.0


def test_student_quantile_groups_are_deterministic_and_bounded():
    logits = torch.tensor([-3.0, -1.0, 0.0, 2.0, 4.0])
    groups = assign_student_quantile_groups(logits, num_groups=3)

    assert groups.tolist() == [0, 0, 1, 1, 2]
    assert int(groups.min().item()) >= 0
    assert int(groups.max().item()) < 3


def test_group_concept_logits_are_finite_and_relative_to_concept_base():
    student_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    exercise_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    q_matrix = torch.eye(2, dtype=torch.float32)
    group_ids = torch.tensor([1, 0], dtype=torch.long)
    concept_rate = torch.tensor([0.5, 0.5], dtype=torch.float32)

    logits = build_group_concept_logits(
        student_ids=student_ids,
        exercise_ids=exercise_ids,
        labels=labels,
        q_matrix=q_matrix,
        student_group_ids=group_ids,
        num_groups=2,
        concept_rate=concept_rate,
        smoothing=1.0,
    )

    assert torch.isfinite(logits).all()
    assert logits[1, 0] > 0.0
    assert logits[0, 0] < 0.0
