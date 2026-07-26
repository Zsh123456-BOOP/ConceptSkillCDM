#!/usr/bin/env python
"""Synthetic invariants for cross-fitted residual concept relations."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.residual_relation import (
    add_student_excluded_item_residual,
    assign_student_folds,
    build_evidence_state,
    build_exercise_concepts,
    build_relation,
    estimate_relations_from_residuals,
    score_queries,
    sorted_id_map,
    topk_relation_estimate,
    topk_relation_matrix,
)


def _row(student, item, concepts, label):
    return {
        "stu_id": student,
        "exer_id": item,
        "cpt_seq": concepts,
        "label": float(label),
    }


def _direction_frame() -> pd.DataFrame:
    rows = []
    for student in range(30):
        high = int(student % 2 == 0)
        for repeat in range(3):
            rows.append(_row(student, 100 + repeat, "0", high))
            rows.append(_row(student, 200 + repeat, "1", high))
            rows.append(_row(student, 300 + repeat, "2", 1 - high))
    return pd.DataFrame(rows)


def main() -> None:
    dense_matrix = np.asarray(
        (
            (0.0, 0.2, -0.6, 0.1),
            (0.1, 0.0, 0.4, -0.2),
            (0.0, -0.3, 0.0, 0.7),
            (0.0, 0.0, 0.0, 0.0),
        )
    )
    sparse_matrix = topk_relation_matrix(dense_matrix, topk=2)
    assert np.all(np.count_nonzero(sparse_matrix, axis=1) <= 2)
    np.testing.assert_allclose(
        np.abs(sparse_matrix).sum(axis=1),
        np.abs(dense_matrix).sum(axis=1),
        atol=1e-12,
        rtol=0.0,
    )
    assert sparse_matrix[0, 2] < 0.0
    assert sparse_matrix[2, 3] > 0.0
    np.testing.assert_array_equal(
        topk_relation_matrix(dense_matrix, topk=3),
        dense_matrix,
    )

    # Each fold is pruned from its own coefficients.  A stronger edge in a
    # different fold cannot determine this fold's retained support.
    fold_a = dense_matrix.copy()
    fold_b = dense_matrix.copy()
    fold_b[0] = (0.0, 0.9, -0.1, 0.2)
    fold_a_top1 = topk_relation_matrix(fold_a, topk=1)
    fold_b_top1 = topk_relation_matrix(fold_b, topk=1)
    assert np.flatnonzero(fold_a_top1[0]).tolist() == [2]
    assert np.flatnonzero(fold_b_top1[0]).tolist() == [1]

    frame = _direction_frame()
    concept_map = sorted_id_map((0, 1, 2, 3))
    relation = build_relation(frame, concept_map)
    assert relation.raw[1, 0] > 0.0

    shared_only = pd.DataFrame(
        [
            _row(student, 10 + repeat, "0,1", (student + repeat) % 2)
            for student in range(8)
            for repeat in range(3)
        ]
    )
    shared_relation = build_relation(
        shared_only,
        sorted_id_map((0, 1)),
        min_pair_students=2,
    )
    assert np.count_nonzero(shared_relation.raw) == 0
    assert np.count_nonzero(shared_relation.partial) == 0

    # General student ability induces raw cross-concept correlation.  A third
    # disjoint concept supplies the nuisance variable and partials it out.
    ability_rows = []
    for student in range(30):
        high = int(student % 2 == 0)
        for repeat in range(3):
            for concept in range(3):
                ability_rows.append(
                    _row(student, 100 * concept + repeat, str(concept), high)
                )
    ability = build_relation(
        pd.DataFrame(ability_rows),
        sorted_id_map((0, 1, 2)),
    )
    assert ability.raw[1, 0] > 0.0
    assert abs(ability.partial[1, 0]) < abs(ability.raw[1, 0])

    # A non-zero cross moment is not covariance.  Independent source/target
    # deviations with non-zero means must not create a relation.
    independent_rows = []
    for student, (source, target) in enumerate(
        ((0.2, 0.3), (0.2, 0.5), (0.4, 0.3), (0.4, 0.5))
    ):
        independent_rows.extend(
            (
                {"stu_id": student, "concepts": (0,), "residual": source},
                {"stu_id": student, "concepts": (1,), "residual": target},
            )
        )
    independent = estimate_relations_from_residuals(
        pd.DataFrame(independent_rows),
        sorted_id_map((0, 1)),
        min_pair_students=2,
    )
    assert abs(independent.raw[1, 0]) < 1e-12
    assert abs(independent.raw[0, 1]) < 1e-12

    # Independent x, z and g must remain unrelated.  Directly subtracting the
    # same noisy g from x and z would create a positive spurious relation.
    nuisance_rows = []
    student = 0
    for source in (-0.6, 0.6):
        for target in (-0.4, 0.4):
            for nuisance in (-0.8, 0.8):
                nuisance_rows.extend(
                    (
                        {"stu_id": student, "concepts": (0,), "residual": source},
                        {"stu_id": student, "concepts": (1,), "residual": target},
                        {"stu_id": student, "concepts": (2,), "residual": nuisance},
                    )
                )
                student += 1
    nuisance_relation = estimate_relations_from_residuals(
        pd.DataFrame(nuisance_rows),
        sorted_id_map((0, 1, 2)),
        min_pair_students=2,
    )
    assert abs(nuisance_relation.partial[1, 0]) < 1e-12
    assert abs(nuisance_relation.partial[0, 1]) < 1e-12

    # The regressor is the per-student source mean, not its noisy row-level
    # second moment.  A true unit slope must survive large within-source noise.
    mean_slope_rows = []
    for student, source_mean in enumerate(np.linspace(-0.45, 0.45, 40)):
        for repeat in range(20):
            noise = 0.4 if repeat % 2 == 0 else -0.4
            mean_slope_rows.append(
                {
                    "stu_id": student,
                    "concepts": (0,),
                    "residual": source_mean + noise,
                }
            )
        mean_slope_rows.extend(
            (
                {"stu_id": student, "concepts": (1,), "residual": source_mean},
                {"stu_id": student, "concepts": (2,), "residual": 0.0},
            )
        )
    mean_slope = estimate_relations_from_residuals(
        pd.DataFrame(mean_slope_rows),
        sorted_id_map((0, 1, 2)),
    )
    assert 0.95 < mean_slope.partial[1, 0] <= 1.0

    # The fixed support floor has an exact boundary at 20 students.
    support_rows = []
    for student in range(20):
        source = -0.5 if student % 2 else 0.5
        support_rows.extend(
            (
                {"stu_id": student, "concepts": (0,), "residual": source},
                {"stu_id": student, "concepts": (1,), "residual": source},
                {"stu_id": student, "concepts": (2,), "residual": 0.0},
            )
        )
    support_19 = estimate_relations_from_residuals(
        pd.DataFrame(support_rows[:-3]),
        sorted_id_map((0, 1, 2)),
    )
    support_20 = estimate_relations_from_residuals(
        pd.DataFrame(support_rows),
        sorted_id_map((0, 1, 2)),
    )
    assert support_19.raw[1, 0] == 0.0
    assert support_19.partial[1, 0] == 0.0
    assert support_20.raw[1, 0] > 0.0
    assert support_20.partial[1, 0] > 0.0

    assignment = assign_student_folds(
        frame["stu_id"].unique(),
        folds=5,
        seed=42,
    )
    repeated_assignment = assign_student_folds(
        frame["stu_id"].unique(),
        folds=5,
        seed=42,
    )
    other_seed_assignment = assign_student_folds(
        frame["stu_id"].unique(),
        folds=5,
        seed=43,
    )
    assert assignment == repeated_assignment
    assert any(
        assignment[student] != other_seed_assignment[student]
        for student in assignment
    )
    fold_sizes = [
        sum(fold == value for fold in assignment.values())
        for value in range(5)
    ]
    assert max(fold_sizes) - min(fold_sizes) <= 1

    held_out = {student for student, fold in assignment.items() if fold == 0}
    complement = sorted(set(frame["stu_id"]) - held_out)
    before = build_relation(frame, concept_map, included_students=complement)
    flipped = frame.copy()
    flipped.loc[flipped["stu_id"].isin(held_out), "label"] = (
        1.0 - flipped.loc[flipped["stu_id"].isin(held_out), "label"]
    )
    after = build_relation(flipped, concept_map, included_students=complement)
    np.testing.assert_array_equal(before.raw, after.raw)
    np.testing.assert_array_equal(before.partial, after.partial)

    # Rebuild the full evidence state after flipping the current outcome.  The
    # held-out fold relation and exact-LOO direct channels must jointly make
    # this query's rate/base/raw/partial logits outcome-invariant.
    q_by_item = build_exercise_concepts(frame)
    residual_before = add_student_excluded_item_residual(
        frame,
        exercise_concepts=q_by_item,
    )
    query_student = min(held_out)
    query_position = int(frame.index[frame["stu_id"] == query_student][0])
    query_fold = assignment[query_student]
    assert query_fold == 0
    current_flipped = frame.copy()
    current_flipped.loc[query_position, "label"] = (
        1.0 - current_flipped.loc[query_position, "label"]
    )
    residual_after = add_student_excluded_item_residual(
        current_flipped,
        exercise_concepts=q_by_item,
    )
    before_scores = score_queries(
        residual_before.iloc[[query_position]].reset_index(drop=True),
        build_evidence_state(residual_before, concept_map),
        concept_map,
        {query_fold: before},
        {query_student: query_fold},
        leave_one_out=True,
    )
    after_scores = score_queries(
        residual_after.iloc[[query_position]].reset_index(drop=True),
        build_evidence_state(residual_after, concept_map),
        concept_map,
        {query_fold: after},
        {query_student: query_fold},
        leave_one_out=True,
    )
    for left, right in (
        (before_scores.rate_evidence, after_scores.rate_evidence),
        (before_scores.base_logit, after_scores.base_logit),
        (before_scores.raw_logit, after_scores.raw_logit),
        (before_scores.partial_logit, after_scores.partial_logit),
    ):
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=0.0)

    topk_before = topk_relation_estimate(before, topk=1)
    topk_after = topk_relation_estimate(after, topk=1)
    np.testing.assert_array_equal(topk_before.raw, topk_after.raw)
    np.testing.assert_array_equal(topk_before.partial, topk_after.partial)
    before_topk_scores = score_queries(
        residual_before.iloc[[query_position]].reset_index(drop=True),
        build_evidence_state(residual_before, concept_map),
        concept_map,
        {query_fold: topk_before},
        {query_student: query_fold},
        leave_one_out=True,
    )
    after_topk_scores = score_queries(
        residual_after.iloc[[query_position]].reset_index(drop=True),
        build_evidence_state(residual_after, concept_map),
        concept_map,
        {query_fold: topk_after},
        {query_student: query_fold},
        leave_one_out=True,
    )
    for left, right in (
        (before_topk_scores.rate_evidence, after_topk_scores.rate_evidence),
        (before_topk_scores.base_logit, after_topk_scores.base_logit),
        (before_topk_scores.raw_logit, after_topk_scores.raw_logit),
        (before_topk_scores.partial_logit, after_topk_scores.partial_logit),
    ):
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=0.0)

    for matrix in (
        relation.raw,
        relation.partial,
        shared_relation.raw,
        shared_relation.partial,
    ):
        assert np.isfinite(matrix).all()
        assert np.allclose(np.diag(matrix), 0.0)
        assert np.all(np.abs(matrix).sum(axis=1) <= 1.0 + 1e-12)
    assert np.count_nonzero(relation.raw[concept_map[3]]) == 0
    assert np.count_nonzero(relation.partial[concept_map[3]]) == 0
    print("smoke_residual_relation: PASS")


if __name__ == "__main__":
    main()
