"""Cross-fitted concept relations estimated from response residuals.

This module is deliberately independent of the training/model stack.  It fits
signed source-to-target relations from train-only response residuals while
enforcing three boundaries:

* an item's expected correctness excludes every response from the query
  student, matching :func:`src.dataset.build_student_concept_response_stats`;
* a source/target pair uses only pair-exclusive responses;
* an out-of-fold relation never reads responses from students in that fold.

The returned matrix is indexed as ``relation[target, source]`` and has
off-diagonal row L1 norm at most one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12
RESIDUAL_RELATION_FOLDS = 5
RESIDUAL_RELATION_MIN_PAIR_STUDENTS = 20


@dataclass(frozen=True)
class RelationEstimate:
    """Raw and ability-partialled residual relations plus fit diagnostics."""

    raw: np.ndarray
    partial: np.ndarray
    raw_students: np.ndarray
    partial_students: np.ndarray
    raw_weight: np.ndarray
    partial_weight: np.ndarray


@dataclass(frozen=True)
class EvidenceState:
    """Train-only residual sufficient statistics."""

    student_map: Dict
    residual_sum: np.ndarray
    count: np.ndarray
    student_concept_correct: np.ndarray
    concept_correct: np.ndarray
    concept_count: np.ndarray
    global_correct: float
    global_count: float


@dataclass(frozen=True)
class QueryScores:
    """Leakage-controlled logits and target evidence counts."""

    labels: np.ndarray
    target_count: np.ndarray
    rate_evidence: np.ndarray
    base_logit: np.ndarray
    raw_logit: np.ndarray
    partial_logit: np.ndarray


def topk_relation_matrix(matrix: np.ndarray, topk: int) -> np.ndarray:
    """Keep each target row's strongest signed edges without changing its L1 mass."""

    relation = np.asarray(matrix, dtype=np.float64)
    if relation.ndim != 2 or relation.shape[0] != relation.shape[1]:
        raise ValueError("relation matrix must be square")
    if topk < 1:
        raise ValueError("topk must be positive")
    concepts = relation.shape[0]
    result = relation.copy()
    np.fill_diagonal(result, 0.0)
    if topk >= concepts - 1:
        return result

    result.fill(0.0)
    for target, row in enumerate(relation):
        sources = np.flatnonzero((np.abs(row) > EPS) & (np.arange(concepts) != target))
        if sources.size <= topk:
            kept = sources
        else:
            order = np.lexsort((sources, -np.abs(row[sources])))
            kept = sources[order[:topk]]
        if kept.size == 0:
            continue
        original_l1 = float(np.abs(row[sources]).sum())
        kept_l1 = float(np.abs(row[kept]).sum())
        result[target, kept] = row[kept] * (original_l1 / kept_l1)
    return result


def topk_relation_estimate(
    estimate: RelationEstimate,
    topk: int,
) -> RelationEstimate:
    """Prune raw and partial matrices independently, preserving fit metadata."""

    return RelationEstimate(
        raw=topk_relation_matrix(estimate.raw, topk),
        partial=topk_relation_matrix(estimate.partial, topk),
        raw_students=estimate.raw_students,
        partial_students=estimate.partial_students,
        raw_weight=estimate.raw_weight,
        partial_weight=estimate.partial_weight,
    )


def parse_concepts(value) -> Tuple[int, ...]:
    """Parse one ``cpt_seq`` value into a stable, duplicate-free tuple."""

    if isinstance(value, (list, tuple, set, np.ndarray)):
        tokens: Iterable = value
    elif pd.isna(value):
        tokens = ()
    else:
        tokens = str(value).split(",")
    return tuple(sorted({int(token) for token in tokens if str(token).strip()}))


def sorted_id_map(values: Iterable) -> Dict:
    """Map sorted external IDs to contiguous internal IDs."""

    return {value: index for index, value in enumerate(sorted(set(values)))}


def assign_student_folds(
    student_ids: Iterable,
    folds: int = 5,
    seed: int = 42,
) -> Dict:
    """Assign label-independent folds by permuting sorted internal IDs."""

    if folds < 2:
        raise ValueError(f"folds must be at least 2, got {folds}")
    sorted_students = np.asarray(sorted(set(student_ids)), dtype=object)
    permutation = np.random.default_rng(seed).permutation(len(sorted_students))
    return {
        sorted_students[internal_id]: int(position % folds)
        for position, internal_id in enumerate(permutation)
    }


def build_exercise_concepts(frame: pd.DataFrame) -> Dict:
    """Build the train-only item-to-Q mapping used for every query row."""

    if "exer_id" not in frame or "cpt_seq" not in frame:
        raise ValueError("Q construction requires exer_id and cpt_seq")
    mapping: Dict = {}
    for exercise, values in frame.groupby("exer_id", sort=False)["cpt_seq"]:
        concepts = set()
        for value in values:
            concepts.update(parse_concepts(value))
        if concepts:
            mapping[exercise] = tuple(sorted(concepts))
    return mapping


def add_student_excluded_item_residual(
    frame: pd.DataFrame,
    *,
    exercise_concepts: Optional[Mapping] = None,
) -> pd.DataFrame:
    """Return mapped rows with student-excluded item expectation and residual.

    The calculation exactly mirrors ``src/dataset.py``:

    ``p_si = (correct_item_without_s + rate_global_without_s) /
             (count_item_without_s + 1)``.
    """

    required = {"stu_id", "exer_id", "cpt_seq", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"interaction frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("cannot estimate residuals from an empty frame")

    work = frame.loc[:, ["stu_id", "exer_id", "cpt_seq", "label"]].copy()
    work["label"] = pd.to_numeric(work["label"], errors="raise").astype(np.float64)
    if not work["label"].between(0.0, 1.0).all():
        raise ValueError("labels must lie in [0, 1]")
    q_by_item = (
        build_exercise_concepts(work)
        if exercise_concepts is None
        else exercise_concepts
    )
    work["concepts"] = work["exer_id"].map(q_by_item)
    work = work[
        work["concepts"].map(lambda value: isinstance(value, tuple) and bool(value))
    ].reset_index(drop=True)
    if work.empty:
        raise ValueError("no interaction has a mapped concept")

    student = work.groupby("stu_id", sort=False)["label"].agg(["sum", "count"])
    item = work.groupby("exer_id", sort=False)["label"].agg(["sum", "count"])
    pair = work.groupby(["stu_id", "exer_id"], sort=False)["label"].agg(["sum", "count"])

    work = work.join(student.rename(columns={"sum": "s_sum", "count": "s_count"}), on="stu_id")
    work = work.join(item.rename(columns={"sum": "i_sum", "count": "i_count"}), on="exer_id")
    work = work.join(
        pair.rename(columns={"sum": "si_sum", "count": "si_count"}),
        on=["stu_id", "exer_id"],
    )
    global_sum = float(work["label"].sum())
    global_count = float(len(work))
    other_student_count = global_count - work["s_count"].to_numpy(dtype=np.float64)
    other_student_sum = global_sum - work["s_sum"].to_numpy(dtype=np.float64)
    student_rate = np.divide(
        other_student_sum,
        other_student_count,
        out=np.full(len(work), 0.5, dtype=np.float64),
        where=other_student_count > 0.0,
    )
    item_other_count = (
        work["i_count"].to_numpy(dtype=np.float64)
        - work["si_count"].to_numpy(dtype=np.float64)
    )
    item_other_sum = (
        work["i_sum"].to_numpy(dtype=np.float64)
        - work["si_sum"].to_numpy(dtype=np.float64)
    )
    expected = (item_other_sum + student_rate) / (item_other_count + 1.0)
    work["item_expectation"] = expected
    work["residual"] = work["label"].to_numpy(dtype=np.float64) - expected
    return work.loc[
        :, ["stu_id", "exer_id", "label", "concepts", "item_expectation", "residual"]
    ]


def student_excluded_item_expectation(
    train: pd.DataFrame,
    queries: pd.DataFrame,
) -> np.ndarray:
    """Compute validation item expectations from train responses only."""

    train_label = pd.to_numeric(train["label"], errors="raise").astype(np.float64)
    student = train.assign(label=train_label).groupby("stu_id")["label"].agg(["sum", "count"])
    item = train.assign(label=train_label).groupby("exer_id")["label"].agg(["sum", "count"])
    pair = (
        train.assign(label=train_label)
        .groupby(["stu_id", "exer_id"])["label"]
        .agg(["sum", "count"])
    )
    student_sum = queries["stu_id"].map(student["sum"]).to_numpy(dtype=np.float64)
    student_count = queries["stu_id"].map(student["count"]).to_numpy(dtype=np.float64)
    item_sum = queries["exer_id"].map(item["sum"]).to_numpy(dtype=np.float64)
    item_count = queries["exer_id"].map(item["count"]).to_numpy(dtype=np.float64)
    query_pairs = pd.MultiIndex.from_frame(queries[["stu_id", "exer_id"]])
    pair_stats = pair.reindex(query_pairs).fillna(0.0)
    pair_sum = pair_stats["sum"].to_numpy(dtype=np.float64)
    pair_count = pair_stats["count"].to_numpy(dtype=np.float64)
    global_sum = float(train_label.sum())
    global_count = float(len(train))
    other_student_count = global_count - student_count
    student_rate = np.divide(
        global_sum - student_sum,
        other_student_count,
        out=np.full(len(queries), 0.5, dtype=np.float64),
        where=other_student_count > 0.0,
    )
    return (item_sum - pair_sum + student_rate) / (item_count - pair_count + 1.0)


def build_evidence_state(
    residual_frame: pd.DataFrame,
    concept_map: Dict[int, int],
) -> EvidenceState:
    """Aggregate residual sums/counts without row-by-concept dense expansion."""

    student_map = sorted_id_map(residual_frame["stu_id"].unique())
    residual_sum = np.zeros((len(student_map), len(concept_map)), dtype=np.float64)
    count = np.zeros_like(residual_sum)
    correct = np.zeros_like(residual_sum)
    concept_correct = np.zeros(len(concept_map), dtype=np.float64)
    concept_count = np.zeros(len(concept_map), dtype=np.float64)
    for student, concepts, label, residual in zip(
        residual_frame["stu_id"].values,
        residual_frame["concepts"].values,
        residual_frame["label"].values,
        residual_frame["residual"].values,
    ):
        encoded = [concept_map[c] for c in concepts if c in concept_map]
        if not encoded:
            continue
        row = student_map[student]
        residual_sum[row, encoded] += float(residual)
        count[row, encoded] += 1.0
        correct[row, encoded] += float(label)
        concept_correct[encoded] += float(label)
        concept_count[encoded] += 1.0
    return EvidenceState(
        student_map=student_map,
        residual_sum=residual_sum,
        count=count,
        student_concept_correct=correct,
        concept_correct=concept_correct,
        concept_count=concept_count,
        global_correct=float(residual_frame["label"].sum()),
        global_count=float(len(residual_frame)),
    )


def _logit(probability: float) -> float:
    clipped = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def _rate_evidence(
    student_correct: np.ndarray,
    count: np.ndarray,
    concept_correct: np.ndarray,
    concept_count: np.ndarray,
    global_rate: float,
) -> np.ndarray:
    """Mirror ``model_cdm._build_response_evidence`` for selected concepts."""

    concept_rate = (concept_correct + global_rate) / (concept_count + 1.0)
    posterior = (student_correct + concept_rate) / (count + 1.0)
    reliability = count / (count + 1.0)
    concept_logit = np.log(
        np.clip(concept_rate, 1e-4, 1.0 - 1e-4)
        / np.clip(1.0 - concept_rate, 1e-4, 1.0 - 1e-4)
    )
    posterior_logit = np.log(
        np.clip(posterior, 1e-4, 1.0 - 1e-4)
        / np.clip(1.0 - posterior, 1e-4, 1.0 - 1e-4)
    )
    return np.clip(
        (posterior_logit - concept_logit) * reliability,
        -4.0,
        4.0,
    )


def score_queries(
    queries: pd.DataFrame,
    evidence: EvidenceState,
    concept_map: Dict[int, int],
    relations: Mapping[int, RelationEstimate],
    relation_group_by_student: Mapping,
    *,
    leave_one_out: bool,
) -> QueryScores:
    """Score OOF train or full-graph validation queries.

    ``queries`` must contain canonical ``concepts`` and ``item_expectation``.
    Training queries additionally contain their current-row ``residual``.
    """

    required = {"stu_id", "label", "concepts", "item_expectation"}
    if leave_one_out:
        required.add("residual")
    missing = sorted(required - set(queries.columns))
    if missing:
        raise ValueError(f"query frame is missing columns: {missing}")

    size = len(queries)
    labels = queries["label"].to_numpy(dtype=np.float64)
    target_count = np.zeros(size, dtype=np.float64)
    mean_rate_evidence = np.zeros(size, dtype=np.float64)
    base_logit = np.zeros(size, dtype=np.float64)
    raw_logit = np.zeros(size, dtype=np.float64)
    partial_logit = np.zeros(size, dtype=np.float64)

    concept_values = queries["concepts"].values
    item_expectation = queries["item_expectation"].to_numpy(dtype=np.float64)
    query_residual = (
        queries["residual"].to_numpy(dtype=np.float64)
        if leave_one_out
        else None
    )
    grouped_positions = queries.groupby("stu_id", sort=False).indices
    for student, positions in grouped_positions.items():
        if student not in evidence.student_map:
            raise ValueError(f"query student is not present in evidence state: {student!r}")
        group = relation_group_by_student[student]
        relation = relations[group]
        student_row = evidence.student_map[student]
        sums = evidence.residual_sum[student_row]
        counts = evidence.count[student_row]
        correct = evidence.student_concept_correct[student_row]
        z_full = np.divide(sums, counts + 1.0)
        full_rate = _rate_evidence(
            correct,
            counts,
            evidence.concept_correct,
            evidence.concept_count,
            evidence.global_correct / evidence.global_count,
        )
        active = np.flatnonzero(counts > 0.0)
        if active.size:
            raw_prop = relation.raw[:, active] @ z_full[active]
            partial_prop = relation.partial[:, active] @ z_full[active]
        else:
            raw_prop = np.zeros(len(concept_map), dtype=np.float64)
            partial_prop = np.zeros(len(concept_map), dtype=np.float64)

        for position in positions:
            concepts = np.asarray(
                [
                    concept_map[c]
                    for c in concept_values[position]
                    if c in concept_map
                ],
                dtype=np.int64,
            )
            if concepts.size == 0:
                raise ValueError("query row has no train-mapped concept")
            if leave_one_out:
                label = float(labels[position])
                residual = float(query_residual[position])
                loo_count = counts[concepts] - 1.0
                if np.any(loo_count < 0.0):
                    raise ValueError("current row is absent from its concept evidence")
                loo_z = np.divide(
                    sums[concepts] - residual,
                    loo_count + 1.0,
                )
                delta = loo_z - z_full[concepts]
                raw_target = (
                    raw_prop[concepts]
                    + relation.raw[np.ix_(concepts, concepts)] @ delta
                )
                partial_target = (
                    partial_prop[concepts]
                    + relation.partial[np.ix_(concepts, concepts)] @ delta
                )
                query_count = loo_count
                global_count = max(evidence.global_count - 1.0, 1.0)
                global_rate = (evidence.global_correct - label) / global_count
                rate = _rate_evidence(
                    np.maximum(correct[concepts] - label, 0.0),
                    np.maximum(loo_count, 0.0),
                    np.maximum(evidence.concept_correct[concepts] - label, 0.0),
                    np.maximum(evidence.concept_count[concepts] - 1.0, 0.0),
                    global_rate,
                )
                residual_direct = loo_z
            else:
                query_count = counts[concepts]
                raw_target = raw_prop[concepts]
                partial_target = partial_prop[concepts]
                rate = full_rate[concepts]
                residual_direct = z_full[concepts]

            mean_rate = float(rate.mean())
            base = (
                _logit(item_expectation[position])
                + float(np.mean(rate + residual_direct))
            )
            raw_addition = float(np.mean(raw_target / (query_count + 1.0)))
            partial_addition = float(np.mean(partial_target / (query_count + 1.0)))
            base_logit[position] = base
            raw_logit[position] = base + raw_addition
            partial_logit[position] = base + partial_addition
            target_count[position] = float(query_count.min())
            mean_rate_evidence[position] = mean_rate

    return QueryScores(
        labels=labels,
        target_count=target_count,
        rate_evidence=mean_rate_evidence,
        base_logit=base_logit,
        raw_logit=raw_logit,
        partial_logit=partial_logit,
    )


def _finalize_relation(
    cross_moment: np.ndarray,
    source_second_moment: np.ndarray,
    source_sum: np.ndarray,
    target_sum: np.ndarray,
    weight: np.ndarray,
    students: np.ndarray,
    min_pair_students: int,
) -> np.ndarray:
    covariance = cross_moment - np.divide(
        source_sum * target_sum,
        weight,
        out=np.zeros_like(cross_moment),
        where=weight > EPS,
    )
    variance = source_second_moment - np.divide(
        source_sum * source_sum,
        weight,
        out=np.zeros_like(source_second_moment),
        where=weight > EPS,
    )
    relation = np.zeros_like(cross_moment)
    valid = (
        (students >= min_pair_students)
        & (weight > EPS)
        & (variance > EPS)
        & np.isfinite(covariance)
    )
    relation[valid] = (
        covariance[valid]
        / variance[valid]
        * weight[valid]
        / (weight[valid] + 1.0)
    )
    np.fill_diagonal(relation, 0.0)
    relation[~np.isfinite(relation)] = 0.0
    row_l1 = np.abs(relation).sum(axis=1, keepdims=True)
    relation /= np.maximum(1.0, row_l1)
    return relation


def _finalize_partial_relation(
    cross_moment: np.ndarray,
    source_second_moment: np.ndarray,
    source_sum: np.ndarray,
    target_sum: np.ndarray,
    nuisance_sum: np.ndarray,
    nuisance_second_moment: np.ndarray,
    source_nuisance_moment: np.ndarray,
    target_nuisance_moment: np.ndarray,
    weight: np.ndarray,
    students: np.ndarray,
    min_pair_students: int,
) -> np.ndarray:
    """Fit the weighted source slope after partialling out student ability."""

    safe_weight = np.where(weight > EPS, weight, 1.0)
    covariance_xz = cross_moment - source_sum * target_sum / safe_weight
    variance_x = (
        source_second_moment - source_sum * source_sum / safe_weight
    )
    covariance_xg = (
        source_nuisance_moment - source_sum * nuisance_sum / safe_weight
    )
    covariance_zg = (
        target_nuisance_moment - target_sum * nuisance_sum / safe_weight
    )
    variance_g = (
        nuisance_second_moment - nuisance_sum * nuisance_sum / safe_weight
    )
    partial_covariance = covariance_xz.copy()
    partial_variance = variance_x.copy()
    has_nuisance_variance = variance_g > EPS
    partial_covariance[has_nuisance_variance] -= (
        covariance_xg[has_nuisance_variance]
        * covariance_zg[has_nuisance_variance]
        / variance_g[has_nuisance_variance]
    )
    partial_variance[has_nuisance_variance] -= (
        covariance_xg[has_nuisance_variance] ** 2
        / variance_g[has_nuisance_variance]
    )

    relation = np.zeros_like(cross_moment)
    valid = (
        (students >= min_pair_students)
        & (weight > EPS)
        & (partial_variance > EPS)
        & np.isfinite(partial_covariance)
    )
    relation[valid] = (
        partial_covariance[valid]
        / partial_variance[valid]
        * weight[valid]
        / (weight[valid] + 1.0)
    )
    np.fill_diagonal(relation, 0.0)
    relation[~np.isfinite(relation)] = 0.0
    row_l1 = np.abs(relation).sum(axis=1, keepdims=True)
    relation /= np.maximum(1.0, row_l1)
    return relation


def estimate_relations_from_residuals(
    residual_frame: pd.DataFrame,
    concept_map: Dict[int, int],
    *,
    min_pair_students: int = 20,
) -> RelationEstimate:
    """Fit strict pair-exclusive raw and ability-partialled relations."""

    required = {"stu_id", "concepts", "residual"}
    missing = sorted(required - set(residual_frame.columns))
    if missing:
        raise ValueError(f"residual frame is missing columns: {missing}")
    if min_pair_students < 2:
        raise ValueError("min_pair_students must be at least 2")
    concept_count = len(concept_map)
    shape = (concept_count, concept_count)
    raw_c = np.zeros(shape, dtype=np.float64)
    raw_v = np.zeros(shape, dtype=np.float64)
    raw_x = np.zeros(shape, dtype=np.float64)
    raw_z = np.zeros(shape, dtype=np.float64)
    raw_w = np.zeros(shape, dtype=np.float64)
    raw_n = np.zeros(shape, dtype=np.int64)
    partial_c = np.zeros(shape, dtype=np.float64)
    partial_v = np.zeros(shape, dtype=np.float64)
    partial_x = np.zeros(shape, dtype=np.float64)
    partial_z = np.zeros(shape, dtype=np.float64)
    partial_g = np.zeros(shape, dtype=np.float64)
    partial_gg = np.zeros(shape, dtype=np.float64)
    partial_xg = np.zeros(shape, dtype=np.float64)
    partial_zg = np.zeros(shape, dtype=np.float64)
    partial_w = np.zeros(shape, dtype=np.float64)
    partial_n = np.zeros(shape, dtype=np.int64)

    for _, student_rows in residual_frame.groupby("stu_id", sort=False):
        encoded_rows = []
        local_set = set()
        for concepts, residual in zip(
            student_rows["concepts"].values,
            student_rows["residual"].values,
        ):
            encoded = tuple(concept_map[c] for c in concepts if c in concept_map)
            if not encoded:
                continue
            encoded_rows.append((encoded, float(residual)))
            local_set.update(encoded)
        if len(local_set) < 2:
            continue

        global_ids = np.asarray(sorted(local_set), dtype=np.int64)
        local_index = {concept: index for index, concept in enumerate(global_ids)}
        size = len(global_ids)
        support = np.zeros(size, dtype=np.float64)
        weighted_sum = np.zeros(size, dtype=np.float64)
        contain_count = np.zeros(size, dtype=np.float64)
        contain_sum = np.zeros(size, dtype=np.float64)
        shared_support = np.zeros((size, size), dtype=np.float64)
        shared_sum = np.zeros((size, size), dtype=np.float64)
        shared_row_count = np.zeros((size, size), dtype=np.float64)
        shared_row_sum = np.zeros((size, size), dtype=np.float64)
        total_sum = 0.0

        for concepts, residual in encoded_rows:
            local = np.asarray([local_index[c] for c in concepts], dtype=np.int64)
            allocation = 1.0 / float(len(local))
            support[local] += allocation
            weighted_sum[local] += allocation * residual
            contain_count[local] += 1.0
            contain_sum[local] += residual
            block = np.ix_(local, local)
            shared_support[block] += allocation
            shared_sum[block] += allocation * residual
            shared_row_count[block] += 1.0
            shared_row_sum[block] += residual
            total_sum += residual

        # Local matrices use [source, target]; global accumulators use
        # [target, source], hence the transpose at accumulation.
        source_support = support[:, None] - shared_support
        target_support = support[None, :] - shared_support
        pair_valid = (
            (source_support > EPS)
            & (target_support > EPS)
            & ~np.eye(size, dtype=bool)
        )
        source_mean = np.divide(
            weighted_sum[:, None] - shared_sum,
            source_support,
            out=np.zeros((size, size), dtype=np.float64),
            where=source_support > EPS,
        )
        target_mean = np.divide(
            weighted_sum[None, :] - shared_sum,
            target_support,
            out=np.zeros((size, size), dtype=np.float64),
            where=target_support > EPS,
        )
        mass = source_support * target_support
        omega = np.divide(mass, mass + 1.0, out=np.zeros_like(mass), where=mass > 0.0)

        raw_local_c = omega * source_mean * target_mean * pair_valid
        raw_local_v = omega * source_mean * source_mean * pair_valid
        block = np.ix_(global_ids, global_ids)
        raw_c[block] += raw_local_c.T
        raw_v[block] += raw_local_v.T
        raw_x[block] += (omega * source_mean * pair_valid).T
        raw_z[block] += (omega * target_mean * pair_valid).T
        raw_w[block] += (omega * pair_valid).T
        raw_n[block] += pair_valid.T.astype(np.int64)

        other_count = (
            float(len(encoded_rows))
            - contain_count[:, None]
            - contain_count[None, :]
            + shared_row_count
        )
        other_sum = (
            total_sum
            - contain_sum[:, None]
            - contain_sum[None, :]
            + shared_row_sum
        )
        partial_valid = pair_valid & (other_count > 0.0)
        nuisance = np.divide(
            other_sum,
            other_count,
            out=np.zeros_like(other_sum),
            where=other_count > 0.0,
        )
        partial_c[block] += (
            omega * source_mean * target_mean * partial_valid
        ).T
        partial_v[block] += (
            omega * source_mean * source_mean * partial_valid
        ).T
        partial_x[block] += (omega * source_mean * partial_valid).T
        partial_z[block] += (omega * target_mean * partial_valid).T
        partial_g[block] += (omega * nuisance * partial_valid).T
        partial_gg[block] += (omega * nuisance * nuisance * partial_valid).T
        partial_xg[block] += (
            omega * source_mean * nuisance * partial_valid
        ).T
        partial_zg[block] += (
            omega * target_mean * nuisance * partial_valid
        ).T
        partial_w[block] += (omega * partial_valid).T
        partial_n[block] += partial_valid.T.astype(np.int64)

    raw = _finalize_relation(
        raw_c,
        raw_v,
        raw_x,
        raw_z,
        raw_w,
        raw_n,
        min_pair_students,
    )
    partial = _finalize_partial_relation(
        partial_c,
        partial_v,
        partial_x,
        partial_z,
        partial_g,
        partial_gg,
        partial_xg,
        partial_zg,
        partial_w,
        partial_n,
        min_pair_students,
    )
    return RelationEstimate(
        raw=raw,
        partial=partial,
        raw_students=raw_n,
        partial_students=partial_n,
        raw_weight=raw_w,
        partial_weight=partial_w,
    )


def build_relation(
    train: pd.DataFrame,
    concept_map: Dict[int, int],
    *,
    included_students: Optional[Sequence] = None,
    exercise_concepts: Optional[Mapping] = None,
    min_pair_students: int = 20,
) -> RelationEstimate:
    """Recompute residuals on the selected raw rows, then fit relations."""

    subset = train
    if included_students is not None:
        included = set(included_students)
        subset = train[train["stu_id"].isin(included)]
    residuals = add_student_excluded_item_residual(
        subset.reset_index(drop=True),
        exercise_concepts=exercise_concepts,
    )
    return estimate_relations_from_residuals(
        residuals,
        concept_map,
        min_pair_students=min_pair_students,
    )


def build_cross_fitted_relations(
    train: pd.DataFrame,
    concept_map: Dict[int, int],
    folds: int = 5,
    seed: int = 42,
    min_pair_students: int = 20,
) -> Tuple[RelationEstimate, Dict[int, RelationEstimate], Dict]:
    """Build one full relation and one complement-student relation per fold."""

    assignment = assign_student_folds(
        train["stu_id"].unique(),
        folds=folds,
        seed=seed,
    )
    q_by_item = build_exercise_concepts(train)
    full = build_relation(
        train,
        concept_map,
        exercise_concepts=q_by_item,
        min_pair_students=min_pair_students,
    )
    by_fold = {}
    for fold in range(folds):
        complement = [student for student, value in assignment.items() if value != fold]
        by_fold[fold] = build_relation(
            train,
            concept_map,
            included_students=complement,
            exercise_concepts=q_by_item,
            min_pair_students=min_pair_students,
        )
    return full, by_fold, assignment


def _validate_internal_id_map(
    name: str,
    id_map: Mapping,
) -> None:
    if not id_map:
        raise ValueError(f"{name} must not be empty")
    internal_ids = []
    for raw_id, internal_id in id_map.items():
        if pd.isna(raw_id):
            raise ValueError(f"{name} contains a missing external ID")
        if isinstance(internal_id, (bool, np.bool_)) or not isinstance(
            internal_id,
            (int, np.integer),
        ):
            raise ValueError(f"{name} internal IDs must be integers")
        internal_ids.append(int(internal_id))
    if sorted(internal_ids) != list(range(len(id_map))):
        raise ValueError(
            f"{name} internal IDs must be unique and contiguous from zero"
        )


def _relation_bundle_diagnostics(
    full: np.ndarray,
    fold_relations: np.ndarray,
) -> Dict[str, float]:
    """Summarize only the relation properties needed for run auditing."""

    concept_count = int(full.shape[0])
    possible_edges = concept_count * max(0, concept_count - 1)
    denominator = float(possible_edges) if possible_edges else 1.0
    full_edge_mask = np.abs(full) > EPS
    full_edge_count = int(np.count_nonzero(full_edge_mask))
    fold_edge_counts = np.count_nonzero(
        np.abs(fold_relations) > EPS,
        axis=(1, 2),
    ).astype(np.float64)
    diagnostics = {
        "concept_count": float(concept_count),
        "full_edge_count": float(full_edge_count),
        "full_edge_density": float(full_edge_count / denominator),
        "full_positive_edge_count": float(np.count_nonzero(full > EPS)),
        "full_negative_edge_count": float(np.count_nonzero(full < -EPS)),
        "fold_edge_count_mean": float(fold_edge_counts.mean()),
        "fold_edge_density_mean": float(fold_edge_counts.mean() / denominator),
    }

    if full_edge_count:
        full_sign = np.sign(full)
        same_sign_count = np.sum(
            (np.sign(fold_relations) == full_sign[None, :, :])
            & full_edge_mask[None, :, :],
            axis=0,
        )
        stable_edges = int(np.count_nonzero(same_sign_count >= 4))
        diagnostics["full_sign_stable_ge4of5_share"] = float(
            stable_edges / full_edge_count
        )
    else:
        diagnostics["full_sign_stable_ge4of5_share"] = 0.0
    return diagnostics


def build_residual_relation_bundle(
    train_df: pd.DataFrame,
    stu_id_map: Mapping,
    cpt_id_map: Mapping,
    seed: int,
    topk: int,
) -> Dict:
    """Build train-only full and student-cross-fitted partial relations.

    Matrices follow ``relation[target, source]``.  Each fold relation is
    estimated, top-k filtered, and L1-normalized solely from that fold's
    complement students.  The full relation is estimated independently.
    """
    import torch

    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("train_df must be a pandas DataFrame")
    required = {"stu_id", "exer_id", "cpt_seq", "label"}
    missing = sorted(required - set(train_df.columns))
    if missing:
        raise ValueError(f"training frame is missing columns: {missing}")
    if train_df.empty:
        raise ValueError("cannot build a residual relation from an empty frame")
    if isinstance(topk, (bool, np.bool_)) or not isinstance(
        topk,
        (int, np.integer),
    ) or int(topk) <= 0:
        raise ValueError("residual relation topk must be a positive integer")
    _validate_internal_id_map("stu_id_map", stu_id_map)
    _validate_internal_id_map("cpt_id_map", cpt_id_map)

    train_students = set(train_df["stu_id"].unique())
    mapped_students = set(stu_id_map)
    missing_students = train_students - mapped_students
    extra_students = mapped_students - train_students
    if missing_students or extra_students:
        raise ValueError(
            "stu_id_map keys must exactly match cleaned training students"
        )
    train_concepts = {
        concept
        for value in train_df["cpt_seq"].values
        for concept in parse_concepts(value)
    }
    missing_concepts = train_concepts - set(cpt_id_map)
    if missing_concepts:
        raise ValueError(
            "cpt_id_map is missing concepts present in cleaned training data"
        )

    labels = pd.to_numeric(train_df["label"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(labels).all() or np.any((labels < 0.0) | (labels > 1.0)):
        raise ValueError("training labels must be finite values in [0, 1]")

    full_estimate, fold_estimates, assignment = build_cross_fitted_relations(
        train_df,
        dict(cpt_id_map),
        folds=RESIDUAL_RELATION_FOLDS,
        seed=int(seed),
        min_pair_students=RESIDUAL_RELATION_MIN_PAIR_STUDENTS,
    )
    full = topk_relation_matrix(full_estimate.partial, int(topk))
    folds = np.stack(
        [
            topk_relation_matrix(fold_estimates[fold].partial, int(topk))
            for fold in range(RESIDUAL_RELATION_FOLDS)
        ],
        axis=0,
    )

    expected_shape = (len(cpt_id_map), len(cpt_id_map))
    if full.shape != expected_shape or folds.shape != (
        RESIDUAL_RELATION_FOLDS,
        *expected_shape,
    ):
        raise RuntimeError("residual relation matrices are internally misaligned")
    if not np.isfinite(full).all() or not np.isfinite(folds).all():
        raise ValueError("residual relation matrices must contain only finite values")
    if not np.allclose(np.diag(full), 0.0) or not np.allclose(
        np.diagonal(folds, axis1=1, axis2=2),
        0.0,
    ):
        raise RuntimeError("residual relation matrices must not contain self edges")
    if np.any(np.abs(full).sum(axis=1) > 1.0 + 1e-8) or np.any(
        np.abs(folds).sum(axis=2) > 1.0 + 1e-8
    ):
        raise RuntimeError("residual relation row L1 norm exceeds one")

    student_fold = np.empty(len(stu_id_map), dtype=np.int64)
    for raw_student, internal_student in stu_id_map.items():
        if raw_student not in assignment:
            raise RuntimeError(
                f"student fold assignment is missing student {raw_student!r}"
            )
        student_fold[int(internal_student)] = int(assignment[raw_student])
    if np.any(student_fold < 0) or np.any(
        student_fold >= RESIDUAL_RELATION_FOLDS
    ):
        raise RuntimeError("student fold IDs are outside the fixed fold range")

    diagnostics = _relation_bundle_diagnostics(full, folds)
    if not all(np.isfinite(value) for value in diagnostics.values()):
        raise RuntimeError("residual relation diagnostics must be finite scalars")
    return {
        "schema": "residual_relation_bundle.v1",
        "mode": "partial_topk",
        "seed": int(seed),
        "topk": int(topk),
        "folds": RESIDUAL_RELATION_FOLDS,
        "min_pair_students": RESIDUAL_RELATION_MIN_PAIR_STUDENTS,
        "full_relation": torch.from_numpy(
            np.ascontiguousarray(full, dtype=np.float32)
        ),
        "fold_relations": torch.from_numpy(
            np.ascontiguousarray(folds, dtype=np.float32)
        ),
        "student_fold": torch.from_numpy(student_fold),
        "diagnostics": diagnostics,
    }
