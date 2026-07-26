#!/usr/bin/env python
"""Closed-form response-evidence floor for the Graph-IRT mainline.

Computes validation AUC for label-free/train-only scoring rules that use no
learned parameters at all:

  global    : one global train correct rate for every row
  item      : per-item train correct rate (Laplace-smoothed toward global)
  evidence  : Q-masked mean of the student-concept posterior logit, i.e. the
              same empirical-Bayes statistic the model consumes as evidence
  evid+item : evidence logit minus the item difficulty logit (a fixed a=1,
              b=item 2PL with no learning)
  P0 current: current weighted-sum propagation over the train-only student
              co-exposure graph
  P1 avail  : P0 renormalized over concepts for which this student has evidence
  P2 source : reliability-normalized evidence, shrunk by effective source count
  P3 target : P2 additionally attenuated by the target concept's evidence count

These floors quantify how much of the prediction signal is already carried by
the train-only sufficient statistics before any representation learning, and
therefore how much headroom the graph/state machinery must add.  Every
propagation variant uses the same fixed, train-only, label-free graph; the
validation labels are used only for the final metrics.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-4
PROPAGATION_VARIANTS = (
    "p0_current",
    "p1_available",
    "p2_source_confidence",
    "p3_target_scarcity",
    "p4_scarcity_correction",
)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _explode_concepts(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[["stu_id", "exer_id", "cpt_seq", "label"]].copy()
    rows["cpt"] = rows["cpt_seq"].astype(str).str.split(",")
    rows = rows.explode("cpt")
    rows = rows[rows["cpt"].str.strip() != ""]
    rows["cpt"] = rows["cpt"].str.strip().astype(np.int64)
    return rows


def _build_coexposure_prior(train_sc: pd.DataFrame, concept_ids: np.ndarray) -> np.ndarray:
    """Row-stochastic student co-exposure prior, mirroring src/dataset.py."""
    concept_index = {int(c): i for i, c in enumerate(concept_ids)}
    C = len(concept_ids)
    counts = np.zeros((C, C), dtype=np.float64)
    for _, concepts in train_sc.groupby("stu_id")["cpt"]:
        unique = sorted({concept_index[int(c)] for c in concepts if int(c) in concept_index})
        if len(unique) < 2:
            continue
        weight = 1.0 / np.sqrt(float(len(unique) - 1))
        idx = np.asarray(unique)
        counts[np.ix_(idx, idx)] += weight * weight
    np.fill_diagonal(counts, 0.0)
    row_sum = counts.sum(axis=1, keepdims=True)
    prior = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)
    return prior


def _build_item_prior(train_sc: pd.DataFrame, concept_ids: np.ndarray) -> np.ndarray:
    """Row-stochastic concept co-occurrence prior from train items only."""
    concept_index = {int(c): i for i, c in enumerate(concept_ids)}
    C = len(concept_ids)
    counts = np.zeros((C, C), dtype=np.float64)
    for _, concepts in train_sc.groupby("exer_id")["cpt"]:
        unique = sorted({concept_index[int(c)] for c in concepts if int(c) in concept_index})
        if len(unique) < 2:
            continue
        idx = np.asarray(unique)
        counts[np.ix_(idx, idx)] += 1.0
    np.fill_diagonal(counts, 0.0)
    row_sum = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)


def _build_graph_prior(
    train_sc: pd.DataFrame,
    concept_ids: np.ndarray,
    graph_source: str,
) -> np.ndarray:
    exposure = _build_coexposure_prior(train_sc, concept_ids)
    if graph_source == "exposure":
        return exposure
    item = _build_item_prior(train_sc, concept_ids)
    if graph_source == "item":
        return item
    if graph_source == "blend":
        prior = item + exposure
        row_sum = prior.sum(axis=1, keepdims=True)
        return np.divide(prior, row_sum, out=np.zeros_like(prior), where=row_sum > 0)
    raise ValueError(f"unknown graph_source: {graph_source!r}")


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def _propagated_evidence_scores(
    train_sc: pd.DataFrame,
    valid: pd.DataFrame,
    valid_sc: pd.DataFrame,
    concept_rate: pd.Series,
    global_rate: float,
    query_source_policy: str = "self",
    graph_source: str = "exposure",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compare four one-hop propagation rules on one fixed co-exposure graph.

    Relation rows have ``target, source`` semantics and no self-loop.  ``self``
    therefore reproduces the existing closed-form P0.  ``query`` additionally
    removes every concept attached to the current multi-concept item from the
    source set, preventing one historical item label copied across Q concepts
    from being counted as both direct and propagated evidence.
    """
    if query_source_policy not in {"self", "query"}:
        raise ValueError(f"unknown query_source_policy: {query_source_policy!r}")
    concept_ids = np.sort(train_sc["cpt"].unique())
    concept_index = {int(c): i for i, c in enumerate(concept_ids)}
    C = len(concept_ids)
    prior = _build_graph_prior(train_sc, concept_ids, graph_source)

    student_ids = np.sort(train_sc["stu_id"].unique())
    student_index = {int(s): i for i, s in enumerate(student_ids)}
    S = len(student_ids)

    correct = np.zeros((S, C), dtype=np.float64)
    count = np.zeros((S, C), dtype=np.float64)
    sc = train_sc.groupby(["stu_id", "cpt"])["label"].agg(["sum", "count"]).reset_index()
    rows = sc["stu_id"].map(student_index).to_numpy()
    cols = sc["cpt"].map(concept_index).to_numpy()
    correct[rows, cols] = sc["sum"].to_numpy(dtype=np.float64)
    count[rows, cols] = sc["count"].to_numpy(dtype=np.float64)

    prior_rate = np.full(C, global_rate, dtype=np.float64)
    for cid, rate in concept_rate.items():
        if int(cid) in concept_index:
            prior_rate[concept_index[int(cid)]] = float(rate)
    posterior = (correct + prior_rate[None, :]) / (count + 1.0)
    gap = _logit(posterior) - _logit(prior_rate)[None, :]
    reliability = count / (count + 1.0)
    anchored_gap = gap * reliability
    available = (count > 0.0).astype(np.float64)

    # All four variants share this numerator.  P0 is the current weighted sum.
    numerator = anchored_gap @ prior.T
    available_mass = available @ prior.T
    reliability_mass = reliability @ prior.T
    effective_source_count = count @ prior.T

    valid_sc = valid_sc.copy()
    stu_rows = valid_sc["stu_id"].map(student_index).to_numpy()
    cpt_cols = valid_sc["cpt"].map(lambda c: concept_index.get(int(c), -1)).to_numpy()
    ok = (stu_rows >= 0) & (cpt_cols >= 0)
    pair_numerator = np.zeros(len(valid_sc), dtype=np.float64)
    pair_available_mass = np.zeros(len(valid_sc), dtype=np.float64)
    pair_reliability_mass = np.zeros(len(valid_sc), dtype=np.float64)
    pair_effective_count = np.zeros(len(valid_sc), dtype=np.float64)
    pair_direct = np.zeros(len(valid_sc), dtype=np.float64)
    pair_target_count = np.zeros(len(valid_sc), dtype=np.float64)
    pair_prior_logit = np.full(
        len(valid_sc),
        float(_logit(np.asarray(global_rate))),
        dtype=np.float64,
    )
    pair_numerator[ok] = numerator[stu_rows[ok], cpt_cols[ok]]
    pair_available_mass[ok] = available_mass[stu_rows[ok], cpt_cols[ok]]
    pair_reliability_mass[ok] = reliability_mass[stu_rows[ok], cpt_cols[ok]]
    pair_effective_count[ok] = effective_source_count[stu_rows[ok], cpt_cols[ok]]
    pair_direct[ok] = anchored_gap[stu_rows[ok], cpt_cols[ok]]
    pair_target_count[ok] = count[stu_rows[ok], cpt_cols[ok]]
    pair_prior_logit[ok] = _logit(prior_rate)[cpt_cols[ok]]

    if query_source_policy == "query":
        # Operate only on the small Q set of each row, not on a row-by-C mask.
        for positions in valid_sc.groupby(level=0, sort=False).indices.values():
            positions = np.asarray(positions, dtype=np.int64)
            positions = positions[ok[positions]]
            if positions.size == 0:
                continue
            student = int(stu_rows[positions[0]])
            targets = cpt_cols[positions]
            sources = np.unique(targets)
            query_weights = prior[np.ix_(targets, sources)]
            pair_numerator[positions] -= query_weights @ anchored_gap[student, sources]
            pair_available_mass[positions] -= query_weights @ available[student, sources]
            pair_reliability_mass[positions] -= query_weights @ reliability[student, sources]
            pair_effective_count[positions] -= query_weights @ count[student, sources]

    # Floating-point subtraction in strict multi-concept mode can leave tiny
    # negative masses.  They represent exact zeros.
    pair_available_mass = np.maximum(pair_available_mass, 0.0)
    pair_reliability_mass = np.maximum(pair_reliability_mass, 0.0)
    pair_effective_count = np.maximum(pair_effective_count, 0.0)

    p0 = pair_numerator
    p1 = _safe_divide(pair_numerator, pair_available_mass)
    source_mean = _safe_divide(pair_numerator, pair_reliability_mass)
    source_confidence = pair_effective_count / (pair_effective_count + 1.0)
    p2 = source_mean * source_confidence
    p3 = p2 / (pair_target_count + 1.0)
    # Normalize only the part added beyond the current weighted sum, and only
    # while the target itself is scarce.  This returns smoothly to P0 as direct
    # target evidence accumulates instead of suppressing the whole graph term.
    p4 = p0 + (p1 - p0) / (pair_target_count + 1.0)
    pair_propagation = {
        "p0_current": p0,
        "p1_available": p1,
        "p2_source_confidence": p2,
        "p3_target_scarcity": p3,
        "p4_scarcity_correction": p4,
    }

    row_scores: dict[str, np.ndarray] = {}
    for name, propagated in pair_propagation.items():
        concept_score = pair_prior_logit + pair_direct + propagated
        per_row = pd.Series(concept_score, index=valid_sc.index).groupby(level=0).mean()
        score = per_row.reindex(valid.index).to_numpy(dtype=np.float64)
        row_scores[name] = np.nan_to_num(
            score,
            nan=float(_logit(np.asarray(global_rate))),
        )

    row_min_count = (
        pd.Series(pair_target_count, index=valid_sc.index)
        .groupby(level=0)
        .min()
        .reindex(valid.index)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    row_concept_count = (
        valid_sc.groupby(level=0)
        .size()
        .reindex(valid.index)
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )
    return row_scores, row_min_count, row_concept_count


def _auc(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        labels = labels[mask]
        scores = scores[mask]
    if len(labels) == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def probe_dataset(
    data_dir: str,
    leak_mode: str = "none",
    query_source_policy: str = "self",
    graph_source: str = "exposure",
) -> dict:
    """Score valid rows from sufficient statistics.

    ``leak_mode`` quantifies two classic leakage bugs against the clean
    train-only + leave-one-out contract:
      none   -> train-only statistics (the honest floor)
      corpus -> statistics built from train+valid, as when a prior matrix is
                computed before the split
      self   -> train-only statistics plus the current row's own label, the
                exact shortcut exact-LOO removes
    """
    if leak_mode not in {"none", "corpus", "self"}:
        raise ValueError(f"unknown leak_mode: {leak_mode!r}")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(data_dir, "valid.csv"))

    seen_students = set(train["stu_id"].unique())
    seen_items = set(train["exer_id"].unique())
    valid = valid[
        valid["stu_id"].isin(seen_students) & valid["exer_id"].isin(seen_items)
    ].reset_index(drop=True)

    if leak_mode == "corpus":
        train = pd.concat([train, valid], ignore_index=True)

    global_rate = float(train["label"].mean())
    labels = valid["label"].to_numpy(dtype=np.float64)

    # Per-item train rate with one pseudo-observation at the global rate.
    item_stats = train.groupby("exer_id")["label"].agg(["sum", "count"])
    item_rate = (item_stats["sum"] + global_rate) / (item_stats["count"] + 1.0)
    item_logit_by_exer = _logit(item_rate.to_numpy())
    item_logit = pd.Series(item_logit_by_exer, index=item_stats.index)
    valid_item_logit = valid["exer_id"].map(item_logit).to_numpy(dtype=np.float64)

    # Student-concept sufficient statistics from train only.
    train_sc = _explode_concepts(train)
    concept_stats = train_sc.groupby("cpt")["label"].agg(["sum", "count"])
    concept_rate = (concept_stats["sum"] + global_rate) / (concept_stats["count"] + 1.0)

    sc_stats = train_sc.groupby(["stu_id", "cpt"])["label"].agg(["sum", "count"])

    valid_sc = _explode_concepts(valid)
    keys = pd.MultiIndex.from_frame(valid_sc[["stu_id", "cpt"]])
    matched = sc_stats.reindex(keys)
    correct = matched["sum"].to_numpy(dtype=np.float64)
    count = matched["count"].to_numpy(dtype=np.float64)
    correct = np.nan_to_num(correct, nan=0.0)
    count = np.nan_to_num(count, nan=0.0)
    if leak_mode == "self":
        row_labels = valid_sc["label"].to_numpy(dtype=np.float64)
        correct = correct + row_labels
        count = count + 1.0

    cpt_prior = valid_sc["cpt"].map(concept_rate).to_numpy(dtype=np.float64)
    cpt_prior = np.nan_to_num(cpt_prior, nan=global_rate)

    posterior = (correct + cpt_prior) / (count + 1.0)
    valid_sc = valid_sc.assign(post_logit=_logit(posterior), evid_count=count)
    per_row = valid_sc.groupby(level=0).agg(
        post_logit=("post_logit", "mean"),
        evid_count=("evid_count", "mean"),
    )
    evidence_logit = per_row["post_logit"].reindex(valid.index).to_numpy(dtype=np.float64)
    evidence_logit = np.nan_to_num(evidence_logit, nan=_logit(np.asarray(global_rate)))
    mean_evid_count = per_row["evid_count"].reindex(valid.index).to_numpy(dtype=np.float64)

    valid_sc_for_prop = _explode_concepts(valid)
    propagated_logits, row_min_count, row_concept_count = _propagated_evidence_scores(
        train_sc,
        valid,
        valid_sc_for_prop,
        concept_rate,
        global_rate,
        query_source_policy=query_source_policy,
        graph_source=graph_source,
    )

    results = {
        "rows": int(len(valid)),
        "global_rate": global_rate,
        "query_source_policy": query_source_policy,
        "graph_source": graph_source,
        "auc_item": _auc(labels, valid_item_logit),
        "auc_evidence": _auc(labels, evidence_logit),
        "auc_evid_plus_item": float(
            roc_auc_score(labels, evidence_logit + (valid_item_logit - _logit(np.asarray(global_rate))))
        ),
        "zero_evidence_row_share": float(np.mean(np.nan_to_num(mean_evid_count, nan=0.0) == 0.0)),
        "n_lt_3_row_share": float(np.mean(row_min_count < 3.0)),
        "multi_concept_row_share": float(np.mean(row_concept_count > 1)),
    }
    item_adjustment = valid_item_logit - _logit(np.asarray(global_rate))
    masks = {
        "all": np.ones(len(valid), dtype=bool),
        "n_eq_0": row_min_count == 0.0,
        "n_lt_3": row_min_count < 3.0,
        "single": row_concept_count == 1,
        "multi": row_concept_count > 1,
    }
    for name in PROPAGATION_VARIANTS:
        scores = propagated_logits[name]
        results[f"auc_{name}"] = _auc(labels, scores)
        for bucket, mask in masks.items():
            results[f"auc_{name}_plus_item_{bucket}"] = _auc(
                labels,
                scores + item_adjustment,
                mask,
            )
    # Preserve the old result names for downstream notebooks.
    results["auc_propagated"] = results["auc_p0_current"]
    results["auc_prop_plus_item"] = results["auc_p0_current_plus_item_all"]
    for left, right in zip(PROPAGATION_VARIANTS[:-1], PROPAGATION_VARIANTS[1:]):
        results[f"delta_{right}_minus_{left}"] = (
            results[f"auc_{right}_plus_item_all"]
            - results[f"auc_{left}_plus_item_all"]
        )
    results["delta_p3_minus_p0"] = (
        results["auc_p3_target_scarcity_plus_item_all"]
        - results["auc_p0_current_plus_item_all"]
    )
    results["delta_p4_minus_p0"] = (
        results["auc_p4_scarcity_correction_plus_item_all"]
        - results["auc_p0_current_plus_item_all"]
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset dir names under data/")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument(
        "--leak_mode",
        default="none",
        choices=("none", "corpus", "self"),
        help="Quantify classic leakage bugs against the train-only + LOO contract.",
    )
    parser.add_argument(
        "--query_source_policy",
        default="self",
        choices=("self", "query"),
        help=(
            "Exclude only the target concept (self), or every concept attached "
            "to the current query item, from propagation sources."
        ),
    )
    parser.add_argument(
        "--graph_source",
        default="exposure",
        choices=("exposure", "item", "blend"),
        help="Fixed train-only graph used by every propagation variant.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path for one flat validation-summary row per dataset.",
    )
    args = parser.parse_args()

    names = [token.strip() for token in args.datasets.split(",") if token.strip()]
    print(
        f"leak_mode={args.leak_mode} "
        f"query_source_policy={args.query_source_policy} "
        f"graph_source={args.graph_source}"
    )
    print(
        f"{'dataset':<14} {'rows':>8} {'P0+item':>9} {'P1+item':>9} "
        f"{'P4+item':>9} {'P1-P0':>8} {'P4-P0':>8} "
        f"{'n<3%':>7} {'multi%':>7}"
    )
    records = []
    for name in names:
        data_dir = os.path.join(args.data_root, name)
        if not os.path.isfile(os.path.join(data_dir, "train.csv")):
            print(f"{name:<14} MISSING")
            continue
        r = probe_dataset(
            data_dir,
            leak_mode=args.leak_mode,
            query_source_policy=args.query_source_policy,
            graph_source=args.graph_source,
        )
        records.append({"dataset": name, **r})
        print(
            f"{name:<14} {r['rows']:>8} "
            f"{r['auc_p0_current_plus_item_all']:>9.5f} "
            f"{r['auc_p1_available_plus_item_all']:>9.5f} "
            f"{r['auc_p4_scarcity_correction_plus_item_all']:>9.5f} "
            f"{r['delta_p1_available_minus_p0_current']:>+8.5f} "
            f"{r['delta_p4_minus_p0']:>+8.5f} "
            f"{100.0 * r['n_lt_3_row_share']:>6.1f}% "
            f"{100.0 * r['multi_concept_row_share']:>6.1f}%"
        )
    if args.output_csv:
        output_dir = os.path.dirname(os.path.abspath(args.output_csv))
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame.from_records(records).to_csv(args.output_csv, index=False)
        print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
