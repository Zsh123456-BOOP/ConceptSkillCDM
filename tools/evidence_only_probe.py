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
  prop      : evidence logit propagated one hop over the train-only student
              co-exposure concept graph (closed-form graph calibration)
  prop+item : propagated evidence combined with the item difficulty logit

These floors quantify how much of the prediction signal is already carried by
the train-only sufficient statistics before any representation learning, and
therefore how much headroom the graph/state machinery must add.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-4


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


def _propagated_evidence_scores(
    train_sc: pd.DataFrame,
    valid: pd.DataFrame,
    valid_sc: pd.DataFrame,
    concept_rate: pd.Series,
    global_rate: float,
) -> np.ndarray:
    """One-hop propagation of per-student posterior logits over co-exposure."""
    concept_ids = np.sort(train_sc["cpt"].unique())
    concept_index = {int(c): i for i, c in enumerate(concept_ids)}
    C = len(concept_ids)
    prior = _build_coexposure_prior(train_sc, concept_ids)

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
    propagated_gap = anchored_gap @ prior.T

    valid_sc = valid_sc.copy()
    stu_rows = valid_sc["stu_id"].map(student_index).to_numpy()
    cpt_cols = valid_sc["cpt"].map(lambda c: concept_index.get(int(c), -1)).to_numpy()
    ok = (stu_rows >= 0) & (cpt_cols >= 0)
    scores = np.zeros(len(valid_sc), dtype=np.float64)
    scores[ok] = (
        _logit(prior_rate)[cpt_cols[ok]]
        + anchored_gap[stu_rows[ok], cpt_cols[ok]]
        + propagated_gap[stu_rows[ok], cpt_cols[ok]]
    )
    scores[~ok] = _logit(np.asarray(global_rate))
    valid_sc = valid_sc.assign(prop_logit=scores)
    per_row = valid_sc.groupby(level=0)["prop_logit"].mean()
    result = per_row.reindex(valid.index).to_numpy(dtype=np.float64)
    return np.nan_to_num(result, nan=_logit(np.asarray(global_rate)))


def probe_dataset(data_dir: str) -> dict:
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(data_dir, "valid.csv"))

    seen_students = set(train["stu_id"].unique())
    seen_items = set(train["exer_id"].unique())
    valid = valid[
        valid["stu_id"].isin(seen_students) & valid["exer_id"].isin(seen_items)
    ].reset_index(drop=True)

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
    propagated_logit = _propagated_evidence_scores(
        train_sc,
        valid,
        valid_sc_for_prop,
        concept_rate,
        global_rate,
    )

    results = {
        "rows": int(len(valid)),
        "global_rate": global_rate,
        "auc_item": float(roc_auc_score(labels, valid_item_logit)),
        "auc_evidence": float(roc_auc_score(labels, evidence_logit)),
        "auc_evid_plus_item": float(
            roc_auc_score(labels, evidence_logit + (valid_item_logit - _logit(np.asarray(global_rate))))
        ),
        "auc_propagated": float(roc_auc_score(labels, propagated_logit)),
        "auc_prop_plus_item": float(
            roc_auc_score(labels, propagated_logit + (valid_item_logit - _logit(np.asarray(global_rate))))
        ),
        "zero_evidence_row_share": float(np.mean(np.nan_to_num(mean_evid_count, nan=0.0) == 0.0)),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset dir names under data/")
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()

    names = [token.strip() for token in args.datasets.split(",") if token.strip()]
    print(
        f"{'dataset':<14} {'rows':>8} {'item':>7} {'evid':>7} {'evid+item':>9} "
        f"{'prop':>7} {'prop+item':>9} {'no-evid%':>8}"
    )
    for name in names:
        data_dir = os.path.join(args.data_root, name)
        if not os.path.isfile(os.path.join(data_dir, "train.csv")):
            print(f"{name:<14} MISSING")
            continue
        r = probe_dataset(data_dir)
        print(
            f"{name:<14} {r['rows']:>8} {r['auc_item']:>7.4f} {r['auc_evidence']:>7.4f} "
            f"{r['auc_evid_plus_item']:>9.4f} {r['auc_propagated']:>7.4f} "
            f"{r['auc_prop_plus_item']:>9.4f} {100.0 * r['zero_evidence_row_share']:>7.1f}%"
        )


if __name__ == "__main__":
    main()
