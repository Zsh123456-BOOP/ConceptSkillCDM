"""Closed-form probe: how much target-concept information do neighbor
concepts' response statistics carry, as a function of graph distance?

For each dataset, build the label-free concept graph from train.csv (union of
item co-occurrence and student co-exposure, top-K neighbors per concept by
count). For every (evaluation row, target concept), predict the response using ONLY
other concepts' self-excluded deviations, aggregated over one of three tiers:
1-hop prior neighbors, 2-hop neighbors (neighbors of neighbors, excluding
1-hop and self), and K random non-neighbor concepts. The per-tier score is
``concept mean + weighted neighbor deviation``; the concept-mean-only baseline
is reported alongside. Knowledge-structure theory predicts the AUC ordering
1-hop > 2-hop > random.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    frame = pd.DataFrame({"score": scores, "rank": ranks})
    ranks = frame.groupby("score")["rank"].transform("mean").to_numpy()
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _load(dataset: str, split: str):
    data_dir = os.path.join(ROOT, "data", dataset)
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    evaluation = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    for frame in (train, evaluation):
        frame["cpt_list"] = (
            frame["cpt_seq"].astype(str).str.strip('"').str.split(",")
        )
    return train, evaluation


def _build_graph(train: pd.DataFrame, num_concepts: int, topk: int) -> np.ndarray:
    counts = np.zeros((num_concepts, num_concepts), dtype=np.float64)
    item_concepts = (
        train.drop_duplicates("exer_id")["cpt_list"]
        .apply(lambda cs: sorted({int(c) for c in cs}))
    )
    for concepts in item_concepts:
        for i, a in enumerate(concepts):
            for b in concepts[i + 1 :]:
                counts[a, b] += 1.0
                counts[b, a] += 1.0
    student_concepts = (
        train.explode("cpt_list")
        .assign(cpt=lambda d: d["cpt_list"].astype(int))
        .groupby("stu_id")["cpt"]
        .agg(lambda s: sorted(set(s)))
    )
    for concepts in student_concepts:
        arr = np.asarray(concepts, dtype=np.int64)
        if len(arr) < 2:
            continue
        counts[np.ix_(arr, arr)] += 1.0
    np.fill_diagonal(counts, 0.0)
    adjacency = np.zeros_like(counts)
    for c in range(num_concepts):
        row = counts[c]
        if row.max() <= 0:
            continue
        keep = np.argsort(row)[::-1][:topk]
        keep = keep[row[keep] > 0]
        adjacency[c, keep] = row[keep]
    return adjacency


def _tier_weights(adjacency: np.ndarray, topk: int, seed: int):
    num_concepts = adjacency.shape[0]
    one_hop = adjacency.copy()
    second = (adjacency > 0).astype(np.float64) @ adjacency
    second[adjacency > 0] = 0.0
    np.fill_diagonal(second, 0.0)
    two_hop = np.zeros_like(second)
    for c in range(num_concepts):
        row = second[c]
        if row.max() <= 0:
            continue
        keep = np.argsort(row)[::-1][:topk]
        keep = keep[row[keep] > 0]
        two_hop[c, keep] = row[keep]
    rng = np.random.RandomState(seed)
    random_tier = np.zeros_like(one_hop)
    for c in range(num_concepts):
        forbidden = set(np.nonzero(one_hop[c])[0]) | {c}
        candidates = [k for k in range(num_concepts) if k not in forbidden]
        if not candidates:
            continue
        picked = rng.choice(candidates, size=min(topk, len(candidates)), replace=False)
        random_tier[c, picked] = 1.0
    return {"1hop": one_hop, "2hop": two_hop, "random": random_tier}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="assist_17,junyi,nips34,ednet_kt1,moocradar,xes3g5m",
    )
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--output_csv",
        default="results/neighbor_information_valid.csv",
    )
    args = parser.parse_args()

    records = []
    for dataset in args.datasets.split(","):
        train, evaluation = _load(dataset, args.split)
        train_x = train.explode("cpt_list")
        train_x["cpt"] = train_x["cpt_list"].astype(int)
        num_concepts = int(train_x["cpt"].max()) + 1
        students = sorted(train["stu_id"].unique())
        stu_index = {s: i for i, s in enumerate(students)}

        stats = (
            train_x.groupby(["stu_id", "cpt"])["label"]
            .agg(n="count", S="sum")
            .reset_index()
        )
        concept_mean = np.full(num_concepts, 0.5)
        grouped = train_x.groupby("cpt")["label"].mean()
        concept_mean[grouped.index.to_numpy()] = grouped.to_numpy()

        deviation = np.zeros((len(students), num_concepts))
        has_data = np.zeros((len(students), num_concepts))
        rows_idx = stats["stu_id"].map(stu_index).to_numpy()
        cols_idx = stats["cpt"].to_numpy()
        m_hat = (stats["S"].to_numpy() + 1.0) / (stats["n"].to_numpy() + 2.0)
        deviation[rows_idx, cols_idx] = m_hat - concept_mean[cols_idx]
        has_data[rows_idx, cols_idx] = 1.0

        adjacency = _build_graph(train, num_concepts, args.topk)
        tiers = _tier_weights(adjacency, args.topk, args.seed)

        evaluation_x = evaluation.explode("cpt_list").copy()
        evaluation_x["cpt"] = evaluation_x["cpt_list"].astype(int)
        evaluation_x = evaluation_x[evaluation_x["stu_id"].isin(stu_index)]
        evaluation_x["srow"] = evaluation_x["stu_id"].map(stu_index)
        srow = evaluation_x["srow"].to_numpy()
        cpt = evaluation_x["cpt"].to_numpy()
        row_id = evaluation_x.index.to_numpy()
        labels_row = evaluation.loc[evaluation_x.index.unique(), "label"]

        n_same = np.zeros_like(cpt, dtype=np.float64)
        pairs = pd.MultiIndex.from_arrays(
            [evaluation_x["stu_id"], evaluation_x["cpt"]]
        )
        stat_map = stats.set_index(["stu_id", "cpt"])["n"]
        n_same = stat_map.reindex(pairs).fillna(0).to_numpy()

        def row_scores(agg_matrix: np.ndarray) -> pd.Series:
            per_pair = concept_mean[cpt] + agg_matrix[srow, cpt]
            return pd.Series(per_pair, index=row_id).groupby(level=0).mean()

        zero_mask = (
            pd.Series(n_same, index=row_id).groupby(level=0).max() == 0
        )
        results = {}
        base_scores = pd.Series(concept_mean[cpt], index=row_id).groupby(level=0).mean()
        results["concept_mean"] = base_scores
        for tier_name, weights in tiers.items():
            wsum = has_data @ weights.T
            agg = np.divide(
                deviation @ weights.T,
                wsum,
                out=np.zeros_like(wsum),
                where=wsum > 0,
            )
            results[tier_name] = row_scores(agg)

        labels_arr = labels_row.to_numpy()
        for tier_name, scores in results.items():
            aligned = scores.reindex(labels_row.index).to_numpy()
            records.append(
                {
                    "dataset": dataset,
                    "split": args.split,
                    "tier": tier_name,
                    "auc_all": _auc(labels_arr, aligned),
                    "auc_zero_same_concept": _auc(
                        labels_arr[zero_mask.to_numpy()],
                        aligned[zero_mask.to_numpy()],
                    ),
                    "rows": len(labels_arr),
                    "rows_zero_same_concept": int(zero_mask.sum()),
                }
            )
        print(f"{dataset}/{args.split}: done ({len(labels_arr)} rows)")

    out = pd.DataFrame(records)
    out_path = os.path.join(ROOT, args.output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
