"""Per-sample leakage-shift fan data for the analysis figure.

For every (evaluation row, target concept) pair, the self-included statistic minus
the self-excluded statistic equals (y - m_hat) / (n + 3) in closed form, where
m_hat = (S + 1) / (n + 2) is the smoothed train-only rate and n, S are the
train-only same-concept counts. This script materialises a subsample of those
per-pair shifts so the figure can overlay the measured points on the
theoretical envelopes +-(n + 1) / ((n + 2)(n + 3)).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pairs(dataset: str, split: str) -> pd.DataFrame:
    data_dir = os.path.join(ROOT, "data", dataset)
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    evaluation = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    for frame in (train, evaluation):
        frame["cpt_list"] = (
            frame["cpt_seq"].astype(str).str.strip('"').str.split(",")
        )
    train_x = train.explode("cpt_list")
    train_x["cpt"] = train_x["cpt_list"].astype(int)
    stats = (
        train_x.groupby(["stu_id", "cpt"])["label"]
        .agg(n="count", S="sum")
        .reset_index()
    )
    evaluation_x = evaluation.explode("cpt_list")
    evaluation_x["cpt"] = evaluation_x["cpt_list"].astype(int)
    merged = evaluation_x.merge(stats, on=["stu_id", "cpt"], how="left")
    merged[["n", "S"]] = merged[["n", "S"]].fillna(0)
    m_hat = (merged["S"] + 1.0) / (merged["n"] + 2.0)
    return pd.DataFrame(
        {
            "dataset": dataset,
            "split": split,
            "n": merged["n"].astype(int),
            "label": merged["label"].astype(int),
            "delta": (merged["label"] - m_hat) / (merged["n"] + 3.0),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="assist_17,junyi,nips34,ednet_kt1,moocradar,xes3g5m",
    )
    parser.add_argument("--per_dataset", type=int, default=1500)
    parser.add_argument("--max_n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--output_csv",
        default="results/leakage_fan_valid.csv",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    frames = []
    for dataset in args.datasets.split(","):
        pairs = _pairs(dataset, args.split)
        pairs = pairs[pairs["n"] <= args.max_n]
        if len(pairs) > args.per_dataset:
            pairs = pairs.iloc[
                rng.choice(len(pairs), size=args.per_dataset, replace=False)
            ]
        frames.append(pairs)
        print(f"{dataset}/{args.split}: {len(pairs)} pairs sampled")
    out = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(ROOT, args.output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
