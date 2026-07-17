#!/usr/bin/env python
"""Pooled evidence-gain curve: AUC(full) - AUC(w/o evidence) per support bucket.

For every dataset the validation rows are bucketed by the minimum same-concept
train evidence count of the row's Q concepts; the per-bucket AUC gap between
the full model and its no-evidence ablation is then averaged across datasets
with row-count weights. One curve, all datasets, no cherry-picking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.dataset import CognitiveDiagnosisDataset  # noqa: E402
from src.experiment_utils import compute_metrics  # noqa: E402
from src.trainer import _build_model, _require_graph_irt_checkpoint, _strip_module_prefix  # noqa: E402


def _load(checkpoint_dir: str, device: torch.device):
    path = os.path.join(checkpoint_dir, "best_model.pth")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require_graph_irt_checkpoint(checkpoint, path)
    model = _build_model(checkpoint["args"], checkpoint["info_dict"], device)
    model.load_state_dict(_strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    return model, checkpoint["args"], checkpoint["info_dict"]


def _rows(model, loaded_args, info_dict, batch_size, device):
    valid = pd.read_csv(os.path.join(loaded_args["data_dir"], "valid.csv"))
    stu_map, exer_map = info_dict["stu_id_map"], info_dict["exer_id_map"]
    valid = valid[
        valid["stu_id"].isin(stu_map) & valid["exer_id"].isin(exer_map)
    ].reset_index(drop=True)
    dataset = CognitiveDiagnosisDataset(valid, stu_map, exer_map, info_dict["cpt_id_map"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    q_matrix = info_dict["q_matrix"]
    counts = info_dict["response_evidence_stats"]["student_concept_count"]
    labels, probs, support, students = [], [], [], []
    with torch.no_grad():
        for student_ids, exercise_ids, y in loader:
            p = model(student_ids.to(device), exercise_ids.to(device), return_logits=False)
            q_rows = q_matrix[exercise_ids] > 0
            row_counts = counts[student_ids]
            s = torch.where(
                q_rows, row_counts, torch.full_like(row_counts, float("inf"))
            ).min(dim=1).values
            labels.extend(y.tolist())
            probs.extend(p.cpu().reshape(-1).tolist())
            support.extend(s.cpu().tolist())
            students.extend(student_ids.tolist())
    return (
        np.asarray(labels),
        np.asarray(probs),
        np.asarray(support),
        np.asarray(students),
    )


def _bucket_auc(labels, probs, mask):
    if int(mask.sum()) == 0 or len(set(labels[mask])) < 2:
        return None
    return compute_metrics(
        labels[mask], (probs[mask] > 0.5).astype(float), probs[mask]
    )["auc"]


def _bootstrap_gap_ci(labels, probs_full, probs_woa, students, mask, reps, rng):
    """Student-level paired bootstrap CI for AUC(full) - AUC(woA) in a bucket."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return None, None
    student_ids = students[idx]
    unique_students = np.unique(student_ids)
    by_student = {s: idx[student_ids == s] for s in unique_students}
    gaps = []
    for _ in range(reps):
        sampled = rng.choice(unique_students, size=len(unique_students), replace=True)
        rows = np.concatenate([by_student[s] for s in sampled])
        if len(set(labels[rows])) < 2:
            continue
        auc_f = compute_metrics(
            labels[rows], (probs_full[rows] > 0.5).astype(float), probs_full[rows]
        )["auc"]
        auc_w = compute_metrics(
            labels[rows], (probs_woa[rows] > 0.5).astype(float), probs_woa[rows]
        )["auc"]
        gaps.append(auc_f - auc_w)
    if len(gaps) < max(10, reps // 4):
        return None, None
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        required=True,
        help="JSON mapping dataset -> {full: dir, woA: dir}",
    )
    parser.add_argument("--bucket_edges", default="0,1,3,6,12")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Student-level paired bootstrap replicates for per-bucket gain CIs (0 disables).",
    )
    parser.add_argument("--bootstrap_seed", type=int, default=7)
    parser.add_argument("--output_csv", default="results/evidence_gain_curve.csv")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    pairs = json.loads(args.pairs)
    edges = sorted({float(t) for t in args.bucket_edges.split(",")}) + [float("inf")]
    rng = np.random.RandomState(args.bootstrap_seed)

    records = []
    for dataset_name, dirs in pairs.items():
        full_model, loaded_args, info_dict = _load(dirs["full"], device)
        labels, probs_full, support, students = _rows(
            full_model, loaded_args, info_dict, args.batch_size, device
        )
        del full_model
        woa_model, loaded_args_b, info_dict_b = _load(dirs["woA"], device)
        labels_b, probs_woa, _, _ = _rows(
            woa_model, loaded_args_b, info_dict_b, args.batch_size, device
        )
        del woa_model
        assert len(labels) == len(labels_b), dataset_name
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (support >= low) & (support < high) if high != float("inf") else (support >= low)
            auc_full = _bucket_auc(labels, probs_full, mask)
            auc_woa = _bucket_auc(labels, probs_woa, mask)
            if auc_full is None or auc_woa is None:
                continue
            ci_low, ci_high = (None, None)
            if args.bootstrap > 0:
                ci_low, ci_high = _bootstrap_gap_ci(
                    labels, probs_full, probs_woa, students, mask, args.bootstrap, rng
                )
            records.append(
                {
                    "dataset": dataset_name,
                    "bucket_low": low,
                    "bucket_high": high,
                    "rows": int(mask.sum()),
                    "auc_full": auc_full,
                    "auc_woA": auc_woa,
                    "gain": auc_full - auc_woa,
                    "gain_ci_low": ci_low,
                    "gain_ci_high": ci_high,
                }
            )
        print(f"{dataset_name}: done")

    frame = pd.DataFrame(records)
    bucket_count = frame.groupby("dataset")["bucket_low"].count()
    max_buckets = int(bucket_count.max())
    complete_datasets = set(bucket_count[bucket_count == max_buckets].index)

    def _pool(group_frame: pd.DataFrame, tag: str) -> list:
        pooled_rows = []
        for (low, high), group in group_frame.groupby(["bucket_low", "bucket_high"]):
            weight = group["rows"] / group["rows"].sum()
            pooled_rows.append(
                {
                    "dataset": tag,
                    "bucket_low": low,
                    "bucket_high": high,
                    "rows": int(group["rows"].sum()),
                    "auc_full": float((group["auc_full"] * weight).sum()),
                    "auc_woA": float((group["auc_woA"] * weight).sum()),
                    "gain": float((group["gain"] * weight).sum()),
                    "gain_ci_low": None,
                    "gain_ci_high": None,
                }
            )
        return pooled_rows

    pooled = _pool(frame, "POOLED")
    # Composition-controlled pool: only datasets contributing to every bucket,
    # so the curve shape cannot come from the dataset mix changing across
    # buckets.
    complete = _pool(
        frame[frame["dataset"].isin(complete_datasets)], "POOLED_COMPLETE"
    )
    frame = pd.concat([frame, pd.DataFrame(pooled + complete)], ignore_index=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"complete-coverage datasets: {sorted(complete_datasets)}")
    print(frame[frame["dataset"].str.startswith("POOLED")].to_string(index=False))
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
