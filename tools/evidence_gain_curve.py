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
    labels, probs, support = [], [], []
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
    return np.asarray(labels), np.asarray(probs), np.asarray(support)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        required=True,
        help="JSON mapping dataset -> {full: dir, woA: dir}",
    )
    parser.add_argument("--bucket_edges", default="0,1,3,6,12")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--output_csv", default="results/evidence_gain_curve.csv")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    pairs = json.loads(args.pairs)
    edges = sorted({float(t) for t in args.bucket_edges.split(",")}) + [float("inf")]

    records = []
    for dataset_name, dirs in pairs.items():
        full_model, loaded_args, info_dict = _load(dirs["full"], device)
        labels, probs_full, support = _rows(full_model, loaded_args, info_dict, args.batch_size, device)
        del full_model
        woa_model, loaded_args_b, info_dict_b = _load(dirs["woA"], device)
        labels_b, probs_woa, _ = _rows(woa_model, loaded_args_b, info_dict_b, args.batch_size, device)
        del woa_model
        assert len(labels) == len(labels_b), dataset_name
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (support >= low) & (support < high) if high != float("inf") else (support >= low)
            n = int(mask.sum())
            if n == 0 or len(set(labels[mask])) < 2:
                continue
            auc_full = compute_metrics(
                labels[mask], (probs_full[mask] > 0.5).astype(float), probs_full[mask]
            )["auc"]
            auc_woa = compute_metrics(
                labels[mask], (probs_woa[mask] > 0.5).astype(float), probs_woa[mask]
            )["auc"]
            records.append(
                {
                    "dataset": dataset_name,
                    "bucket_low": low,
                    "bucket_high": high,
                    "rows": n,
                    "auc_full": auc_full,
                    "auc_woA": auc_woa,
                    "gain": auc_full - auc_woa,
                }
            )
        print(f"{dataset_name}: done")

    frame = pd.DataFrame(records)
    pooled = []
    for (low, high), group in frame.groupby(["bucket_low", "bucket_high"]):
        weight = group["rows"] / group["rows"].sum()
        pooled.append(
            {
                "dataset": "POOLED",
                "bucket_low": low,
                "bucket_high": high,
                "rows": int(group["rows"].sum()),
                "auc_full": float((group["auc_full"] * weight).sum()),
                "auc_woA": float((group["auc_woA"] * weight).sum()),
                "gain": float((group["gain"] * weight).sum()),
            }
        )
    frame = pd.concat([frame, pd.DataFrame(pooled)], ignore_index=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(frame[frame["dataset"] == "POOLED"].to_string(index=False))
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
