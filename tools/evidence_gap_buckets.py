#!/usr/bin/env python
"""Per-bucket validation AUC by same-concept train evidence (S1 experiment).

Buckets every validation row by the minimum train-time response count over
the item's Q concepts for that student (the row's concept-evidence support),
then reports AUC inside each bucket.  Comparing model variants across buckets
quantifies where graph-propagated evidence actually helps: the mainline claim
predicts the largest gains in the low-evidence buckets.

Reads a validation-selected checkpoint directory (never touches test.csv).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiment_utils import compute_metrics  # noqa: E402
from src.trainer import _build_model, _require_graph_irt_checkpoint, _strip_module_prefix  # noqa: E402
from src.dataset import CognitiveDiagnosisDataset  # noqa: E402

import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument(
        "--bucket_edges",
        default="0,1,3,6",
        help="Comma-separated lower edges; e.g. 0,1,3,6 -> [0], [1,2], [3,5], [6,inf)",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    _require_graph_irt_checkpoint(checkpoint, model_path)
    loaded_args = checkpoint["args"]
    info_dict = checkpoint["info_dict"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    model = _build_model(loaded_args, info_dict, device)
    model.load_state_dict(_strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    model.eval()

    data_dir = loaded_args["data_dir"]
    valid = pd.read_csv(os.path.join(data_dir, "valid.csv"))
    stu_map, exer_map = info_dict["stu_id_map"], info_dict["exer_id_map"]
    valid = valid[
        valid["stu_id"].isin(stu_map) & valid["exer_id"].isin(exer_map)
    ].reset_index(drop=True)

    dataset = CognitiveDiagnosisDataset(valid, stu_map, exer_map, info_dict["cpt_id_map"])
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False)

    q_matrix = info_dict["q_matrix"]
    counts = info_dict["response_evidence_stats"]["student_concept_count"]

    all_labels, all_probs, all_support = [], [], []
    with torch.no_grad():
        for student_ids, exercise_ids, labels in loader:
            probs = model(
                student_ids.to(device),
                exercise_ids.to(device),
                return_logits=False,
            )
            q_rows = q_matrix[exercise_ids] > 0
            row_counts = counts[student_ids]
            support = torch.where(
                q_rows,
                row_counts,
                torch.full_like(row_counts, float("inf")),
            ).min(dim=1).values
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().reshape(-1).tolist())
            all_support.extend(support.cpu().tolist())

    labels = np.asarray(all_labels)
    probs = np.asarray(all_probs)
    support = np.asarray(all_support)

    edges = [float(tok) for tok in args.bucket_edges.split(",")]
    edges = sorted(set(edges)) + [float("inf")]
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (support >= low) & (support < high)
        n = int(mask.sum())
        if n == 0 or len(set(labels[mask])) < 2:
            auc = float("nan")
        else:
            auc = compute_metrics(
                labels[mask], (probs[mask] > 0.5).astype(float), probs[mask]
            )["auc"]
        label = f"[{low:g},{high:g})"
        rows.append({"bucket": label, "rows": n, "auc": auc})
        print(f"{label:>10}  rows={n:>7}  auc={auc:.4f}")

    overall = compute_metrics(labels, (probs > 0.5).astype(float), probs)["auc"]
    rows.append({"bucket": "all", "rows": len(labels), "auc": overall})
    print(f"{'all':>10}  rows={len(labels):>7}  auc={overall:.4f}")

    out_path = args.output_csv or os.path.join(args.checkpoint_dir, "evidence_gap_buckets.csv")
    pd.DataFrame(rows).assign(
        dataset=loaded_args.get("dataset_name"),
        model_variant=loaded_args.get("model_variant"),
        seed=loaded_args.get("seed"),
    ).to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
