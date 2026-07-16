#!/usr/bin/env python
"""Expected Calibration Error on the validation split (M4 experiment).

The single 2PL readout anchored on empirical rates should produce well
calibrated probabilities; this tool quantifies it. Never touches test.csv.
"""

from __future__ import annotations

import argparse
import glob
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
from src.trainer import _build_model, _require_graph_irt_checkpoint, _strip_module_prefix  # noqa: E402


def ece_score(labels: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probs >= low) & (probs < high) if high < 1.0 else (probs >= low)
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / total) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def evaluate(checkpoint_dir: str, batch_size: int, device: torch.device) -> dict:
    model_path = os.path.join(checkpoint_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    _require_graph_irt_checkpoint(checkpoint, model_path)
    loaded_args, info_dict = checkpoint["args"], checkpoint["info_dict"]

    model = _build_model(loaded_args, info_dict, device)
    model.load_state_dict(_strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    model.eval()

    valid = pd.read_csv(os.path.join(loaded_args["data_dir"], "valid.csv"))
    stu_map, exer_map = info_dict["stu_id_map"], info_dict["exer_id_map"]
    valid = valid[
        valid["stu_id"].isin(stu_map) & valid["exer_id"].isin(exer_map)
    ].reset_index(drop=True)
    dataset = CognitiveDiagnosisDataset(valid, stu_map, exer_map, info_dict["cpt_id_map"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    labels, probs = [], []
    with torch.no_grad():
        for student_ids, exercise_ids, y in loader:
            p = model(student_ids.to(device), exercise_ids.to(device), return_logits=False)
            labels.extend(y.tolist())
            probs.extend(p.cpu().reshape(-1).tolist())
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    return {
        "dataset": loaded_args.get("dataset_name"),
        "variant": loaded_args.get("model_variant"),
        "rows": len(labels),
        "ece": ece_score(labels, probs),
        "brier": float(np.mean((probs - labels) ** 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_glob", required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"{'dataset':<12} {'variant':<24} {'rows':>8} {'ECE':>8} {'Brier':>8}")
    for run_dir in sorted(glob.glob(args.checkpoint_glob)):
        if not os.path.isfile(os.path.join(run_dir, "best_model.pth")):
            continue
        r = evaluate(run_dir, args.batch_size, device)
        print(f"{r['dataset']:<12} {r['variant']:<24} {r['rows']:>8} {r['ece']:>8.4f} {r['brier']:>8.4f}")


if __name__ == "__main__":
    main()
