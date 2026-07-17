#!/usr/bin/env python
"""Effective per-channel anchor contribution to theta (S4 replacement).

The raw anchor weights understate channel importance because a learnable
count-gate multiplies each channel. This tool reports the mean absolute
theta shift each channel actually delivers on the validation split:
E[ |weight_c * gate * evidence| ] per channel, which folds weight, gate,
and the evidence distribution into one comparable number. Never opens test.
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
from src.trainer import _build_model, _require_graph_irt_checkpoint, _strip_module_prefix  # noqa: E402


CHANNEL_NAMES = ["direct", "residual", "prop_h1", "prop_h2", "prop_h3", "prop_h4"]


def evaluate(checkpoint_dir: str, batch_size: int, device: torch.device) -> dict:
    path = os.path.join(checkpoint_dir, "best_model.pth")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require_graph_irt_checkpoint(checkpoint, path)
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

    q_matrix = info_dict["q_matrix"].to(device)
    weights = model.diagnosis_head.evidence_anchor_weights().to(device)  # (C, K)
    totals = None
    row_total = 0
    with torch.no_grad():
        for student_ids, exercise_ids, _ in loader:
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            q_vec = q_matrix[exercise_ids]
            resp, loo_count = model._build_response_evidence(student_ids, None, q_vec, None)
            anchor = model._compose_evidence_anchor(model.relation_learning(), resp, loo_count)
            if anchor is None:
                return {"dataset": loaded_args.get("dataset_name"), "channels": {}, "rows": 0}
            # per-concept contribution then Q-masked to the row's concepts
            contrib = (anchor * weights.unsqueeze(0)).abs()  # (B, C, K)
            mask = (q_vec > 0).float().unsqueeze(-1)
            masked = (contrib * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, K)
            batch_sum = masked.sum(dim=0)
            totals = batch_sum if totals is None else totals + batch_sum
            row_total += masked.size(0)
    means = (totals / max(1, row_total)).cpu().tolist()
    channels = {CHANNEL_NAMES[i]: float(v) for i, v in enumerate(means)}
    return {"dataset": loaded_args.get("dataset_name"), "channels": channels, "rows": row_total}


def _valid_auc(model, loaded_args, info_dict, batch_size, device) -> float:
    from src.experiment_utils import compute_metrics

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
    import numpy as np

    labels = np.asarray(labels)
    probs = np.asarray(probs)
    return compute_metrics(labels, (probs > 0.5).astype(float), probs)["auc"]


def causal_channel_drops(checkpoint_dir: str, batch_size: int, device: torch.device) -> dict:
    """Validation AUC drop when one anchor channel's weights are zeroed.

    Zeroing the per-concept weight column removes exactly that channel's theta
    contribution at inference; the drop is a causal measure of what the
    channel delivers on top of the remaining ones.
    """
    path = os.path.join(checkpoint_dir, "best_model.pth")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require_graph_irt_checkpoint(checkpoint, path)
    loaded_args, info_dict = checkpoint["args"], checkpoint["info_dict"]
    model = _build_model(loaded_args, info_dict, device)
    model.load_state_dict(_strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    model.eval()

    base_auc = _valid_auc(model, loaded_args, info_dict, batch_size, device)
    raw = model.diagnosis_head.evidence_anchor_raw
    original = raw.detach().clone()
    drops = {"base_auc": base_auc}
    for k in range(raw.shape[-1]):
        with torch.no_grad():
            raw.copy_(original)
            raw[:, k] = -20.0  # softplus(-20) ~= 0: channel k contributes nothing
        auc_k = _valid_auc(model, loaded_args, info_dict, batch_size, device)
        drops[f"drop_{CHANNEL_NAMES[k]}"] = base_auc - auc_k
    with torch.no_grad():
        raw.copy_(original)
    return {"dataset": loaded_args.get("dataset_name"), "drops": drops}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="JSON mapping dataset -> checkpoint dir")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--output_csv", default="results/anchor_contribution.csv")
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Also report the validation AUC drop when each channel is zeroed at inference.",
    )
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    pairs = json.loads(args.pairs)
    rows = []
    for dataset_name, ckpt in pairs.items():
        r = evaluate(ckpt, args.batch_size, device)
        record = {"dataset": dataset_name, "rows": r["rows"]}
        record.update(r["channels"])
        if args.causal:
            record.update(causal_channel_drops(ckpt, args.batch_size, device)["drops"])
        rows.append(record)
        print(f"{dataset_name}: {record}")
    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
