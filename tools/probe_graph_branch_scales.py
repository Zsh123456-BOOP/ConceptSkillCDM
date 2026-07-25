#!/usr/bin/env python
"""Probe frozen v1 state/propagation branch scales on validation only.

The production v1 prediction can be decomposed exactly at the concept-ability
level into:

    theta = theta_base + state_scale * delta_state
                       + propagation_scale * delta_propagation

where ``delta_state`` is the projected graph-state change and
``delta_propagation`` is the contribution of the graph-propagated evidence
anchor channels.  This diagnostic evaluates a small scale grid without
retraining or opening a test split.  The ``(1, 1)`` cell must reproduce the
stored frozen checkpoint predictions exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import CognitiveDiagnosisDataset  # noqa: E402
from src.trainer import (  # noqa: E402
    _build_model,
    _require_graph_irt_checkpoint,
    _strip_module_prefix,
)


BUCKETS: Tuple[Tuple[str, int, int | None], ...] = (
    ("all", 0, None),
    ("n<3", 0, 3),
    ("n=0", 0, 1),
    ("n=1-2", 1, 3),
    ("n>=3", 3, None),
)


@dataclass(frozen=True)
class FrozenBranchOutputs:
    labels: np.ndarray
    base_logits: np.ndarray
    state_logit_delta: np.ndarray
    propagation_logit_delta: np.ndarray
    target_count: np.ndarray


def _parse_scales(value: str) -> Tuple[float, ...]:
    scales = tuple(
        float(token.strip())
        for token in str(value).split(",")
        if token.strip()
    )
    if not scales:
        raise ValueError("at least one branch scale is required")
    if any(not math.isfinite(scale) or scale < 0.0 for scale in scales):
        raise ValueError("branch scales must be finite and non-negative")
    return scales


def compose_scaled_logits(
    base_logits: torch.Tensor,
    state_logit_delta: torch.Tensor,
    propagation_logit_delta: torch.Tensor,
    *,
    state_scale: float,
    propagation_scale: float,
) -> torch.Tensor:
    """Return logits for two branch multipliers relative to frozen v1."""
    expected = tuple(base_logits.shape)
    for name, value in (
        ("state_logit_delta", state_logit_delta),
        ("propagation_logit_delta", propagation_logit_delta),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {tuple(value.shape)}"
            )
    return (
        base_logits
        + (float(state_scale) - 1.0) * state_logit_delta
        + (float(propagation_scale) - 1.0) * propagation_logit_delta
    )


def _validation_loader(
    loaded_args: Mapping[str, object],
    info_dict: Mapping[str, object],
    batch_size: int,
) -> DataLoader:
    valid_path = os.path.join(str(loaded_args["data_dir"]), "valid.csv")
    valid = pd.read_csv(valid_path)
    stu_map = info_dict["stu_id_map"]
    exer_map = info_dict["exer_id_map"]
    valid = valid[
        valid["stu_id"].isin(stu_map) & valid["exer_id"].isin(exer_map)
    ].reset_index(drop=True)
    dataset = CognitiveDiagnosisDataset(
        valid,
        stu_map,
        exer_map,
        info_dict["cpt_id_map"],
    )
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False)


def collect_frozen_branch_outputs(
    checkpoint_dir: str,
    *,
    batch_size: int,
    device: torch.device,
) -> Tuple[str, FrozenBranchOutputs]:
    path = os.path.join(checkpoint_dir, "best_model.pth")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require_graph_irt_checkpoint(checkpoint, path)
    loaded_args = checkpoint["args"]
    info_dict = checkpoint["info_dict"]
    if str(loaded_args.get("model_variant", "")) != "full":
        raise ValueError(f"branch probe requires a full checkpoint, got {path}")
    if str(loaded_args.get("gec_mode", "v1")) != "v1":
        raise ValueError(f"branch probe requires gec_mode='v1', got {path}")

    model = _build_model(loaded_args, info_dict, device)
    model.load_state_dict(
        _strip_module_prefix(checkpoint["model_state_dict"]),
        strict=True,
    )
    model.eval()
    if model.prediction_head != "irt2pl":
        raise ValueError("exact branch-logit decomposition requires irt2pl")
    if model.evidence_anchor_mode != "full":
        raise ValueError("exact propagation decomposition requires the full anchor")
    if model._anchor_channels <= 2:
        raise ValueError("checkpoint has no graph-propagated anchor channels")

    # Sparse graph matmul can change the final float32 rounding when the
    # flattened batch width changes.  Reuse the checkpoint's validation batch
    # size by default so near-tied predictions retain their original ordering.
    effective_batch_size = (
        int(batch_size)
        if int(batch_size) > 0
        else int(loaded_args["batch_size"])
    )
    loader = _validation_loader(
        loaded_args,
        info_dict,
        effective_batch_size,
    )
    anchor_weights = model.diagnosis_head.evidence_anchor_weights().to(device)
    labels_parts: List[torch.Tensor] = []
    base_parts: List[torch.Tensor] = []
    state_parts: List[torch.Tensor] = []
    propagation_parts: List[torch.Tensor] = []
    count_parts: List[torch.Tensor] = []

    with torch.no_grad():
        for student_ids, exercise_ids, labels in loader:
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            logits, details = model(
                student_ids,
                exercise_ids,
                return_details=True,
                return_logits=True,
            )
            q_vector = details["q_vector"]
            mask = (q_vector > 0).to(dtype=logits.dtype)
            denominator = mask.sum(dim=1).clamp(min=1.0)

            projected_state = model.diagnosis_head.theta_proj(
                details["knowledge_state"]
            ).squeeze(-1)
            projected_initial = model.diagnosis_head.theta_proj(
                details["initial_state"]
            ).squeeze(-1)
            state_theta_delta = projected_state - projected_initial

            anchor = details["evidence_anchor"]
            propagation_theta_delta = (
                anchor[..., 2:] * anchor_weights[:, 2:].unsqueeze(0)
            ).sum(dim=-1)
            discrimination = details["irt_a"]
            state_logit_delta = discrimination * (
                (state_theta_delta * mask).sum(dim=1) / denominator
            )
            propagation_logit_delta = discrimination * (
                (propagation_theta_delta * mask).sum(dim=1) / denominator
            )

            student_count = model.response_student_concept_count[student_ids]
            masked_count = student_count.masked_fill(mask <= 0, float("inf"))
            target_count = masked_count.min(dim=1).values
            if not bool(torch.isfinite(target_count).all()):
                raise RuntimeError("validation row has no mapped target concept")

            labels_parts.append(labels.reshape(-1).cpu().float())
            base_parts.append(logits.reshape(-1).cpu())
            state_parts.append(state_logit_delta.reshape(-1).cpu())
            propagation_parts.append(propagation_logit_delta.reshape(-1).cpu())
            count_parts.append(target_count.reshape(-1).cpu())

    outputs = FrozenBranchOutputs(
        labels=torch.cat(labels_parts).numpy().astype(np.float64),
        base_logits=torch.cat(base_parts).numpy().astype(np.float64),
        state_logit_delta=torch.cat(state_parts).numpy().astype(np.float64),
        propagation_logit_delta=torch.cat(propagation_parts)
        .numpy()
        .astype(np.float64),
        target_count=torch.cat(count_parts).numpy().astype(np.float64),
    )
    dataset_name = str(loaded_args.get("dataset_name", "unknown"))
    stored_auc = float(checkpoint.get("val_auc", float("nan")))
    reproduced_auc = _metrics(outputs.labels, outputs.base_logits)["auc"]
    # The stored metric is computed from float32 probabilities whereas this
    # offline grid keeps float64 logits.  Near-tied predictions can therefore
    # differ by a few billionths in rank AUC even though the logits are the
    # exact frozen outputs.
    if not math.isfinite(stored_auc) or abs(reproduced_auc - stored_auc) > 1e-7:
        raise RuntimeError(
            f"{dataset_name}: frozen (1,1) AUC mismatch "
            f"stored={stored_auc:.12f} reproduced={reproduced_auc:.12f}"
        )
    return dataset_name, outputs


def _metrics(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    if labels.size == 0:
        return {"auc": float("nan"), "bce_loss": float("nan"), "rmse": float("nan")}
    probabilities = np.where(
        logits >= 0.0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )
    auc = (
        float(roc_auc_score(labels, probabilities))
        if np.unique(labels).size >= 2
        else float("nan")
    )
    bce = float(np.mean(np.logaddexp(0.0, logits) - labels * logits))
    rmse = float(np.sqrt(np.mean(np.square(labels - probabilities))))
    return {"auc": auc, "bce_loss": bce, "rmse": rmse}


def _bucket_mask(count: np.ndarray, low: int, high: int | None) -> np.ndarray:
    mask = count >= float(low)
    if high is not None:
        mask &= count < float(high)
    return mask


def evaluate_grid(
    dataset: str,
    outputs: FrozenBranchOutputs,
    scales: Sequence[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    base = torch.from_numpy(outputs.base_logits)
    state = torch.from_numpy(outputs.state_logit_delta)
    propagation = torch.from_numpy(outputs.propagation_logit_delta)
    full_metrics = _metrics(outputs.labels, outputs.base_logits)

    for state_scale in scales:
        for propagation_scale in scales:
            logits = compose_scaled_logits(
                base,
                state,
                propagation,
                state_scale=state_scale,
                propagation_scale=propagation_scale,
            ).numpy()
            for bucket, low, high in BUCKETS:
                mask = (
                    np.ones(outputs.labels.shape, dtype=bool)
                    if bucket == "all"
                    else _bucket_mask(outputs.target_count, low, high)
                )
                metrics = _metrics(outputs.labels[mask], logits[mask])
                rows.append(
                    {
                        "dataset": dataset,
                        "state_scale": float(state_scale),
                        "propagation_scale": float(propagation_scale),
                        "bucket": bucket,
                        "rows": int(mask.sum()),
                        **metrics,
                        "auc_delta_from_full": (
                            metrics["auc"] - full_metrics["auc"]
                            if bucket == "all" and math.isfinite(metrics["auc"])
                            else float("nan")
                        ),
                    }
                )
    return rows


def _best_all_row(rows: Iterable[Mapping[str, object]]) -> Mapping[str, object]:
    eligible = [
        row
        for row in rows
        if row["bucket"] == "all" and math.isfinite(float(row["auc"]))
    ]
    return max(
        eligible,
        key=lambda row: (
            float(row["auc"]),
            -abs(float(row["state_scale"]) - 1.0)
            - abs(float(row["propagation_scale"]) - 1.0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        required=True,
        help="JSON mapping dataset name to a full/v1 checkpoint directory.",
    )
    parser.add_argument(
        "--scales",
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated non-negative state/propagation multipliers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Validation batch size; 0 reuses each checkpoint's stored value.",
    )
    parser.add_argument(
        "--output_csv",
        default="results/graph_branch_scale_probe.csv",
    )
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    pairs = json.loads(args.pairs)
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError("--pairs must decode to a non-empty object")
    scales = _parse_scales(args.scales)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )

    all_rows: List[Dict[str, object]] = []
    for requested_name, checkpoint_dir in pairs.items():
        dataset, outputs = collect_frozen_branch_outputs(
            str(checkpoint_dir),
            batch_size=args.batch_size,
            device=device,
        )
        if dataset != str(requested_name):
            raise ValueError(
                f"pair key {requested_name!r} does not match checkpoint dataset "
                f"{dataset!r}"
            )
        rows = evaluate_grid(dataset, outputs, scales)
        all_rows.extend(rows)
        best = _best_all_row(rows)
        print(
            f"{dataset}: best state={best['state_scale']:.2f} "
            f"prop={best['propagation_scale']:.2f} "
            f"AUC={best['auc']:.6f} "
            f"delta={best['auc_delta_from_full']:+.6f}"
        )

    output_path = os.path.realpath(args.output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_path, index=False)
    print(f"rows={len(all_rows)} -> {output_path}")


if __name__ == "__main__":
    main()
