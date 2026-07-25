#!/usr/bin/env python
"""Evaluate the graph 2x2 checkpoints on validation evidence-count subsets.

The tool never opens ``test.csv``.  Each validation row is assigned the
minimum train-only student-concept response count among the concepts attached
to its exercise.  It reports metrics for all rows and the overlapping
diagnostic subsets ``n=0`` and ``n<3``, then computes same-dataset/same-seed
paired contrasts across the four graph variants.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.dataset import CognitiveDiagnosisDataset  # noqa: E402
from src.experiment_utils import compute_metrics  # noqa: E402
from src.trainer import (  # noqa: E402
    STRICT_CHECKPOINT_LOADING,
    _build_model,
    _require_graph_irt_checkpoint,
    _strip_module_prefix,
    _validate_checkpoint_data_identity,
)
from tools.summarize_validation_runs import _candidate_dirs, _tokens  # noqa: E402


VARIANTS = (
    "full",
    "no_evidence_propagation",
    "no_message_passing",
    "no_graph_calibration",
)
CONTRASTS: Tuple[Tuple[str, Mapping[str, float]], ...] = (
    (
        "propagation|state_on",
        {"full": 1.0, "no_evidence_propagation": -1.0},
    ),
    (
        "propagation|state_off",
        {"no_message_passing": 1.0, "no_graph_calibration": -1.0},
    ),
    (
        "state|propagation_on",
        {"full": 1.0, "no_message_passing": -1.0},
    ),
    (
        "state|propagation_off",
        {"no_evidence_propagation": 1.0, "no_graph_calibration": -1.0},
    ),
    (
        "interaction",
        {
            "full": 1.0,
            "no_evidence_propagation": -1.0,
            "no_message_passing": -1.0,
            "no_graph_calibration": 1.0,
        },
    ),
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _bucket_masks(support: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "all": np.ones(support.shape, dtype=bool),
        "n=0": support == 0,
        "n<3": support < 3,
    }


def _metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    if labels.size == 0:
        return {"auc": float("nan"), "bce_loss": float("nan"), "rmse": float("nan")}
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    bce = -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    base = compute_metrics(labels, (probs > 0.5).astype(float), probs)
    return {
        "auc": float(base["auc"]),
        "bce_loss": float(bce),
        "rmse": float(base["rmse"]),
    }


def _read_validation(
    checkpoint_dir: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    checkpoint_path = checkpoint_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require_graph_irt_checkpoint(checkpoint, str(checkpoint_path))
    loaded_args = checkpoint.get("args", {})
    info_dict = checkpoint.get("info_dict")
    if not isinstance(info_dict, dict):
        raise RuntimeError(f"checkpoint is missing info_dict: {checkpoint_path}")

    data_dir, _ = _validate_checkpoint_data_identity(
        SimpleNamespace(explicit_arg_dests=[]),
        loaded_args,
        info_dict,
    )
    valid = pd.read_csv(Path(data_dir) / "valid.csv")
    required = {"stu_id", "exer_id", "label"}
    missing = sorted(required - set(valid.columns))
    if missing:
        raise ValueError(f"validation split is missing columns {missing}: {data_dir}")
    student_map = info_dict["stu_id_map"]
    exercise_map = info_dict["exer_id_map"]
    valid = valid[
        valid["stu_id"].isin(student_map) & valid["exer_id"].isin(exercise_map)
    ].reset_index(drop=True)
    if valid.empty:
        raise ValueError(f"validation split has no train-seen rows: {data_dir}")

    dataset = CognitiveDiagnosisDataset(
        valid,
        student_map,
        exercise_map,
        info_dict["cpt_id_map"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = _build_model(loaded_args, info_dict, device)
    incompatible = model.load_state_dict(
        _strip_module_prefix(checkpoint["model_state_dict"]),
        strict=STRICT_CHECKPOINT_LOADING,
    )
    if not STRICT_CHECKPOINT_LOADING:
        missing_keys = list(getattr(incompatible, "missing_keys", []))
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"checkpoint mismatch: missing={missing_keys}, unexpected={unexpected_keys}"
            )
    if hasattr(model, "set_epoch"):
        model.set_epoch(int(checkpoint.get("epoch", 1)))
    model.eval()

    q_matrix = info_dict["q_matrix"]
    counts = info_dict["response_evidence_stats"]["student_concept_count"]
    if not torch.is_tensor(q_matrix):
        q_matrix = torch.as_tensor(q_matrix)
    if not torch.is_tensor(counts):
        counts = torch.as_tensor(counts)
    q_matrix = q_matrix.cpu()
    counts = counts.cpu()

    labels: List[float] = []
    probabilities: List[float] = []
    supports: List[float] = []
    with torch.no_grad():
        for student_ids, exercise_ids, y in loader:
            logits = model(
                student_ids.to(device),
                exercise_ids.to(device),
                return_logits=True,
            ).reshape(-1)
            probs = torch.sigmoid(logits).cpu()
            q_rows = q_matrix[exercise_ids] > 0
            row_counts = counts[student_ids]
            support = torch.where(
                q_rows,
                row_counts,
                torch.full_like(row_counts, float("inf")),
            ).min(dim=1).values
            labels.extend(y.reshape(-1).tolist())
            probabilities.extend(probs.tolist())
            supports.extend(support.tolist())

    label_array = np.asarray(labels, dtype=np.float64)
    prob_array = np.asarray(probabilities, dtype=np.float64)
    support_array = np.asarray(supports, dtype=np.float64)
    identity = {
        "run_dir": str(checkpoint_dir),
        "dataset": str(loaded_args.get("dataset_name", "")),
        "model_variant": str(loaded_args.get("model_variant", "full")),
        "train_evidence_mode": str(loaded_args.get("train_evidence_mode", "excluded")),
        "seed": int(loaded_args.get("seed", 0)),
        "best_epoch": int(checkpoint.get("epoch", 0)),
    }
    rows: List[Dict[str, object]] = []
    for bucket, mask in _bucket_masks(support_array).items():
        subset_labels = label_array[mask]
        subset_probs = prob_array[mask]
        rows.append(
            {
                **identity,
                "bucket": bucket,
                "rows": int(mask.sum()),
                "positives": int((subset_labels == 1).sum()),
                "negatives": int((subset_labels == 0).sum()),
                **_metrics(subset_labels, subset_probs),
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _paired_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    duplicate = frame.duplicated(
        ["dataset", "seed", "bucket", "model_variant"],
        keep=False,
    )
    if duplicate.any():
        columns = ["dataset", "seed", "bucket", "model_variant", "run_dir"]
        raise ValueError(f"duplicate paired cells:\n{frame.loc[duplicate, columns]}")

    records: List[Dict[str, object]] = []
    group_columns = ["dataset", "bucket"]
    for (dataset, bucket), group in frame.groupby(group_columns, sort=True):
        variants = set(group["model_variant"])
        absent = sorted(set(VARIANTS) - variants)
        if absent:
            raise ValueError(f"{dataset}/{bucket} is missing graph variants: {absent}")
        for metric in ("auc", "bce_loss", "rmse"):
            table = group.pivot(index="seed", columns="model_variant", values=metric)
            table = table.reindex(columns=VARIANTS)
            total_seed_cells = int(len(table))
            for contrast, weights in CONTRASTS:
                values = sum(table[cell] * coefficient for cell, coefficient in weights.items())
                values = values[np.isfinite(values)]
                records.append(
                    {
                        "dataset": dataset,
                        "bucket": bucket,
                        "metric": metric,
                        "contrast": contrast,
                        "total_seed_cells": total_seed_cells,
                        "n_paired": int(len(values)),
                        "mean": float(values.mean()) if len(values) else float("nan"),
                        "std": (
                            float(values.std(ddof=1))
                            if len(values) > 1
                            else 0.0 if len(values) == 1 else float("nan")
                        ),
                        "min": float(values.min()) if len(values) else float("nan"),
                        "max": float(values.max()) if len(values) else float("nan"),
                        "positive_seeds": int((values > 0).sum()),
                        "negative_seeds": int((values < 0).sum()),
                        "zero_seeds": int((values == 0).sum()),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_ids", default="", help="Comma-separated run-id suffixes.")
    parser.add_argument(
        "--checkpoint_dirs",
        default="",
        help="Comma-separated explicit checkpoint directories.",
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    checkpoint_dirs = _candidate_dirs(
        _tokens(args.run_ids),
        _tokens(args.checkpoint_dirs),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError("no matching checkpoint directories")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    rows: List[Dict[str, object]] = []
    for index, checkpoint_dir in enumerate(checkpoint_dirs, start=1):
        evaluated = _read_validation(
            checkpoint_dir,
            batch_size=max(1, int(args.batch_size)),
            device=device,
        )
        rows.extend(evaluated)
        identity = evaluated[0]
        print(
            f"[{index}/{len(checkpoint_dirs)}] "
            f"{identity['dataset']} seed={identity['seed']} "
            f"variant={identity['model_variant']}"
        )

    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "seed", "model_variant", "bucket"]
    )
    output = _resolve(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    contrasts = _paired_contrasts(frame)
    contrast_output = output.with_name(f"{output.stem}_paired_contrasts.csv")
    contrasts.to_csv(contrast_output, index=False)
    print(f"rows={len(frame)} -> {output}")
    print(f"contrasts={len(contrasts)} -> {contrast_output}")


if __name__ == "__main__":
    main()
