#!/usr/bin/env python
"""Collect validation-selected metrics from targeted experiment checkpoints.

The script never opens a test split.  It reads each checkpoint's stored
train/validation metrics at the selected epoch and writes both per-run rows and
mean/std summaries grouped by dataset, model variant, and training evidence
mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"


def _tokens(value: str) -> List[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _candidate_dirs(run_ids: Iterable[str], explicit_dirs: Iterable[str]) -> List[Path]:
    candidates = {
        path.resolve()
        for run_id in run_ids
        for path in CHECKPOINTS.glob(f"*_{run_id}")
        if path.is_dir()
    }
    for value in explicit_dirs:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {path}")
        candidates.add(path.resolve())
    return sorted(candidates)


def _metric(row: Dict[str, object], prefix: str, metrics: Dict[str, object]) -> None:
    for name in ("auc", "acc", "rmse", "bce_loss", "loss"):
        value = metrics.get(name)
        if value is not None:
            row[f"{prefix}_{name}"] = float(value)


def _read_run(path: Path) -> Dict[str, object]:
    args_path = path / "args.json"
    checkpoint_path = path / "best_model.pth"
    validation_path = path / "validation_result.json"
    for required in (args_path, checkpoint_path, validation_path):
        if not required.is_file():
            raise FileNotFoundError(f"required run artifact is missing: {required}")

    with args_path.open(encoding="utf-8") as handle:
        args = json.load(handle)
    with validation_path.open(encoding="utf-8") as handle:
        validation = json.load(handle)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    row: Dict[str, object] = {
        "run_dir": str(path),
        "architecture": str(checkpoint.get("architecture", "")),
        "dataset": str(args.get("dataset_name", validation.get("dataset", ""))),
        "model_variant": str(
            args.get("model_variant", validation.get("model_variant", "full"))
        ),
        "train_evidence_mode": str(
            args.get(
                "train_evidence_mode",
                validation.get("train_evidence_mode", "excluded"),
            )
        ),
        "gec_mode": str(args.get("gec_mode", "v1")),
        "seed": int(args.get("seed", validation.get("seed", 0))),
        "best_epoch": int(checkpoint.get("epoch", validation.get("best_epoch", 0))),
    }
    _metric(row, "train", checkpoint.get("train_metrics", {}))
    _metric(row, "val", checkpoint.get("val_metrics", {}))

    test_path = path / "test_results.json"
    if test_path.is_file():
        with test_path.open(encoding="utf-8") as handle:
            test = json.load(handle)
        _metric(row, "test", test.get("metrics", {}))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_ids",
        default="",
        help="Comma-separated run-id suffixes used by run_graph_ablation.py.",
    )
    parser.add_argument(
        "--checkpoint_dirs",
        default="",
        help="Comma-separated explicit checkpoint directories.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Per-run CSV; a sibling *_summary.csv is written automatically.",
    )
    args = parser.parse_args()

    directories = _candidate_dirs(
        _tokens(args.run_ids),
        _tokens(args.checkpoint_dirs),
    )
    if not directories:
        raise FileNotFoundError("no matching checkpoint directories")

    runs = pd.DataFrame([_read_run(path) for path in directories])
    runs = runs.sort_values(
        [
            "architecture",
            "dataset",
            "model_variant",
            "gec_mode",
            "train_evidence_mode",
            "seed",
        ]
    ).reset_index(drop=True)

    output = Path(args.output_csv)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output, index=False)

    metric_columns = [
        column
        for column in runs.columns
        if column.startswith(("train_", "val_", "test_"))
        and pd.api.types.is_numeric_dtype(runs[column])
    ]
    grouped = runs.groupby(
        [
            "architecture",
            "dataset",
            "model_variant",
            "gec_mode",
            "train_evidence_mode",
        ],
        dropna=False,
    )
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary.insert(0, "n_seeds", grouped.size())
    summary = summary.reset_index()

    summary_path = output.with_name(f"{output.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"runs={len(runs)} -> {output}")
    print(f"groups={len(summary)} -> {summary_path}")


if __name__ == "__main__":
    main()
