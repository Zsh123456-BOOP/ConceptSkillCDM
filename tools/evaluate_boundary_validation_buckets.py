#!/usr/bin/env python
"""Evaluate E/N/I checkpoints on train-count validation subsets.

This tool reuses ``evaluate_graph_validation_buckets._read_validation`` and
therefore reads only ``train.csv``-derived checkpoint statistics and
``valid.csv`` rows, never ``test.csv``.  It keeps the ``full`` model variant,
reports ``all``, ``n=0`` and ``n<3`` metrics, and summarizes paired I-N, N-E
and I-E differences for the same dataset, seed and bucket.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

MODES = ("excluded", "neutralized", "self_included")
CONTRASTS: Tuple[Tuple[str, Mapping[str, float]], ...] = (
    ("I-N", {"self_included": 1.0, "neutralized": -1.0}),
    ("N-E", {"neutralized": 1.0, "excluded": -1.0}),
    ("I-E", {"self_included": 1.0, "excluded": -1.0}),
)
SAMPLE_COLUMNS = ("rows", "positives", "negatives")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _paired_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[
        (frame["model_variant"] == "full")
        & frame["train_evidence_mode"].isin(MODES)
    ].copy()
    if frame.empty:
        raise ValueError("no full E/N/I validation-bucket rows")

    key = ["dataset", "seed", "bucket", "train_evidence_mode"]
    duplicate = frame.duplicated(key, keep=False)
    if duplicate.any():
        columns = [*key, "run_dir"]
        raise ValueError(f"duplicate paired cells:\n{frame.loc[duplicate, columns]}")

    records: List[Dict[str, object]] = []
    for (dataset, bucket), group in frame.groupby(["dataset", "bucket"], sort=True):
        absent = sorted(set(MODES) - set(group["train_evidence_mode"]))
        if absent:
            raise ValueError(f"{dataset}/{bucket} is missing evidence modes: {absent}")

        sample_tables = {
            column: group.pivot(
                index="seed",
                columns="train_evidence_mode",
                values=column,
            ).reindex(columns=MODES)
            for column in SAMPLE_COLUMNS
        }
        for column, table in sample_tables.items():
            if table.isna().any().any():
                raise ValueError(
                    f"{dataset}/{bucket} has incomplete {column} sample cells"
                )
            if not table.eq(table.iloc[:, 0], axis=0).all().all():
                raise ValueError(
                    f"{dataset}/{bucket} has mode-dependent {column} counts"
                )

        sample_by_seed = {
            column: table.iloc[:, 0].astype(int)
            for column, table in sample_tables.items()
        }
        for metric in ("auc", "bce_loss", "rmse"):
            table = group.pivot(
                index="seed",
                columns="train_evidence_mode",
                values=metric,
            ).reindex(columns=MODES)
            total_seed_cells = int(len(table))
            for contrast, weights in CONTRASTS:
                values = sum(
                    table[mode] * coefficient
                    for mode, coefficient in weights.items()
                )
                values = values[np.isfinite(values)]
                paired_seeds = values.index
                record: Dict[str, object] = {
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
                for column, sample in sample_by_seed.items():
                    paired_sample = sample.loc[paired_seeds]
                    record[f"{column}_min"] = (
                        int(paired_sample.min()) if len(paired_sample) else 0
                    )
                    record[f"{column}_max"] = (
                        int(paired_sample.max()) if len(paired_sample) else 0
                    )
                records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    import torch

    from tools.evaluate_graph_validation_buckets import _read_validation
    from tools.summarize_validation_runs import _candidate_dirs, _tokens

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
        identity = evaluated[0]
        if (
            identity["model_variant"] != "full"
            or identity["train_evidence_mode"] not in MODES
        ):
            print(
                f"[{index}/{len(checkpoint_dirs)}] skipped "
                f"{identity['dataset']} seed={identity['seed']} "
                f"variant={identity['model_variant']} "
                f"mode={identity['train_evidence_mode']}"
            )
            continue
        rows.extend(evaluated)
        print(
            f"[{index}/{len(checkpoint_dirs)}] "
            f"{identity['dataset']} seed={identity['seed']} "
            f"mode={identity['train_evidence_mode']}"
        )

    if not rows:
        raise ValueError("no full E/N/I checkpoints were selected")
    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "seed", "train_evidence_mode", "bucket"]
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
