#!/usr/bin/env python
"""Compute paired contrasts for evidence-boundary and graph-path experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("val_auc", "val_bce_loss", "val_rmse")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _paired_table(
    frame: pd.DataFrame,
    *,
    column: str,
    required: Iterable[str],
) -> pd.DataFrame:
    duplicate = frame.duplicated(["dataset", "seed", column], keep=False)
    if duplicate.any():
        rows = frame.loc[duplicate, ["dataset", "seed", column, "run_dir"]]
        raise ValueError(f"duplicate paired cells:\n{rows.to_string(index=False)}")
    table = frame.pivot(index=["dataset", "seed"], columns=column)
    for metric in METRICS:
        if metric not in table.columns.get_level_values(0):
            raise ValueError(f"missing metric column: {metric}")
        missing = sorted(set(required) - set(table[metric].columns))
        if missing:
            raise ValueError(f"{metric} is missing paired cells: {missing}")
        if table[metric][list(required)].isna().any().any():
            raise ValueError(f"{metric} contains incomplete paired rows")
    return table


def _summarize(
    records: List[Dict[str, object]],
    *,
    dataset: str,
    metric: str,
    contrast: str,
    values: pd.Series,
) -> None:
    records.append(
        {
            "dataset": dataset,
            "metric": metric,
            "contrast": contrast,
            "n_seeds": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_seeds": int((values > 0).sum()),
            "negative_seeds": int((values < 0).sum()),
            "zero_seeds": int((values == 0).sum()),
        }
    )


def _contrast_rows(
    table: pd.DataFrame,
    contrasts: Iterable[Tuple[str, Dict[str, float]]],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for dataset in table.index.get_level_values("dataset").unique():
        dataset_table = table.xs(dataset, level="dataset")
        for metric in METRICS:
            values_by_cell = table[metric].xs(dataset, level="dataset")
            for name, weights in contrasts:
                values = sum(
                    coefficient * values_by_cell[cell]
                    for cell, coefficient in weights.items()
                )
                _summarize(
                    records,
                    dataset=str(dataset),
                    metric=metric,
                    contrast=name,
                    values=values,
                )
    return pd.DataFrame(records)


def _boundary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["model_variant"] == "full"].copy()
    required = ("excluded", "neutralized", "self_included")
    table = _paired_table(
        frame,
        column="train_evidence_mode",
        required=required,
    )
    return _contrast_rows(
        table,
        (
            ("I-N", {"self_included": 1.0, "neutralized": -1.0}),
            ("N-E", {"neutralized": 1.0, "excluded": -1.0}),
            ("I-E", {"self_included": 1.0, "excluded": -1.0}),
        ),
    )


def _graph(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["train_evidence_mode"] == "excluded"].copy()
    required = (
        "full",
        "no_evidence_propagation",
        "no_message_passing",
        "no_graph_calibration",
    )
    table = _paired_table(frame, column="model_variant", required=required)
    return _contrast_rows(
        table,
        (
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
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary_csv")
    parser.add_argument("--graph_csv")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    if not args.boundary_csv and not args.graph_csv:
        raise ValueError("provide --boundary_csv and/or --graph_csv")

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.boundary_csv:
        output = output_dir / "lea_boundary_paired_contrasts.csv"
        frame = _boundary(_resolve(args.boundary_csv))
        frame.to_csv(output, index=False)
        print(frame.to_string(index=False))
        print(f"saved: {output}")
    if args.graph_csv:
        output = output_dir / "lea_graph_2x2_paired_contrasts.csv"
        frame = _graph(_resolve(args.graph_csv))
        frame.to_csv(output, index=False)
        print(frame.to_string(index=False))
        print(f"saved: {output}")


if __name__ == "__main__":
    main()
