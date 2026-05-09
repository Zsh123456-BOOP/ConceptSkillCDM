#!/usr/bin/env python3
"""Collect selected public full-run result rows and logs into one folder."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


DATASET_RUNS = {
    "frcsub": "public_full_20260508_143744",
    "math2": "public_full_20260508_143744",
    "assist_15": "public_full_20260508_143744",
    "nips34": "public_full_20260508_143744",
    "assist_12": "public_refilter_20260509_010658",
    "ednet_kt1": "public_refilter_20260509_010658",
}

RESULT_COLUMNS = [
    "dataset",
    "run_id",
    "git_sha",
    "test_auc",
    "test_acc",
    "test_rmse",
    "best_val_auc",
    "model_epoch",
    "effective_batch_size",
    "train_rows",
    "valid_rows",
    "test_rows",
    "train_students",
    "train_items",
    "train_concepts",
    "filtered",
    "log_path",
    "run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=Path("results/experiment_results.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--server-logs-dir", type=Path, default=Path("server_logs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest(data_dir: Path, dataset: str) -> dict:
    path = data_dir / dataset / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pick_result_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for dataset, run_id in DATASET_RUNS.items():
        candidates = [
            row for row in rows
            if row.get("dataset") == dataset
            and row.get("model_variant") == "full"
            and f"/{run_id}/" in row.get("run_dir", "")
        ]
        if not candidates:
            raise ValueError(f"no result row found for dataset={dataset} run_id={run_id}")
        selected.append(candidates[-1])
    return selected


def enrich_row(row: dict[str, str], data_dir: Path) -> dict[str, str]:
    dataset = row["dataset"]
    manifest = load_manifest(data_dir, dataset)
    train = manifest.get("splits", {}).get("train", {})
    valid = manifest.get("splits", {}).get("valid", {})
    test = manifest.get("splits", {}).get("test", {})
    run_id = Path(row["run_dir"]).parent.name
    return {
        "dataset": dataset,
        "run_id": run_id,
        "git_sha": row.get("git_sha", ""),
        "test_auc": row.get("test_auc", ""),
        "test_acc": row.get("test_acc", ""),
        "test_rmse": row.get("test_rmse", ""),
        "best_val_auc": row.get("best_val_auc", ""),
        "model_epoch": row.get("model_epoch", ""),
        "effective_batch_size": row.get("effective_batch_size", ""),
        "train_rows": str(train.get("rows", "")),
        "valid_rows": str(valid.get("rows", "")),
        "test_rows": str(test.get("rows", "")),
        "train_students": str(train.get("students", "")),
        "train_items": str(train.get("items", "")),
        "train_concepts": str(train.get("concepts", "")),
        "filtered": "yes" if manifest.get("filter") else "no",
        "log_path": row.get("log_path", ""),
        "run_dir": row.get("run_dir", ""),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: str) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return value


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Public Full Results",
        "",
        "| Dataset | Run | Test AUC | ACC | RMSE | Best Val AUC | Epoch | Train Rows | Filtered |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {run_id} | {auc} | {acc} | {rmse} | {val} | {epoch} | {train_rows} | {filtered} |".format(
                dataset=row["dataset"],
                run_id=row["run_id"],
                auc=fmt_float(row["test_auc"]),
                acc=fmt_float(row["test_acc"]),
                rmse=fmt_float(row["test_rmse"]),
                val=fmt_float(row["best_val_auc"]),
                epoch=row["model_epoch"],
                train_rows=row["train_rows"],
                filtered=row["filtered"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_logs(output_dir: Path, rows: list[dict[str, str]], server_logs_dir: Path) -> None:
    log_out = output_dir / "logs"
    log_out.mkdir(parents=True, exist_ok=True)
    run_ids = sorted({row["run_id"] for row in rows})
    for row in rows:
        dataset = row["dataset"]
        train_log = Path(row["log_path"])
        if train_log.exists():
            shutil.copy2(train_log, log_out / f"{dataset}.train.log")
        stdout_log = Path("logs") / row["run_id"] / f"{dataset}_full.stdout.log"
        if stdout_log.exists():
            shutil.copy2(stdout_log, log_out / f"{dataset}.stdout.log")
    for run_id in run_ids:
        launcher = server_logs_dir / f"{run_id}_launcher.log"
        if launcher.exists():
            shutil.copy2(launcher, log_out / f"{run_id}.launcher.log")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [enrich_row(row, args.data_dir) for row in pick_result_rows(read_results(args.results_csv))]
    write_csv(args.output_dir / "full_results.csv", rows)
    write_markdown(args.output_dir / "full_results.md", rows)
    copy_logs(args.output_dir, rows, args.server_logs_dir)


if __name__ == "__main__":
    main()
