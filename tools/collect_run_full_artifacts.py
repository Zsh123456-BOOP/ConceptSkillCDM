#!/usr/bin/env python3
"""Collect full-run result rows and compact logs for a specific run id."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
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
    "manifest",
    "log_path",
    "run_dir",
]


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _read_manifest(dataset: str) -> tuple[dict, str]:
    path = Path("data") / dataset / "manifest.json"
    if not path.exists():
        return {}, ""
    return json.loads(path.read_text(encoding="utf-8")), path.as_posix()


def _select_result_rows(run_id: str, datasets: list[str]) -> pd.DataFrame:
    results_path = Path("results") / "experiment_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    df = pd.read_csv(results_path)
    if df.empty:
        raise ValueError(f"{results_path} is empty")
    run_mask = (
        df.get("log_path", pd.Series("", index=df.index)).astype(str).str.contains(f"/logs/{run_id}/", regex=False)
        | df.get("run_dir", pd.Series("", index=df.index)).astype(str).str.contains(f"/checkpoints/{run_id}/", regex=False)
    )
    variant_mask = df.get("model_variant", "").astype(str).eq("full")
    dataset_mask = df["dataset"].astype(str).isin(datasets)
    selected = df[run_mask & variant_mask & dataset_mask].copy()
    if selected.empty:
        raise ValueError(f"No full result rows found for run_id={run_id} datasets={datasets}")
    selected["_timestamp_sort"] = pd.to_datetime(selected["timestamp"], errors="coerce")
    selected = selected.sort_values(["dataset", "_timestamp_sort"]).groupby("dataset", as_index=False).tail(1)
    missing = [dataset for dataset in datasets if dataset not in set(selected["dataset"].astype(str))]
    if missing:
        raise ValueError(f"Missing full result rows for datasets: {missing}")
    return selected


def collect(run_id: str, datasets: list[str], output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir()
    (output_dir / "manifests").mkdir()

    rows = []
    selected = _select_result_rows(run_id, datasets)
    for _, row in selected.iterrows():
        dataset = str(row["dataset"])
        manifest, manifest_path = _read_manifest(dataset)
        if manifest:
            (output_dir / "manifests" / f"{dataset}.manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        train_log = Path(str(row.get("log_path", "")))
        if train_log.exists():
            _copy_if_exists(train_log, output_dir / "logs" / f"{dataset}.train.log")
        stdout_log = Path("logs") / run_id / f"{dataset}_full.stdout.log"
        _copy_if_exists(stdout_log, output_dir / "logs" / f"{dataset}.stdout.log")
        launcher_log = next(Path("server_logs").glob(f"*{run_id}*launcher*.log"), None) if Path("server_logs").exists() else None
        if launcher_log is not None:
            _copy_if_exists(launcher_log, output_dir / "logs" / launcher_log.name)

        train_stats = manifest.get("splits", {}).get("train", {})
        valid_stats = manifest.get("splits", {}).get("valid", {})
        test_stats = manifest.get("splits", {}).get("test", {})
        rows.append(
            {
                "dataset": dataset,
                "run_id": run_id,
                "git_sha": row.get("git_sha", ""),
                "test_auc": row.get("test_auc", ""),
                "test_acc": row.get("test_acc", ""),
                "test_rmse": row.get("test_rmse", ""),
                "best_val_auc": row.get("best_val_auc", ""),
                "model_epoch": row.get("model_epoch", ""),
                "effective_batch_size": row.get("effective_batch_size", ""),
                "train_rows": train_stats.get("rows", ""),
                "valid_rows": valid_stats.get("rows", ""),
                "test_rows": test_stats.get("rows", ""),
                "train_students": train_stats.get("students", ""),
                "train_items": train_stats.get("items", ""),
                "train_concepts": train_stats.get("concepts", ""),
                "manifest": manifest_path,
                "log_path": row.get("log_path", ""),
                "run_dir": row.get("run_dir", ""),
            }
        )

    csv_path = output_dir / "full_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Dataset | Test AUC | ACC | RMSE | Best Val AUC | Epoch | Train Rows | Students | Items | Concepts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {test_auc} | {test_acc} | {test_rmse} | {best_val_auc} | {model_epoch} | {train_rows} | {train_students} | {train_items} | {train_concepts} |".format(**row)
        )
    (output_dir / "full_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[collect] wrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    output_dir = args.output_dir or Path("results") / f"{args.run_id}_full_artifacts"
    collect(args.run_id, datasets, output_dir)


if __name__ == "__main__":
    main()
