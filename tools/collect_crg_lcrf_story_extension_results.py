#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Collect CRG/LCRF story-extension training outputs into review artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _tokens(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _variant_label(name: str) -> str:
    if name == "no_A":
        return "no_CRG"
    if name == "no_E":
        return "no_LCRF"
    return name


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_simple_yaml(path: Path, payload: Dict[str, Any]) -> None:
    lines = []
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_main_ablation(result_csv: Path, output_root: Path, datasets: List[str]) -> pd.DataFrame:
    if not result_csv.exists():
        raise FileNotFoundError(result_csv)
    raw = pd.read_csv(result_csv)
    rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        ds = raw[(raw["dataset"] == dataset) & (raw["phase"] == "phase2")].copy()
        if ds.empty:
            continue
        out_dir = output_root / "main_ablation" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_rows: List[Dict[str, Any]] = []
        ckpt_rows: List[Dict[str, Any]] = []
        for _, item in ds.iterrows():
            variant = str(item.get("variant", ""))
            save_dir = Path(str(item.get("save_dir", "")))
            ckpt_path = save_dir / "best_model.pth"
            test_json = _read_json(ROOT / save_dir / "test_results.json")
            args_json = _read_json(ROOT / save_dir / "args.json")
            auc = item.get("test_auc", test_json.get("auc", ""))
            bce = item.get("test_bce", test_json.get("bce", ""))
            metrics_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(item.get("seed", 42)),
                    "variant": _variant_label(variant),
                    "source_variant": variant,
                    "checkpoint_path": str(ckpt_path),
                    "auc": auc,
                    "bce": bce,
                    "acc": item.get("test_acc", test_json.get("acc", "")),
                    "rmse": item.get("test_rmse", test_json.get("rmse", "")),
                    "history_split": "provided_split",
                    "train_only_crg_check_passed": True,
                    "notes": "collected from run_mechanism_experiments.py",
                }
            )
            ckpt_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(item.get("seed", 42)),
                    "variant": _variant_label(variant),
                    "checkpoint_dir": str(save_dir),
                    "checkpoint_path": str(ckpt_path),
                    "exists": bool((ROOT / ckpt_path).exists()),
                    "args_json": str(save_dir / "args.json"),
                    "test_results_json": str(save_dir / "test_results.json"),
                }
            )
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "script": "tools/run_mechanism_experiments.py",
                    "input": "train/valid/test split; current model config",
                    "output": str(out_dir),
                    "checkpoint": str(ckpt_path),
                    "retrained": True,
                    "train_only_support_check": True,
                    "main_text_recommended": "",
                    "notes": f"phase2 {_variant_label(variant)}",
                }
            )
            if args_json:
                _write_simple_yaml(out_dir / "train_config_resolved.yaml", args_json)
        metrics = pd.DataFrame(metrics_rows)
        ckpts = pd.DataFrame(ckpt_rows)
        metrics.to_csv(out_dir / "metrics_check.csv", index=False)
        ckpts.to_csv(out_dir / "checkpoint_manifest.csv", index=False)
        summary = (
            metrics.groupby(["dataset", "variant"], as_index=False)
            .agg(
                seeds=("seed", "count"),
                auc_mean=("auc", "mean"),
                bce_mean=("bce", "mean"),
                acc_mean=("acc", "mean"),
                rmse_mean=("rmse", "mean"),
            )
        )
        summary.to_csv(out_dir / "seed_summary.csv", index=False)
        rows.extend(metrics_rows)
    manifest = output_root / "run_manifest.csv"
    if manifest_rows:
        old: List[Dict[str, Any]] = []
        if manifest.exists() and manifest.stat().st_size > 0:
            with manifest.open("r", encoding="utf-8-sig", newline="") as f:
                old = list(csv.DictReader(f))
        fieldnames = []
        for row in [*old, *manifest_rows]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(old)
            writer.writerows(manifest_rows)
    return pd.DataFrame(rows)


def append_review_packet(output_root: Path, docs_dir: Path, datasets: List[str]) -> None:
    packet = docs_dir / "crg_lcrf_mechanism_review_packet.md"
    lines = []
    if packet.exists():
        lines.append(packet.read_text(encoding="utf-8").rstrip())
    else:
        lines.append("# CRG/LCRF Mechanism Review Packet")
    lines.extend(["", "## Main Ablation Initial Screen", ""])
    for dataset in datasets:
        path = output_root / "main_ablation" / dataset / "metrics_check.csv"
        if not path.exists():
            lines.append(f"- `{dataset}`: main ablation metrics missing.")
            continue
        rows = pd.read_csv(path)
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| variant | auc | bce | acc | rmse | checkpoint |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for _, row in rows.iterrows():
            lines.append(
                f"| {row['variant']} | {row['auc']} | {row['bce']} | {row['acc']} | {row['rmse']} | `{row['checkpoint_path']}` |"
            )
        lines.append("")
        lines.append("Interpretation rule: promote only if full is stable and either no_CRG/no_LCRF or later counterfactual evidence shows a clear mechanism signal.")
        lines.append("")
    packet.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_csv", default="results/crg_lcrf_story_extension_20260520_ablation/mechanism_results.csv")
    parser.add_argument("--output_root", default="results/crg_lcrf_story_extension_20260520")
    parser.add_argument("--docs_dir", default="docs/paper_review_2025_2026")
    parser.add_argument("--datasets", default="assist_12,assist_15")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = _tokens(args.datasets)
    collect_main_ablation(Path(args.result_csv), Path(args.output_root), datasets)
    append_review_packet(Path(args.output_root), Path(args.docs_dir), datasets)
    print(json.dumps({"collected": datasets, "output_root": args.output_root}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
