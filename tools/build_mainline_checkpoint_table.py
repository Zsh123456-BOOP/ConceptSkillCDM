#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build the checkpoint table consumed by the mainline evidence scripts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_abce_ablation import load_result_rows


VARIANT_MAP = {
    "full": "full",
    "no_A": "no_CRG",
    "no_E": "no_LCRF",
}

NOTE_MAP = {
    "full": "full CRG+LCRF checkpoint",
    "no_CRG": "CRG removal measures roadmap contribution",
    "no_LCRF": "LCRF removal measures support-filter contribution",
}

FIELDNAMES = [
    "dataset",
    "variant",
    "checkpoint_path",
    "auc",
    "bce",
    "acc",
    "rmse",
    "auc_drop_from_full",
    "bce_increase_from_full",
    "train_only_crg_check_passed",
    "notes",
]


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _truthy(value: Any, default: bool = True) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok", "passed"}


def _eligible(row: Dict[str, Any], profile: str, seed: Optional[int]) -> bool:
    if str(row.get("profile", "")) != profile:
        return False
    if str(row.get("ablation", "")) not in VARIANT_MAP:
        return False
    if seed is not None and str(row.get("seed", "")) != str(seed):
        return False
    if str(row.get("status", "")).lower() not in {"ok", "metrics_ok"}:
        return False
    if str(row.get("ablation_valid", "")).lower() == "false":
        return False
    save_dir = Path(str(row.get("save_dir", "")))
    return (save_dir / "best_model.pth").exists()


def _best_by_auc(rows: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_auc: Optional[float] = None
    for row in rows:
        auc = _to_float(row.get("test_auc"))
        if auc is None:
            continue
        if best is None or best_auc is None or auc > best_auc:
            best = row
            best_auc = auc
    return best


def build_table(
    *,
    result_csv: Path,
    run_id: str,
    datasets: List[str],
    profile: str,
    seed: Optional[int],
    allow_missing: bool,
) -> List[Dict[str, Any]]:
    rows = load_result_rows(result_csv, run_id=run_id)
    selected = [row for row in rows if str(row.get("dataset", "")) in set(datasets) and _eligible(row, profile, seed)]
    output: List[Dict[str, Any]] = []
    missing: List[str] = []

    for dataset in datasets:
        by_variant: Dict[str, Dict[str, Any]] = {}
        dataset_rows = [row for row in selected if str(row.get("dataset", "")) == dataset]
        for ablation, variant in VARIANT_MAP.items():
            best = _best_by_auc(row for row in dataset_rows if str(row.get("ablation", "")) == ablation)
            if best is None:
                missing.append(f"{dataset}/{ablation}")
                continue
            by_variant[variant] = best

        full_auc = _to_float(by_variant.get("full", {}).get("test_auc"))
        for variant in ("full", "no_CRG", "no_LCRF"):
            row = by_variant.get(variant)
            if row is None:
                continue
            save_dir = Path(str(row["save_dir"]))
            checkpoint = (save_dir / "best_model.pth").resolve()
            test_json = _read_json(save_dir / "test_results.json")
            auc = _to_float(row.get("test_auc"))
            auc_drop = None if full_auc is None or auc is None else full_auc - auc
            clean = test_json.get("train_only_split_hygiene", row.get("clean_baseline", True))
            output.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "checkpoint_path": str(checkpoint),
                    "auc": "" if auc is None else auc,
                    "bce": "",
                    "acc": row.get("test_acc", ""),
                    "rmse": row.get("test_rmse", ""),
                    "auc_drop_from_full": "" if auc_drop is None else auc_drop,
                    "bce_increase_from_full": "",
                    "train_only_crg_check_passed": _truthy(clean),
                    "notes": NOTE_MAP[variant],
                }
            )

    if missing and not allow_missing:
        raise SystemExit("Missing successful checkpoints: " + ", ".join(missing))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--result-csv", default="results/abce_ablation_diagnosis.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--profile", default="best")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    datasets = _parse_csv(args.datasets)
    out_path = Path(args.output or f"results/{args.run_id}/main_table/table_main_ablation.csv")
    table = build_table(
        result_csv=Path(args.result_csv),
        run_id=args.run_id,
        datasets=datasets,
        profile=args.profile,
        seed=args.seed,
        allow_missing=bool(args.allow_missing),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(table)
    print(f"[ok] wrote {out_path} rows={len(table)}")


if __name__ == "__main__":
    main()
