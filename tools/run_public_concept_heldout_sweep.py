#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build concept-heldout splits for ASSIST09/Junyi/ASSIST17 and train models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PublicHoldoutRecipe:
    source_dataset: str
    output_dataset: str
    source_dir: str
    train_frac: float = 0.70
    valid_frac: float = 0.10
    min_interactions: int = 15
    min_unique_concepts: int = 3
    concept_tolerance: float = 0.20


RECIPES: Dict[str, PublicHoldoutRecipe] = {
    "assist_09": PublicHoldoutRecipe(
        source_dataset="assist_09",
        output_dataset="assist_09_chold",
        source_dir="data/assist_09",
    ),
    "junyi": PublicHoldoutRecipe(
        source_dataset="junyi",
        output_dataset="junyi_chold",
        source_dir="data/junyi",
    ),
    "assist_17": PublicHoldoutRecipe(
        source_dataset="assist_17",
        output_dataset="assist_17_chold",
        source_dir="data/assist_17",
    ),
}


def parse_names(text: str) -> List[str]:
    if text.strip().lower() == "all":
        return list(RECIPES)
    out: List[str] = []
    for raw in text.replace("，", ",").split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in RECIPES:
            raise SystemExit(f"Unknown dataset {name!r}. Known: {', '.join(RECIPES)}")
        if name not in out:
            out.append(name)
    return out


def run_cmd(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("[CMD]", " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(list(cmd), cwd=ROOT, check=True)


def build_dataset(args: argparse.Namespace, recipe: PublicHoldoutRecipe) -> None:
    out_dir = ROOT / "data" / recipe.output_dataset
    if (out_dir / "manifest.json").exists() and not args.overwrite:
        print(f"[BUILD-SKIP] {recipe.output_dataset}: {out_dir / 'manifest.json'} exists", flush=True)
        return
    cmd = [
        sys.executable,
        "tools/build_public_concept_heldout_split.py",
        "--source-dataset",
        recipe.source_dataset,
        "--source-dir",
        recipe.source_dir,
        "--output-dataset",
        recipe.output_dataset,
        "--output-dir",
        str(Path("data") / recipe.output_dataset),
        "--train-frac",
        str(recipe.train_frac),
        "--valid-frac",
        str(recipe.valid_frac),
        "--min-interactions",
        str(recipe.min_interactions),
        "--min-unique-concepts",
        str(recipe.min_unique_concepts),
        "--concept-tolerance",
        str(recipe.concept_tolerance),
        "--seed",
        str(args.seed),
    ]
    if args.no_shuffle:
        cmd.append("--no-shuffle")
    if args.overwrite:
        cmd.append("--overwrite")
    run_cmd(cmd, dry_run=args.dry_run)


def write_plan(args: argparse.Namespace, recipes: Sequence[PublicHoldoutRecipe]) -> None:
    out_dir = ROOT / "results" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recipes": [asdict(recipe) for recipe in recipes],
        "variants": args.variants,
        "initial_gpus": args.initial_gpus,
        "watch_gpus": args.watch_gpus,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
    }
    (out_dir / "public_concept_heldout_plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def launch_training(args: argparse.Namespace, recipes: Sequence[PublicHoldoutRecipe]) -> None:
    cmd = [
        sys.executable,
        "tools/run_dynamic_dataset_experiments.py",
        "--run_id",
        args.run_id,
        "--datasets",
        ",".join(recipe.output_dataset for recipe in recipes),
        "--variants",
        args.variants,
        "--initial_gpus",
        args.initial_gpus,
        "--watch_gpus",
        args.watch_gpus,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--cpu_threads_per_job",
        str(args.cpu_threads_per_job),
        "--poll_interval",
        str(args.poll_interval),
        "--skip_r",
    ]
    if args.skip_small_experiments:
        cmd.append("--skip_small_experiments")
    if args.rerun_existing:
        cmd.append("--rerun_existing")
    run_cmd(cmd, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--run_id", default=f"public_concept_heldout_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--variants", default="full")
    parser.add_argument("--initial_gpus", default="0,2")
    parser.add_argument("--watch_gpus", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu_threads_per_job", type=int, default=2)
    parser.add_argument("--poll_interval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip_small_experiments", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes = [RECIPES[name] for name in parse_names(args.datasets)]
    write_plan(args, recipes)
    print(f"Run ID: {args.run_id}", flush=True)
    print("Datasets:", ", ".join(recipe.output_dataset for recipe in recipes), flush=True)
    if not args.skip_build:
        for recipe in recipes:
            build_dataset(args, recipe)
    if args.build_only:
        print("[DONE] build_only", flush=True)
        return
    launch_training(args, recipes)


if __name__ == "__main__":
    main()
