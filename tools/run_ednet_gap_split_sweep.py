#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build several raw-EdNet evidence-gap splits and launch the ablation sweep.

The recipes are intentionally explicit so the server command stays auditable.
Each dataset is built from the raw EdNet-KT1 archives, then trained through the
existing dynamic mechanism launcher.
"""

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
class SplitRecipe:
    name: str
    min_interactions: int
    max_interactions: int
    target_users: int
    train_frac: float
    valid_frac: float
    seed: int
    rationale: str


RECIPES: Dict[str, SplitRecipe] = {
    "ednet_kt1_gap_t70u5000": SplitRecipe(
        name="ednet_kt1_gap_t70u5000",
        min_interactions=60,
        max_interactions=600,
        target_users=5000,
        train_frac=0.70,
        valid_frac=0.10,
        seed=20260529,
        rationale="AUC-first split: longer train history, still keeps a non-trivial evidence-gap test tail.",
    ),
    "ednet_kt1_gap_t60u5000": SplitRecipe(
        name="ednet_kt1_gap_t60u5000",
        min_interactions=60,
        max_interactions=600,
        target_users=5000,
        train_frac=0.60,
        valid_frac=0.10,
        seed=20260529,
        rationale="Primary balanced split: more train history than t45 while preserving a large gap cohort.",
    ),
    "ednet_kt1_gap_t50u5000": SplitRecipe(
        name="ednet_kt1_gap_t50u5000",
        min_interactions=60,
        max_interactions=600,
        target_users=5000,
        train_frac=0.50,
        valid_frac=0.10,
        seed=20260529,
        rationale="Gap-first split: larger test tail for direct-unseen/bridgeable subgroup diagnostics.",
    ),
    "ednet_kt1_gap_t65u3000_long": SplitRecipe(
        name="ednet_kt1_gap_t65u3000_long",
        min_interactions=120,
        max_interactions=900,
        target_users=3000,
        train_frac=0.65,
        valid_frac=0.10,
        seed=20260529,
        rationale="Long-history split: tests whether richer learner histories recover old-EdNet AUC while retaining gaps.",
    ),
}


def parse_recipe_names(text: str) -> List[str]:
    if text.strip().lower() == "all":
        return list(RECIPES)
    names: List[str] = []
    for raw in text.replace("，", ",").split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in RECIPES:
            known = ", ".join(RECIPES)
            raise SystemExit(f"Unknown recipe {name!r}. Known recipes: {known}")
        if name not in names:
            names.append(name)
    return names


def run_cmd(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    print("[CMD]", " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=ROOT, check=True)


def build_split(args: argparse.Namespace, recipe: SplitRecipe) -> None:
    out_dir = ROOT / "data" / recipe.name
    manifest = out_dir / "manifest.json"
    if manifest.exists() and not args.overwrite:
        print(f"[BUILD-SKIP] {recipe.name}: {manifest} exists", flush=True)
        return
    cmd = [
        sys.executable,
        "tools/build_ednet_gap_from_raw.py",
        "--raw-dir",
        str(args.raw_dir),
        "--output-dir",
        str(Path("data") / recipe.name),
        "--output-dataset",
        recipe.name,
        "--min-interactions",
        str(recipe.min_interactions),
        "--max-interactions",
        str(recipe.max_interactions),
        "--target-users",
        str(recipe.target_users),
        "--max-users-scan",
        str(args.max_users_scan),
        "--train-frac",
        str(recipe.train_frac),
        "--valid-frac",
        str(recipe.valid_frac),
        "--seed",
        str(recipe.seed),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run_cmd(cmd, dry_run=args.dry_run)


def write_plan(run_root: Path, recipes: Sequence[SplitRecipe], args: argparse.Namespace) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recipes": [asdict(recipe) for recipe in recipes],
        "variants": args.variants,
        "epochs": args.epochs,
        "patience": args.patience,
        "initial_gpus": args.initial_gpus,
        "watch_gpus": args.watch_gpus,
    }
    (run_root / "split_plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def launch_training(args: argparse.Namespace, recipes: Sequence[SplitRecipe]) -> None:
    datasets = ",".join(recipe.name for recipe in recipes)
    cmd = [
        sys.executable,
        "tools/run_dynamic_dataset_experiments.py",
        "--run_id",
        args.run_id,
        "--datasets",
        datasets,
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
    parser.add_argument("--recipes", default="all", help="Comma-separated recipe names or 'all'.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/_public_raw/ednet_kt1"))
    parser.add_argument("--max-users-scan", type=int, default=0)
    parser.add_argument("--run_id", default=f"ednet_gap_split_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument(
        "--variants",
        default="full,no_A,no_E,E_shuffle_student,A_fused_neutralE,no_A_fair",
    )
    parser.add_argument("--initial_gpus", default="0,2")
    parser.add_argument("--watch_gpus", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu_threads_per_job", type=int, default=2)
    parser.add_argument("--poll_interval", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build_only", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--skip_small_experiments", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes = [RECIPES[name] for name in parse_recipe_names(args.recipes)]
    run_root = ROOT / "results" / args.run_id
    write_plan(run_root, recipes, args)
    print(f"Run ID: {args.run_id}", flush=True)
    print("Recipes:", ", ".join(recipe.name for recipe in recipes), flush=True)
    if not args.skip_build:
        for recipe in recipes:
            build_split(args, recipe)
    if args.build_only:
        print("[DONE] build_only", flush=True)
        return
    launch_training(args, recipes)


if __name__ == "__main__":
    main()
