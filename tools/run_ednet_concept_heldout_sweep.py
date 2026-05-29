#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build EdNet concept-heldout splits and launch full/control training.

This is the main-problem split suggested for testing student-specific unseen
concept diagnosis.  By default it only trains `full`; pass `--variants` later
to add no_CRG/self-only/random-support controls after the best split is chosen.
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
class HoldoutRecipe:
    name: str
    min_interactions: int
    max_interactions: int
    target_users: int
    train_frac: float
    valid_frac: float
    min_unique_concepts: int
    concept_tolerance: float
    seed: int
    rationale: str


RECIPES: Dict[str, HoldoutRecipe] = {
    "ednet_kt1_chold_t70u2000": HoldoutRecipe(
        name="ednet_kt1_chold_t70u2000",
        min_interactions=60,
        max_interactions=600,
        target_users=2000,
        train_frac=0.70,
        valid_frac=0.10,
        min_unique_concepts=6,
        concept_tolerance=0.20,
        seed=20260529,
        rationale="Old-scale 7:1:2 concept-heldout split; first candidate for AUC recovery.",
    ),
    "ednet_kt1_chold_t70u2500": HoldoutRecipe(
        name="ednet_kt1_chold_t70u2500",
        min_interactions=50,
        max_interactions=500,
        target_users=2500,
        train_frac=0.70,
        valid_frac=0.10,
        min_unique_concepts=5,
        concept_tolerance=0.20,
        seed=20260529,
        rationale="Slightly broader old-scale split; more students but bounded per-user length.",
    ),
    "ednet_kt1_chold_t75u1800": HoldoutRecipe(
        name="ednet_kt1_chold_t75u1800",
        min_interactions=80,
        max_interactions=650,
        target_users=1800,
        train_frac=0.75,
        valid_frac=0.10,
        min_unique_concepts=6,
        concept_tolerance=0.18,
        seed=20260529,
        rationale="AUC-first concept-heldout split with more training evidence per student.",
    ),
    "ednet_kt1_chold_t70u1200_long": HoldoutRecipe(
        name="ednet_kt1_chold_t70u1200_long",
        min_interactions=120,
        max_interactions=900,
        target_users=1200,
        train_frac=0.70,
        valid_frac=0.10,
        min_unique_concepts=8,
        concept_tolerance=0.20,
        seed=20260529,
        rationale="Long-history concept-heldout split; closest to strong learner-history diagnosis.",
    ),
}


def parse_recipe_names(text: str) -> List[str]:
    if text.strip().lower() == "all":
        return list(RECIPES)
    out: List[str] = []
    for raw in text.replace("，", ",").split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in RECIPES:
            raise SystemExit(f"Unknown recipe {name!r}. Known recipes: {', '.join(RECIPES)}")
        if name not in out:
            out.append(name)
    return out


def run_cmd(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("[CMD]", " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(list(cmd), cwd=ROOT, check=True)


def build_split(args: argparse.Namespace, recipe: HoldoutRecipe) -> None:
    out_dir = ROOT / "data" / recipe.name
    manifest = out_dir / "manifest.json"
    if manifest.exists() and not args.overwrite:
        print(f"[BUILD-SKIP] {recipe.name}: {manifest} exists", flush=True)
        return
    cmd = [
        sys.executable,
        "tools/build_ednet_concept_heldout_from_raw.py",
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
        "--min-unique-concepts",
        str(recipe.min_unique_concepts),
        "--concept-tolerance",
        str(recipe.concept_tolerance),
        "--seed",
        str(recipe.seed),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run_cmd(cmd, dry_run=args.dry_run)


def write_plan(args: argparse.Namespace, recipes: Sequence[HoldoutRecipe]) -> None:
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
    }
    (out_dir / "concept_heldout_plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def launch_training(args: argparse.Namespace, recipes: Sequence[HoldoutRecipe]) -> None:
    cmd = [
        sys.executable,
        "tools/run_dynamic_dataset_experiments.py",
        "--run_id",
        args.run_id,
        "--datasets",
        ",".join(recipe.name for recipe in recipes),
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
    parser.add_argument("--recipes", default="all")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/_public_raw/ednet_kt1"))
    parser.add_argument("--max-users-scan", type=int, default=40000)
    parser.add_argument("--run_id", default=f"ednet_concept_heldout_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--variants", default="full")
    parser.add_argument("--initial_gpus", default="0,2")
    parser.add_argument("--watch_gpus", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu_threads_per_job", type=int, default=2)
    parser.add_argument("--poll_interval", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip_small_experiments", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes = [RECIPES[name] for name in parse_recipe_names(args.recipes)]
    write_plan(args, recipes)
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
