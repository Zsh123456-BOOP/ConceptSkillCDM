#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-shot launcher for larger graph/module3 grid search.

This script is a thin wrapper around `run_module3_grid.py`, preconfigured for
larger searches (default: 64 grid points per dataset).

Example:
python run_graph64_search.py --datasets assist_09,junyi --seeds 42 --gpus 0,1,2,3 --epochs 15 --max_concurrent 4 --max_per_gpu 1
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch large graph/module3 grid search in one shot.")
    parser.add_argument("--datasets", type=str, default="assist_09,junyi")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--grid_points", type=int, default=64)
    parser.add_argument("--max_concurrent", type=int, default=4)
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_interval", type=int, default=15)
    parser.add_argument("--only_variants", type=str, default="full,no_module3")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def build_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "run_module3_grid.py",
        "--datasets",
        args.datasets,
        "--seeds",
        args.seeds,
        "--gpus",
        args.gpus,
        "--epochs",
        str(args.epochs),
        "--grid_points",
        str(args.grid_points),
        "--max_concurrent",
        str(args.max_concurrent),
        "--max_per_gpu",
        str(args.max_per_gpu),
        "--poll_interval",
        str(args.poll_interval),
        "--only_variants",
        args.only_variants,
    ]
    if args.rerun_existing:
        cmd.append("--rerun_existing")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def main() -> None:
    args = parse_args()
    cmd = build_cmd(args)
    print("[LAUNCH]", shlex.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

