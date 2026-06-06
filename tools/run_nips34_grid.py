#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run a bounded NIPS34 full-model tuning grid.

The grid intentionally keeps the interpretable CRG/LCRF architecture intact. It only
varies training regularization and relation-support strength parameters that
are already exposed by the project.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_utils import calc_effective_max_concurrent, parse_gpu_ids, pick_gpu_with_slot_round_robin  # noqa: E402
from run_abce_ablation import AblationSpec, JobSpec, _build_job_env, build_command, collect_result  # noqa: E402


GRID_KEYS: Tuple[str, ...] = (
    "learning_rate",
    "dropout",
    "weight_decay",
    "graph_topk",
    "lambda_sparse",
    "graph_tau_init",
    "graph_prior_logit_scale",
    "graph_propagation_alpha",
    "graph_query_readout_scale",
    "graph_query_readout_2hop_scale",
    "personal_max_alpha",
    "personal_delta_scale",
    "personal_query_correction_scale",
    "personal_query_correction_max_ratio",
)


@dataclass
class GridJob:
    combo_id: str
    params: Dict[str, Any]
    job: JobSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default="", help="Run id. Default uses timestamp.")
    parser.add_argument("--gpus", default="0,1,3", help="Physical GPU ids, comma separated.")
    parser.add_argument("--max_concurrent", type=int, default=0, help="Default: one job per GPU.")
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit_jobs", type=int, default=0, help="Optional cap for debug runs.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--generate_diagnosis", action="store_true")
    return parser.parse_args()


def _dedupe(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        key = tuple(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return out


def build_grid() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = [
        {},
        {"learning_rate": 2e-4, "dropout": 0.10},
        {"learning_rate": 3e-4, "dropout": 0.10},
        {"learning_rate": 1e-4, "dropout": 0.15},
        {"learning_rate": 2e-4, "dropout": 0.15},
        {"learning_rate": 3e-4, "dropout": 0.15},
        {"learning_rate": 1e-4, "dropout": 0.20},
        {"learning_rate": 2e-4, "dropout": 0.20},
        {"learning_rate": 3e-4, "dropout": 0.20},
        {"weight_decay": 5e-5},
        {"graph_topk": 16},
        {"graph_topk": 32},
        {"lambda_sparse": 0.15},
        {"lambda_sparse": 0.50},
        {"graph_tau_init": 0.45},
        {"graph_tau_init": 0.70},
        {"graph_prior_logit_scale": 0.35},
        {"graph_prior_logit_scale": 0.75},
        {
            "graph_propagation_alpha": 0.05,
            "graph_query_readout_scale": 0.04,
            "graph_query_readout_2hop_scale": 0.02,
        },
        {
            "personal_max_alpha": 0.35,
            "personal_delta_scale": 5.0,
            "personal_query_correction_scale": 0.30,
            "personal_query_correction_max_ratio": 0.08,
        },
    ]
    return _dedupe(candidates)[:20]


def result_paths(run_id: str) -> Tuple[Path, Path, Path]:
    base = Path("results") / run_id
    return (
        base / "nips34_grid_results.csv",
        base / "nips34_grid_best.csv",
        base / "nips34_grid_status.json",
    )


def append_grid_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "combo_id",
        "status",
        "exit_code",
        "test_auc",
        "best_val_auc",
        "best_epoch",
        "model_epoch",
        "test_acc",
        "test_rmse",
        "log_file",
        *GRID_KEYS,
        "params_json",
        "save_dir",
        "log_dir",
        "failure_reason",
        "failure_stage",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _float_or_floor(value: Any) -> float:
    try:
        if value in ("", None, "None"):
            return -1.0
        return float(value)
    except Exception:
        return -1.0


def write_best_csv(source: Path, dest: Path) -> None:
    if not source.exists() or source.stat().st_size == 0:
        return
    with open(source, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (_float_or_floor(r.get("test_auc")), _float_or_floor(r.get("best_val_auc"))), reverse=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["run_id", "combo_id"])
        writer.writeheader()
        writer.writerows(rows)


def update_status(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_jobs(args: argparse.Namespace, run_id: str) -> List[GridJob]:
    ablation = AblationSpec(name="full", flags={}, overrides={})
    jobs: List[GridJob] = []
    for idx, overrides in enumerate(build_grid(), start=1):
        combo_id = f"g{idx:03d}"
        save_dir = Path("checkpoints") / run_id / combo_id
        log_dir = Path("logs") / run_id / combo_id
        if (not args.rerun_existing) and (save_dir / "test_results.json").exists():
            continue

        params: Dict[str, Any] = {
            "epochs": int(args.epochs),
            "early_stop_patience": int(args.early_stop_patience),
            "patience": int(args.patience),
            "num_workers": int(args.num_workers),
            "debug_graph_diag": True,
            "diag_batches": 1,
        }
        params.update(overrides)
        job = JobSpec(
            dataset="nips34",
            seed=int(args.seed),
            profile="grid",
            ablation=ablation,
            model_variant=f"nips34_grid_{combo_id}",
            save_dir=save_dir,
            log_dir=log_dir,
            params=params,
            cmd=[],
        )
        job.cmd = build_command(job, generate_diagnosis=bool(args.generate_diagnosis))
        jobs.append(GridJob(combo_id=combo_id, params=params, job=job))

    if args.limit_jobs and args.limit_jobs > 0:
        return jobs[: int(args.limit_jobs)]
    return jobs


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    run_id = args.run_id or f"nips34_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_csv, best_csv, status_json = result_paths(run_id)

    gpus = parse_gpu_ids(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs provided.")
    max_concurrent = args.max_concurrent or len(gpus)
    max_concurrent = calc_effective_max_concurrent(max_concurrent, gpus, args.max_per_gpu)
    jobs = build_jobs(args, run_id)

    print(f"Run ID: {run_id}")
    print(f"Jobs: {len(jobs)}")
    print(f"GPUs: {gpus} max_concurrent={max_concurrent} max_per_gpu={args.max_per_gpu}")
    print(f"Result CSV: {result_csv}")
    print(f"Best CSV:   {best_csv}")

    running: List[Tuple[subprocess.Popen, int, GridJob]] = []
    gpu_load: Dict[int, int] = {gid: 0 for gid in gpus}
    rr = 0
    idx = 0
    completed = 0
    launched = 0
    poll_interval = max(2, int(args.poll_interval))

    while idx < len(jobs) or running:
        alive: List[Tuple[subprocess.Popen, int, GridJob]] = []
        for proc, gpu, grid_job in running:
            ret = proc.poll()
            if ret is None:
                alive.append((proc, gpu, grid_job))
                continue

            gpu_load[gpu] = max(0, gpu_load.get(gpu, 0) - 1)
            row = collect_result(grid_job.job, ret)
            row["run_id"] = run_id
            row["combo_id"] = grid_job.combo_id
            for key in GRID_KEYS:
                row[key] = grid_job.params.get(key, "")
            append_grid_row(result_csv, row)
            write_best_csv(result_csv, best_csv)
            completed += 1
            print(
                f"[DONE] {grid_job.combo_id} gpu={gpu} exit={ret} "
                f"test_auc={row.get('test_auc')} best_val={row.get('best_val_auc')}"
            )
        running = alive

        while idx < len(jobs) and len(running) < max_concurrent:
            gpu_id, rr = pick_gpu_with_slot_round_robin(gpus, gpu_load, args.max_per_gpu, rr)
            if gpu_id is None:
                break
            grid_job = jobs[idx]
            print(f"[PLAN] {grid_job.combo_id} gpu={gpu_id} params={grid_job.params}")
            print("       CMD:", " ".join(grid_job.job.cmd))
            if not args.dry_run:
                env = _build_job_env(os.environ.copy(), gpu_id)
                proc = subprocess.Popen(grid_job.job.cmd, env=env)
                running.append((proc, gpu_id, grid_job))
                gpu_load[gpu_id] = gpu_load.get(gpu_id, 0) + 1
                launched += 1
            idx += 1

        update_status(
            status_json,
            {
                "run_id": run_id,
                "total_jobs": len(jobs),
                "completed_jobs": completed,
                "running_jobs": len(running),
                "launched_jobs": launched,
                "gpus": gpus,
            },
        )
        if args.dry_run:
            break
        if running:
            time.sleep(poll_interval)
        elif idx < len(jobs):
            time.sleep(1)

    write_best_csv(result_csv, best_csv)
    update_status(
        status_json,
        {
            "run_id": run_id,
            "total_jobs": len(jobs),
            "completed_jobs": completed,
            "running_jobs": len(running),
            "launched_jobs": launched,
            "gpus": gpus,
        },
    )
    print(f"[DONE] completed={completed} launched={launched} total={len(jobs)}")


if __name__ == "__main__":
    main()
