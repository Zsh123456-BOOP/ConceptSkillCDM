#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Minimal assist_09 full-grid runner for interpretable AE tuning."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from best_configs import BEST_CFG  # noqa: E402
from gpu_utils import calc_effective_max_concurrent, parse_gpu_ids, pick_gpu_with_slot_round_robin  # noqa: E402
from run_abce_ablation import AblationSpec, JobSpec, _build_job_env, build_command, collect_result  # noqa: E402

RESULT_CSV = Path("results") / "assist_min_grid_results.csv"
BEST_CSV = Path("results") / "assist_min_grid_best.csv"
STATUS_JSON = Path("results") / "assist_min_grid_status.json"

GRID_KEYS: Tuple[str, ...] = (
    "learning_rate",
    "dropout",
    "ae_logit_residual_scale",
    "ae_irt_logit_scale",
    "ae_lr_mult",
    "graph_query_readout_scale",
    "graph_query_readout_2hop_scale",
    "personal_query_correction_scale",
    "personal_query_correction_max_ratio",
    "lambda_personal_kl",
    "lambda_personal_query_residual",
    "relation_theta_scale",
    "concept_gap_scale",
)


@dataclass
class GridJob:
    combo_id: str
    params: Dict[str, Any]
    job: JobSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded assist_09 full-only minimal grid.")
    parser.add_argument("--run_id", default="", help="Run id; default uses timestamp.")
    parser.add_argument("--gpus", default="2,3", help="Physical GPU ids, comma separated.")
    parser.add_argument("--max_concurrent", type=int, default=0, help="Default: number of GPUs.")
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--time_limit_hours", type=float, default=10.0)
    parser.add_argument("--poll_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove logs/results/checkpoints before running.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit_jobs", type=int, default=0, help="Optional cap for smoke testing.")
    parser.add_argument("--no_generate_diagnosis", action="store_true")
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
    base = {key: BEST_CFG["assist_09"][key] for key in GRID_KEYS}
    candidates: List[Dict[str, Any]] = [dict(base)]

    for lr in (2e-4, 3e-4, 4e-4, 5e-4):
        for dropout in (0.15, 0.20, 0.25):
            item = dict(base)
            item.update(learning_rate=lr, dropout=dropout)
            candidates.append(item)

    for ae_scale in (0.80, 1.00, 1.20):
        for irt_scale in (0.15, 0.20, 0.30):
            for ae_lr_mult in (15.0, 30.0):
                item = dict(base)
                item.update(
                    ae_logit_residual_scale=ae_scale,
                    ae_irt_logit_scale=irt_scale,
                    ae_lr_mult=ae_lr_mult,
                )
                candidates.append(item)

    for graph_scale in (0.02, 0.04, 0.06):
        for personal_scale in (0.15, 0.20, 0.30):
            for max_ratio in (0.05, 0.08):
                item = dict(base)
                item.update(
                    graph_query_readout_scale=graph_scale,
                    graph_query_readout_2hop_scale=max(0.005, graph_scale * 0.5),
                    personal_query_correction_scale=personal_scale,
                    personal_query_correction_max_ratio=max_ratio,
                )
                candidates.append(item)

    for lr in (2e-4, 3e-4):
        for dropout in (0.15, 0.20):
            for ae_scale in (0.80, 1.00):
                for irt_scale in (0.20, 0.30):
                    item = dict(base)
                    item.update(
                        learning_rate=lr,
                        dropout=dropout,
                        ae_logit_residual_scale=ae_scale,
                        ae_irt_logit_scale=irt_scale,
                        graph_query_readout_scale=0.04,
                        graph_query_readout_2hop_scale=0.02,
                    )
                    candidates.append(item)

    for kl in (0.02, 0.04, 0.06):
        for query_reg in (0.04, 0.06, 0.08):
            item = dict(base)
            item.update(lambda_personal_kl=kl, lambda_personal_query_residual=query_reg)
            candidates.append(item)

    for relation_theta_scale in (-0.5, 0.0, 0.5):
        item = dict(base)
        item.update(relation_theta_scale=relation_theta_scale)
        candidates.append(item)

    for gap_scale in (0.05, 0.10, 0.20, 0.35):
        item = dict(base)
        item.update(concept_gap_scale=gap_scale, relation_theta_scale=0.0)
        candidates.append(item)

    return _dedupe(candidates)


def clean_outputs() -> None:
    for name in ("logs", "results", "checkpoints"):
        path = Path(name)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def count_grad_guard(log_file: str) -> int:
    if not log_file:
        return 0
    path = Path(log_file)
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Grad Guard" in line:
                count += 1
    return count


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
        "grad_guard_count",
        "log_file",
        *GRID_KEYS,
        "params_json",
        "save_dir",
        "log_dir",
    ]
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_best_csv(source: Path, dest: Path) -> None:
    if not source.exists():
        return
    with open(source, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(
        key=lambda r: (
            float(r["test_auc"]) if str(r.get("test_auc") or "").strip() else -1.0,
            float(r["best_val_auc"]) if str(r.get("best_val_auc") or "").strip() else -1.0,
        ),
        reverse=True,
    )
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["run_id", "combo_id"])
        writer.writeheader()
        writer.writerows(rows)


def build_jobs(run_id: str, seed: int, limit_jobs: int = 0) -> List[GridJob]:
    base_cfg = dict(BEST_CFG["assist_09"])
    ablation = AblationSpec(name="full", flags={}, overrides={})
    jobs: List[GridJob] = []
    for idx, overrides in enumerate(build_grid(), start=1):
        params = dict(base_cfg)
        params.update(overrides)
        params["debug_graph_diag"] = True
        params["diag_batches"] = 1
        combo_id = f"g{idx:03d}"
        model_variant = f"assist_09_grid_{combo_id}"
        save_dir = Path("checkpoints") / "assist_min_grid" / run_id / combo_id
        log_dir = Path("logs") / "assist_min_grid" / run_id / combo_id
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        job = JobSpec(
            dataset="assist_09",
            seed=int(seed),
            profile="grid",
            ablation=ablation,
            model_variant=model_variant,
            save_dir=save_dir,
            log_dir=log_dir,
            params=params,
            cmd=[],
        )
        job.cmd = build_command(job, generate_diagnosis=False)
        jobs.append(GridJob(combo_id=combo_id, params=overrides, job=job))
    return jobs[:limit_jobs] if limit_jobs and limit_jobs > 0 else jobs


def update_status(run_id: str, total: int, completed: int, running: int, launched: int) -> None:
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "total_jobs": total,
                "completed_jobs": completed,
                "running_jobs": running,
                "launched_jobs": launched,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    run_id = args.run_id or f"assist_min_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.clean:
        clean_outputs()
    Path("results").mkdir(parents=True, exist_ok=True)

    gpus = parse_gpu_ids(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs provided.")
    max_concurrent = args.max_concurrent or len(gpus)
    max_concurrent = calc_effective_max_concurrent(max_concurrent, gpus, args.max_per_gpu)
    jobs = build_jobs(run_id, seed=args.seed, limit_jobs=args.limit_jobs)
    deadline = time.time() + max(0.01, float(args.time_limit_hours)) * 3600.0

    print(f"Run ID: {run_id}")
    print(f"Jobs: {len(jobs)}")
    print(f"GPUs: {gpus} max_concurrent={max_concurrent} max_per_gpu={args.max_per_gpu}")
    print(f"Result CSV: {RESULT_CSV}")
    print(f"Best CSV:   {BEST_CSV}")

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
            row["grad_guard_count"] = count_grad_guard(str(row.get("log_file") or ""))
            for key in GRID_KEYS:
                row[key] = grid_job.params.get(key, grid_job.job.params.get(key))
            append_grid_row(RESULT_CSV, row)
            write_best_csv(RESULT_CSV, BEST_CSV)
            completed += 1
            print(
                f"[DONE] {grid_job.combo_id} gpu={gpu} exit={ret} "
                f"test_auc={row.get('test_auc')} best_val={row.get('best_val_auc')} "
                f"grad_guard={row.get('grad_guard_count')}"
            )
        running = alive

        can_launch = time.time() < deadline
        while can_launch and idx < len(jobs) and len(running) < max_concurrent:
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
            can_launch = time.time() < deadline

        update_status(run_id, len(jobs), completed, len(running), launched)
        if args.dry_run:
            break
        if running:
            time.sleep(poll_interval)
        elif idx < len(jobs) and not can_launch:
            print("[TIME-LIMIT] No new jobs launched after deadline.")
            break
        elif idx < len(jobs):
            time.sleep(1)

    write_best_csv(RESULT_CSV, BEST_CSV)
    update_status(run_id, len(jobs), completed, len(running), launched)
    print(f"[DONE] completed={completed} launched={launched} total={len(jobs)}")


if __name__ == "__main__":
    main()
