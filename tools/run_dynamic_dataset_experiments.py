#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run dataset mechanism jobs with dynamic idle-GPU detection.

This launcher is intentionally explicit and auditable.  It uses the existing
mechanism runner to build full/ablation jobs, starts on the initial GPU list,
and opportunistically adds watched GPUs once they become idle.
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
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_utils import parse_gpu_ids  # noqa: E402
from run_abce_ablation import _build_job_env, collect_result  # noqa: E402
from tools.run_mechanism_experiments import (  # noqa: E402
    _supplement_row,
    append_rows,
    apply_cpu_thread_limits,
    build_jobs,
    run_postprocess,
    update_status,
)


DATASET_ALIASES = {
    "17": "assist_17",
    "assist17": "assist_17",
    "assist_17": "assist_17",
    "junyi": "junyi",
    "frcsub": "frcsub",
    "nips34": "nips34",
    "ednet_ktl": "ednet_kt1_clean15_sample5000",
    "ednet_kt1": "ednet_kt1_clean15_sample5000",
    "ednet_kt1_clean15_sample5000": "ednet_kt1_clean15_sample5000",
    "ednet_gap": "ednet_kt1_gap",
    "ednet_kt1_gap": "ednet_kt1_gap",
    "assist12": "assist_12_clean15_item50",
    "assist_12": "assist_12_clean15_item50",
    "assist 12": "assist_12_clean15_item50",
    "assist_12_clean15_item50": "assist_12_clean15_item50",
}

DEFAULT_DATASETS = (
    "junyi",
    "assist_17",
    "frcsub",
    "nips34",
    "ednet_kt1_clean15_sample5000",
    "ednet_kt1_gap",
    "assist_12_clean15_item50",
)

DEFAULT_VARIANTS = (
    "full",
    "no_A",
    "no_E",
    "E_shuffle_student",
    "A_fused_neutralE",
    "no_A_fair",
)


@dataclass
class RunningJob:
    proc: subprocess.Popen
    gpu: int
    desc: str
    kind: str
    payload: Any


@dataclass
class AnalysisJob:
    desc: str
    cmd: List[str]
    log_file: Path
    status_row: Dict[str, Any]


def _parse_dataset_tokens(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace("，", ",").split(","):
        token = raw.strip()
        if not token:
            continue
        key = token.lower().replace("-", "_")
        mapped = DATASET_ALIASES.get(key, key)
        if mapped not in out:
            out.append(mapped)
    return out


def _csv_tokens(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _query_gpu_stats() -> Dict[int, Dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        text = subprocess.check_output(cmd, encoding="utf-8", stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"[GPU] failed to query nvidia-smi: {exc}")
        return {}
    stats: Dict[int, Dict[str, int]] = {}
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gid = int(parts[0])
        used = int(parts[1])
        total = int(parts[2])
        util = int(parts[3])
        stats[gid] = {"used": used, "total": total, "free": total - used, "util": util}
    return stats


def _query_gpu_processes() -> Dict[int, List[Dict[str, Any]]]:
    """Return compute processes grouped by GPU index.

    Watched GPUs are only meant to be used when they are actually empty.  GPU
    utilization is sampled and can briefly read as low while another user's job
    is still resident, so process occupancy is the safer guard.
    """
    try:
        gpu_text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        app_text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"[GPU] failed to query compute processes: {exc}")
        return {}

    uuid_to_gid: Dict[str, int] = {}
    for line in gpu_text.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        uuid_to_gid[parts[1]] = int(parts[0])

    processes: Dict[int, List[Dict[str, Any]]] = {gid: [] for gid in uuid_to_gid.values()}
    for line in app_text.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gid = uuid_to_gid.get(parts[0])
        if gid is None:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            pid = -1
        try:
            used_memory = int(parts[3])
        except ValueError:
            used_memory = 0
        processes.setdefault(gid, []).append(
            {
                "pid": pid,
                "process_name": parts[2],
                "used_memory": used_memory,
            }
        )
    return processes


def _gpu_can_accept(
    *,
    gid: int,
    stats: Dict[int, Dict[str, int]],
    processes: Dict[int, List[Dict[str, Any]]],
    initial_gpus: Sequence[int],
    gpu_load: Dict[int, int],
    max_per_gpu: int,
    idle_util: int,
    min_free_mem_mb: int,
) -> bool:
    if gpu_load.get(gid, 0) >= max(1, int(max_per_gpu)):
        return False
    stat = stats.get(gid)
    if not stat:
        return gid in initial_gpus
    if stat["free"] < int(min_free_mem_mb):
        return False
    if gid in initial_gpus:
        return True
    if processes.get(gid):
        return False
    return stat["util"] <= int(idle_util)


def _available_gpus(
    *,
    initial_gpus: Sequence[int],
    watch_gpus: Sequence[int],
    gpu_load: Dict[int, int],
    max_per_gpu: int,
    idle_util: int,
    min_free_mem_mb: int,
) -> List[int]:
    stats = _query_gpu_stats()
    processes = _query_gpu_processes()
    candidates: List[int] = []
    for gid in [*initial_gpus, *watch_gpus]:
        if gid in candidates:
            continue
        if _gpu_can_accept(
            gid=gid,
            stats=stats,
            processes=processes,
            initial_gpus=initial_gpus,
            gpu_load=gpu_load,
            max_per_gpu=max_per_gpu,
            idle_util=idle_util,
            min_free_mem_mb=min_free_mem_mb,
        ):
            candidates.append(gid)
    if stats:
        status = " | ".join(
            (
                f"{gid}: util={stats[gid]['util']}%, free={stats[gid]['free']}MiB, "
                f"procs={len(processes.get(gid, []))}, own={gpu_load.get(gid, 0)}"
            )
            for gid in sorted(set([*initial_gpus, *watch_gpus]))
            if gid in stats
        )
        print(f"[GPU] {status}")
    return candidates


def _pick_gpu(candidates: Sequence[int], gpu_load: Dict[int, int], rr: int) -> Tuple[Optional[int], int]:
    if not candidates:
        return None, rr
    n = len(candidates)
    for offset in range(n):
        idx = (rr + offset) % n
        gid = int(candidates[idx])
        if gpu_load.get(gid, 0) <= 0:
            return gid, (idx + 1) % n
    return None, rr


def _write_analysis_status(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    old: List[Dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            old = list(csv.DictReader(f))
    keys = set()
    for row in [*old, *rows]:
        keys.update(row.keys())
    fieldnames = [
        "dataset",
        "analysis",
        "status",
        "exit_code",
        "gpu",
        "log_file",
        "output_dir",
        "started_at",
        "finished_at",
        *sorted(keys - {"dataset", "analysis", "status", "exit_code", "gpu", "log_file", "output_dir", "started_at", "finished_at"}),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old)
        writer.writerows(rows)


def _build_mechanism_args(args: argparse.Namespace, run_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        phase="phase2",
        datasets=",".join(args.datasets),
        variants=",".join(args.variants),
        gpus=",".join(str(g) for g in [*args.initial_gpus, *args.watch_gpus]),
        max_concurrent=len(set([*args.initial_gpus, *args.watch_gpus])),
        max_per_gpu=1,
        poll_interval=args.poll_interval,
        seed=args.seed,
        num_workers=args.num_workers,
        cpu_threads_per_job=args.cpu_threads_per_job,
        phase1_epochs=6,
        phase1_max_train_batches=120,
        phase1_max_val_batches=60,
        phase1_max_test_batches=60,
        phase2_epochs=args.epochs,
        phase2_patience=args.patience,
        limit_jobs=args.limit_jobs,
        dry_run=args.dry_run,
        rerun_existing=args.rerun_existing,
        generate_diagnosis=args.generate_diagnosis,
        skip_r=args.skip_r,
    )


def _existing_checkpoint(path: Path) -> bool:
    return (path / "best_model.pth").exists()


def _analysis_jobs_for_dataset(args: argparse.Namespace, run_id: str, dataset: str) -> List[AnalysisJob]:
    base = Path("checkpoints") / "mechanism" / run_id / "phase2" / dataset
    full = base / "full"
    no_a = base / "no_A"
    no_e = base / "no_E"
    a_fused = base / "A_fused_neutralE"
    no_a_fair = base / "no_A_fair"
    out_root = Path("results") / run_id / "small_experiments" / dataset
    log_root = Path("logs") / "mechanism" / run_id / "analysis" / dataset
    jobs: List[AnalysisJob] = []

    if _existing_checkpoint(a_fused) and _existing_checkpoint(no_a_fair):
        a_out = out_root / "a_support_evidence"
        jobs.append(
            AnalysisJob(
                desc=f"{dataset}/a_support_evidence",
                cmd=[
                    sys.executable,
                    "tools/analyze_a_support_evidence.py",
                    "--dataset_name",
                    dataset,
                    "--output_dir",
                    str(a_out),
                    "--variant_dir",
                    f"A_fused={a_fused}",
                    "--variant_dir",
                    f"no_A={no_a_fair}",
                    "--device",
                    "cuda",
                    "--batch_size",
                    str(args.analysis_batch_size),
                    "--num_workers",
                    str(args.analysis_num_workers),
                ],
                log_file=log_root / "a_support_evidence.log",
                status_row={"dataset": dataset, "analysis": "a_support_evidence", "output_dir": str(a_out)},
            )
        )
        a_corrupt_out = out_root / "a_support_corruption"
        jobs.append(
            AnalysisJob(
                desc=f"{dataset}/a_support_corruption",
                cmd=[
                    sys.executable,
                    "tools/analyze_a_support_corruption.py",
                    "--dataset_name",
                    dataset,
                    "--output_dir",
                    str(a_corrupt_out),
                    "--full_save_dir",
                    str(a_fused),
                    "--device",
                    "cuda",
                    "--batch_size",
                    str(args.analysis_batch_size),
                    "--num_workers",
                    str(args.analysis_num_workers),
                    "--corruption_fracs",
                    "0",
                    "0.25",
                    "0.50",
                    "0.75",
                    "1.0",
                    "--random_trials",
                    "5",
                ],
                log_file=log_root / "a_support_corruption.log",
                status_row={"dataset": dataset, "analysis": "a_support_corruption", "output_dir": str(a_corrupt_out)},
            )
        )
    else:
        print(f"[ANALYSIS-SKIP] {dataset}: missing A_fused_neutralE or no_A_fair checkpoint")

    if _existing_checkpoint(full) and _existing_checkpoint(no_a) and _existing_checkpoint(no_e):
        case_out = out_root / "ae_case_studies"
        jobs.append(
            AnalysisJob(
                desc=f"{dataset}/ae_case_studies",
                cmd=[
                    sys.executable,
                    "tools/export_ae_case_studies.py",
                    "--dataset_name",
                    dataset,
                    "--full_dir",
                    str(full),
                    "--no_a_dir",
                    str(no_a),
                    "--no_e_dir",
                    str(no_e),
                    "--output_dir",
                    str(case_out),
                    "--top_cases",
                    str(args.case_top_cases),
                    "--candidate_pool",
                    str(args.case_candidate_pool),
                    "--batch_size",
                    str(args.analysis_batch_size),
                    "--num_workers",
                    str(args.analysis_num_workers),
                    "--device",
                    "cuda",
                    "--e_counterfactuals",
                ],
                log_file=log_root / "ae_case_studies.log",
                status_row={"dataset": dataset, "analysis": "ae_case_studies", "output_dir": str(case_out)},
            )
        )
    else:
        print(f"[ANALYSIS-SKIP] {dataset}: missing full/no_A/no_E checkpoint")
    return jobs


def _plot_jobs_for_dataset(args: argparse.Namespace, run_id: str, dataset: str) -> List[AnalysisJob]:
    out_root = Path("results") / run_id / "small_experiments" / dataset
    log_root = Path("logs") / "mechanism" / run_id / "analysis" / dataset
    jobs: List[AnalysisJob] = []
    specs = [
        (
            "a_support_evidence_plot",
            out_root / "a_support_evidence",
            [
                "Rscript",
                "tools/plot_a_support_evidence.R",
                str(out_root / "a_support_evidence"),
                str(out_root / "a_support_evidence" / "figures"),
            ],
        ),
        (
            "a_support_corruption_plot",
            out_root / "a_support_corruption",
            [
                "Rscript",
                "tools/plot_a_support_corruption.R",
                str(out_root / "a_support_corruption"),
                str(out_root / "a_support_corruption" / "figures"),
            ],
        ),
        (
            "ae_case_studies_plot",
            out_root / "ae_case_studies",
            [
                "Rscript",
                "tools/plot_ae_case_studies.R",
                str(out_root / "ae_case_studies"),
                str(out_root / "ae_case_studies" / "figures"),
            ],
        ),
    ]
    for name, in_dir, cmd in specs:
        if in_dir.exists():
            jobs.append(
                AnalysisJob(
                    desc=f"{dataset}/{name}",
                    cmd=cmd,
                    log_file=log_root / f"{name}.log",
                    status_row={"dataset": dataset, "analysis": name, "output_dir": str(in_dir)},
                )
            )
    return jobs


def _launch_analysis(job: AnalysisJob, gpu_id: int, args: argparse.Namespace) -> subprocess.Popen:
    job.log_file.parent.mkdir(parents=True, exist_ok=True)
    env = _build_job_env(os.environ.copy(), gpu_id)
    env = apply_cpu_thread_limits(env, args.cpu_threads_per_job)
    print(f"[PLAN] analysis {job.desc} gpu={gpu_id}")
    print("       CMD:", " ".join(job.cmd))
    log_f = job.log_file.open("w", encoding="utf-8")
    return subprocess.Popen(job.cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)


def _run_queue(
    *,
    args: argparse.Namespace,
    run_id: str,
    train_jobs: Sequence[Any],
    result_csv: Path,
    status_json: Path,
) -> None:
    gpu_load: Dict[int, int] = {gid: 0 for gid in set([*args.initial_gpus, *args.watch_gpus])}
    running: List[RunningJob] = []
    idx = 0
    rr = 0
    completed = 0
    launched = 0
    poll_interval = max(2, int(args.poll_interval))

    while idx < len(train_jobs) or running:
        alive: List[RunningJob] = []
        completed_rows: List[Dict[str, Any]] = []
        for item in running:
            ret = item.proc.poll()
            if ret is None:
                alive.append(item)
                continue
            gpu_load[item.gpu] = max(0, gpu_load.get(item.gpu, 0) - 1)
            row = collect_result(item.payload.job, ret)
            row = _supplement_row(row, item.payload, run_id)
            completed_rows.append(row)
            completed += 1
            print(
                f"[DONE] {item.desc} gpu={item.gpu} exit={ret} "
                f"test_auc={row.get('test_auc')} best_val={row.get('best_val_auc')}"
            )
        running = alive
        append_rows(result_csv, completed_rows)

        while idx < len(train_jobs):
            candidates = _available_gpus(
                initial_gpus=args.initial_gpus,
                watch_gpus=args.watch_gpus,
                gpu_load=gpu_load,
                max_per_gpu=1,
                idle_util=args.idle_util,
                min_free_mem_mb=args.min_free_mem_mb,
            )
            gpu_id, rr = _pick_gpu(candidates, gpu_load, rr)
            if gpu_id is None:
                break
            mech_job = train_jobs[idx]
            print(f"[PLAN] train {mech_job.phase}/{mech_job.job.dataset}/{mech_job.variant} gpu={gpu_id}")
            print("       CMD:", " ".join(mech_job.job.cmd))
            env = _build_job_env(os.environ.copy(), gpu_id)
            env = apply_cpu_thread_limits(env, args.cpu_threads_per_job)
            proc = subprocess.Popen(mech_job.job.cmd, env=env)
            running.append(RunningJob(proc=proc, gpu=gpu_id, desc=f"{mech_job.phase}/{mech_job.job.dataset}/{mech_job.variant}", kind="train", payload=mech_job))
            gpu_load[gpu_id] = gpu_load.get(gpu_id, 0) + 1
            launched += 1
            idx += 1

        update_status(
            status_json,
            {
                "run_id": run_id,
                "stage": "training",
                "total_jobs": len(train_jobs),
                "completed_jobs": completed,
                "running_jobs": len(running),
                "launched_jobs": launched,
                "initial_gpus": args.initial_gpus,
                "watch_gpus": args.watch_gpus,
                "result_csv": str(result_csv),
            },
        )
        if running:
            time.sleep(poll_interval)
        elif idx < len(train_jobs):
            time.sleep(poll_interval)


def _run_analysis_queue(
    *,
    args: argparse.Namespace,
    run_id: str,
    jobs: Sequence[AnalysisJob],
    status_csv: Path,
    status_json: Path,
    stage: str,
) -> None:
    gpu_load: Dict[int, int] = {gid: 0 for gid in set([*args.initial_gpus, *args.watch_gpus])}
    running: List[RunningJob] = []
    idx = 0
    rr = 0
    completed = 0
    launched = 0
    poll_interval = max(2, int(args.poll_interval))
    while idx < len(jobs) or running:
        alive: List[RunningJob] = []
        rows: List[Dict[str, Any]] = []
        for item in running:
            ret = item.proc.poll()
            if ret is None:
                alive.append(item)
                continue
            gpu_load[item.gpu] = max(0, gpu_load.get(item.gpu, 0) - 1)
            payload: AnalysisJob = item.payload
            row = dict(payload.status_row)
            row.update(
                {
                    "status": "ok" if ret == 0 else "failed",
                    "exit_code": int(ret),
                    "gpu": item.gpu,
                    "log_file": str(payload.log_file),
                    "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            rows.append(row)
            completed += 1
            print(f"[DONE] analysis {payload.desc} gpu={item.gpu} exit={ret}")
        running = alive
        _write_analysis_status(status_csv, rows)

        while idx < len(jobs):
            candidates = _available_gpus(
                initial_gpus=args.initial_gpus,
                watch_gpus=args.watch_gpus,
                gpu_load=gpu_load,
                max_per_gpu=1,
                idle_util=args.idle_util,
                min_free_mem_mb=args.min_free_mem_mb,
            )
            gpu_id, rr = _pick_gpu(candidates, gpu_load, rr)
            if gpu_id is None:
                break
            job = jobs[idx]
            row = dict(job.status_row)
            row.update(
                {
                    "status": "running",
                    "exit_code": "",
                    "gpu": gpu_id,
                    "log_file": str(job.log_file),
                    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            _write_analysis_status(status_csv, [row])
            proc = _launch_analysis(job, gpu_id, args)
            running.append(RunningJob(proc=proc, gpu=gpu_id, desc=job.desc, kind="analysis", payload=job))
            gpu_load[gpu_id] = gpu_load.get(gpu_id, 0) + 1
            launched += 1
            idx += 1

        update_status(
            status_json,
            {
                "run_id": run_id,
                "stage": stage,
                "total_jobs": len(jobs),
                "completed_jobs": completed,
                "running_jobs": len(running),
                "launched_jobs": launched,
                "analysis_status_csv": str(status_csv),
            },
        )
        if running:
            time.sleep(poll_interval)
        elif idx < len(jobs):
            time.sleep(poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default="", help="Default: dataset_mechanism_<timestamp>")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--initial_gpus", default="2", help="GPUs allowed immediately.")
    parser.add_argument("--watch_gpus", default="0,1,3", help="GPUs used only when idle.")
    parser.add_argument("--idle_util", type=int, default=10)
    parser.add_argument("--min_free_mem_mb", type=int, default=8000)
    parser.add_argument("--poll_interval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu_threads_per_job", type=int, default=2)
    parser.add_argument("--limit_jobs", type=int, default=0)
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--generate_diagnosis", action="store_true")
    parser.add_argument("--skip_r", action="store_true")
    parser.add_argument("--skip_small_experiments", action="store_true")
    parser.add_argument("--analysis_batch_size", type=int, default=2048)
    parser.add_argument("--analysis_num_workers", type=int, default=2)
    parser.add_argument("--case_top_cases", type=int, default=3)
    parser.add_argument("--case_candidate_pool", type=int, default=40)
    parser.add_argument("--dry_run", action="store_true")
    ns = parser.parse_args()
    ns.datasets = _parse_dataset_tokens(ns.datasets)
    ns.variants = _csv_tokens(ns.variants)
    ns.initial_gpus = parse_gpu_ids(ns.initial_gpus)
    ns.watch_gpus = parse_gpu_ids(ns.watch_gpus)
    if not ns.initial_gpus and not ns.watch_gpus:
        raise SystemExit("No GPUs selected.")
    return ns


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    run_id = args.run_id or f"dataset_mechanism_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_csv = Path("results") / run_id / "mechanism_results.csv"
    status_json = Path("results") / run_id / "dynamic_status.json"
    analysis_status_csv = Path("results") / run_id / "small_experiments_status.csv"

    mech_args = _build_mechanism_args(args, run_id)
    jobs = build_jobs(mech_args, run_id)
    print(f"Run ID: {run_id}")
    print(f"Datasets: {args.datasets}")
    print(f"Variants: {args.variants}")
    print(f"Training jobs: {len(jobs)}")
    print(f"Initial GPUs: {args.initial_gpus}; watch GPUs: {args.watch_gpus}")
    print(f"Idle condition for watched GPUs: util <= {args.idle_util}%, free >= {args.min_free_mem_mb} MiB")
    print(f"Result CSV: {result_csv}")
    if args.dry_run:
        for item in jobs:
            print(f"[DRY] train {item.phase}/{item.job.dataset}/{item.variant}: {' '.join(item.job.cmd)}")
        return

    _run_queue(args=args, run_id=run_id, train_jobs=jobs, result_csv=result_csv, status_json=status_json)
    run_postprocess(result_csv, mech_args)
    if args.skip_small_experiments:
        print("[DONE] training complete; small experiments skipped.")
        return

    analysis_jobs: List[AnalysisJob] = []
    for dataset in args.datasets:
        analysis_jobs.extend(_analysis_jobs_for_dataset(args, run_id, dataset))
    _run_analysis_queue(
        args=args,
        run_id=run_id,
        jobs=analysis_jobs,
        status_csv=analysis_status_csv,
        status_json=status_json,
        stage="small_experiments",
    )
    plot_jobs: List[AnalysisJob] = []
    for dataset in args.datasets:
        plot_jobs.extend(_plot_jobs_for_dataset(args, run_id, dataset))
    _run_analysis_queue(
        args=args,
        run_id=run_id,
        jobs=plot_jobs,
        status_csv=analysis_status_csv,
        status_json=status_json,
        stage="small_experiment_plots",
    )
    update_status(
        status_json,
        {
            "run_id": run_id,
            "stage": "done",
            "result_csv": str(result_csv),
            "analysis_status_csv": str(analysis_status_csv),
        },
    )
    print(f"[DONE] run_id={run_id}")


if __name__ == "__main__":
    main()
