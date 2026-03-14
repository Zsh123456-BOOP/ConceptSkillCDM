#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_m13_diagnosis.py

Purpose:
1) Run targeted ablations for Module-1 / Module-3:
   - full
   - no_module1
   - no_module3
2) Collect richer diagnostics from logs:
   - graph_entropy_ratio / alpha_std
   - gate_mean / delta_over_irt / mf_abs_mean / irt_abs_mean
   - warning counters (graph-uniform, alpha-collapse, module3 warnings)
   - module activity snapshot
3) Write machine-readable outputs:
   - results/m13_ablation_diagnosis.csv
   - results/m13_ablation_summary.csv

Example:
python run_m13_diagnosis.py --datasets assist_09,junyi --gpus 0 --max_concurrent 1 --max_per_gpu 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from best_configs import BEST_CFG, DEFAULT_SEEDS
from gpu_utils import (
    calc_effective_max_concurrent,
    parse_gpu_ids,
    parse_int_csv,
    pick_gpu_with_slot_round_robin,
)


RESULT_CSV = Path("results") / "m13_ablation_diagnosis.csv"
SUMMARY_CSV = Path("results") / "m13_ablation_summary.csv"
NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


ABLATIONS: List[Dict[str, Any]] = [
    {"name": "full", "flags": {}},
    {"name": "no_module1", "flags": {"ablate_module1": True}},
    {"name": "no_module3", "flags": {"ablate_module3": True}},
]


@dataclass
class JobSpec:
    dataset: str
    seed: int
    profile: str
    ablation: str
    model_variant: str
    save_dir: Path
    log_dir: Path
    params: Dict[str, Any]
    flags: Dict[str, Any]
    cmd: List[str]


def parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def append_arg(cmd: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{key}")
        return
    cmd.extend([f"--{key}", str(value)])


def profile_overrides(dataset: str, profile: str) -> Dict[str, Any]:
    if profile == "baseline":
        return {}
    if profile == "m1_rescue":
        return {
            "graph_topk": 16 if dataset == "assist_09" else 32,
            "graph_tau_init": 0.5,
            "graph_dropout": 0.0,
            "graph_reg_warmup_epochs": 0,
            "lambda_sparse": 0.3,
        }
    if profile == "m3_rescue":
        out = {
            "fusion_gate_bias_init": -0.3,
            "residual_scale_init": 0.3,
            "residual_clip_t": 3.0,
            "exercise_l2_lambda": 1e-5,
        }
        if dataset == "junyi":
            out["dropout"] = 0.3
        return out
    raise ValueError(f"Unknown profile '{profile}'.")


def build_command(job: JobSpec) -> List[str]:
    cmd = [
        sys.executable,
        "main.py",
        "--dataset_name",
        job.dataset,
        "--model_variant",
        job.model_variant,
        "--save_dir",
        str(job.save_dir),
        "--log_dir",
        str(job.log_dir),
        "--seed",
        str(job.seed),
        "--generate_diagnosis",
        "False",
    ]

    skip_keys = {
        "dataset_name",
        "model_variant",
        "save_dir",
        "log_dir",
        "seed",
        "generate_diagnosis",
    }
    for k in sorted(job.params.keys()):
        if k in skip_keys:
            continue
        if k.startswith("ablate_"):
            continue
        append_arg(cmd, k, job.params[k])

    for k, v in job.flags.items():
        append_arg(cmd, k, v)

    return cmd


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_train_log(log_dir: Path) -> Optional[Path]:
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def extract_float(line: str, key: str) -> Optional[float]:
    m = re.search(rf"{re.escape(key)}=(?P<v>{NUM_RE})", line)
    if not m:
        return None
    return float(m.group("v"))


def parse_log_metrics(log_file: Optional[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "reg_bce_ratio": None,
        "graph_entropy_ratio": None,
        "alpha_std": None,
        "gate_mean": None,
        "delta_over_irt": None,
        "mf_abs_mean": None,
        "irt_abs_mean": None,
        "warn_graph_uniform_count": 0,
        "warn_alpha_collapse_count": 0,
        "warn_module3_count": 0,
        "module_activity_epoch10": "",
    }
    if log_file is None or not log_file.exists():
        return out

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[Reg Terms]" in line:
                val_pos = line.rfind("| Val:")
                section = line[val_pos:] if val_pos >= 0 else line
                v = extract_float(section, "reg_bce_ratio")
                if v is not None:
                    out["reg_bce_ratio"] = v

            if "[Diag][M3] Epoch" in line:
                for k in (
                    "graph_entropy_ratio",
                    "alpha_std",
                    "gate_mean",
                    "delta_over_irt",
                    "mf_abs_mean",
                    "irt_abs_mean",
                ):
                    v = extract_float(line, k)
                    if v is not None:
                        out[k] = v

            if "[Diag Warning][Graph]" in line:
                out["warn_graph_uniform_count"] += 1
            if "alpha_std has been near zero" in line:
                out["warn_alpha_collapse_count"] += 1
            if "[Diag Warning][M3]" in line:
                out["warn_module3_count"] += 1

            if "[Module Activity] Epoch 10:" in line:
                out["module_activity_epoch10"] = line.strip()

    return out


def collect_result(job: JobSpec, exit_code: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": job.dataset,
        "seed": job.seed,
        "profile": job.profile,
        "ablation": job.ablation,
        "model_variant": job.model_variant,
        "status": "ok" if exit_code == 0 else "failed",
        "exit_code": int(exit_code),
        "save_dir": str(job.save_dir),
        "log_dir": str(job.log_dir),
    }

    test_json = read_json(job.save_dir / "test_results.json") or {}
    metrics = test_json.get("metrics", {}) if isinstance(test_json, dict) else {}
    row["test_auc"] = metrics.get("auc")
    row["test_acc"] = metrics.get("acc")
    row["test_rmse"] = metrics.get("rmse")
    row["best_val_auc"] = test_json.get("best_val_auc")
    row["model_epoch"] = test_json.get("model_epoch")

    history_json = read_json(job.save_dir / "training_history.json") or {}
    row["best_epoch"] = history_json.get("best_epoch") if isinstance(history_json, dict) else None

    log_file = latest_train_log(job.log_dir)
    row["log_file"] = str(log_file) if log_file else ""
    row.update(parse_log_metrics(log_file))

    row["params_json"] = json.dumps(job.params, ensure_ascii=False, sort_keys=True)
    row["flags_json"] = json.dumps(job.flags, ensure_ascii=False, sort_keys=True)
    return row


def append_result_row(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "dataset",
        "seed",
        "profile",
        "ablation",
        "model_variant",
        "test_auc",
        "test_acc",
        "test_rmse",
        "best_val_auc",
        "best_epoch",
        "model_epoch",
        "reg_bce_ratio",
        "graph_entropy_ratio",
        "alpha_std",
        "gate_mean",
        "delta_over_irt",
        "mf_abs_mean",
        "irt_abs_mean",
        "warn_graph_uniform_count",
        "warn_alpha_collapse_count",
        "warn_module3_count",
        "module_activity_epoch10",
        "status",
        "exit_code",
        "save_dir",
        "log_dir",
        "log_file",
        "params_json",
        "flags_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_result_rows(path: Path, run_id: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for row in csv.DictReader(f):
            if run_id and run_id not in row.get("save_dir", ""):
                continue
            rows.append(row)
    return rows


def try_float(x: Any) -> Optional[float]:
    if x in (None, "", "None"):
        return None
    try:
        return float(x)
    except Exception:
        return None


def diagnose_reason(full_row: Dict[str, Any], delta_m1: Optional[float], delta_m3: Optional[float]) -> str:
    reasons: List[str] = []
    ger = try_float(full_row.get("graph_entropy_ratio"))
    alpha_std = try_float(full_row.get("alpha_std"))
    dor = try_float(full_row.get("delta_over_irt"))
    gate = try_float(full_row.get("gate_mean"))

    if ger is not None and ger > 0.98:
        reasons.append("M1-global-graph-uniform")
    if alpha_std is not None and alpha_std < 1e-6:
        reasons.append("M1-personal-graph-collapsed")
    if delta_m1 is not None and abs(delta_m1) < 0.002:
        reasons.append("M1-ablation-delta-small")

    if delta_m3 is not None:
        if delta_m3 > 0.010:
            reasons.append("M3-strong-positive")
        elif delta_m3 < -0.001:
            reasons.append("M3-negative-transfer")
        else:
            reasons.append("M3-ablation-delta-small")

    if dor is not None and dor < 0.05:
        reasons.append("M3-delta-over-irt-low")
    if gate is not None and gate < 0.5:
        reasons.append("M3-gate-low")
    if gate is not None and gate > 0.9:
        reasons.append("M3-gate-very-high")

    return ";".join(reasons)


def write_summary(path: Path, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        if str(r.get("status", "")).lower() != "ok":
            continue
        key = (str(r.get("dataset", "")), str(r.get("seed", "")), str(r.get("profile", "")))
        grouped.setdefault(key, {})
        grouped[key][str(r.get("ablation", ""))] = r

    summary_rows: List[Dict[str, Any]] = []
    for (dataset, seed, profile), mp in sorted(grouped.items()):
        full = mp.get("full")
        no_m1 = mp.get("no_module1")
        no_m3 = mp.get("no_module3")
        if full is None:
            continue

        full_auc = try_float(full.get("test_auc"))
        no_m1_auc = try_float(no_m1.get("test_auc") if no_m1 else None)
        no_m3_auc = try_float(no_m3.get("test_auc") if no_m3 else None)
        delta_m1 = (full_auc - no_m1_auc) if (full_auc is not None and no_m1_auc is not None) else None
        delta_m3 = (full_auc - no_m3_auc) if (full_auc is not None and no_m3_auc is not None) else None

        row = {
            "dataset": dataset,
            "seed": seed,
            "profile": profile,
            "full_auc": full_auc,
            "no_module1_auc": no_m1_auc,
            "no_module3_auc": no_m3_auc,
            "delta_module1_full_minus_no1": delta_m1,
            "delta_module3_full_minus_no3": delta_m3,
            "full_graph_entropy_ratio": try_float(full.get("graph_entropy_ratio")),
            "full_alpha_std": try_float(full.get("alpha_std")),
            "full_gate_mean": try_float(full.get("gate_mean")),
            "full_delta_over_irt": try_float(full.get("delta_over_irt")),
            "full_warn_graph_uniform_count": full.get("warn_graph_uniform_count"),
            "full_warn_alpha_collapse_count": full.get("warn_alpha_collapse_count"),
            "full_warn_module3_count": full.get("warn_module3_count"),
            "full_module_activity_epoch10": full.get("module_activity_epoch10", ""),
            "diagnosis_reason": diagnose_reason(full, delta_m1, delta_m3),
        }
        summary_rows.append(row)

    fieldnames = [
        "dataset",
        "seed",
        "profile",
        "full_auc",
        "no_module1_auc",
        "no_module3_auc",
        "delta_module1_full_minus_no1",
        "delta_module3_full_minus_no3",
        "full_graph_entropy_ratio",
        "full_alpha_std",
        "full_gate_mean",
        "full_delta_over_irt",
        "full_warn_graph_uniform_count",
        "full_warn_alpha_collapse_count",
        "full_warn_module3_count",
        "full_module_activity_epoch10",
        "diagnosis_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Module1/3 targeted ablation with diagnosis summary.")
    parser.add_argument("--datasets", type=str, default="assist_09,junyi")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds. Default uses DEFAULT_SEEDS.")
    parser.add_argument("--profiles", type=str, default="baseline", help="Comma-separated: baseline,m1_rescue,m3_rescue")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_interval", type=int, default=10)
    parser.add_argument("--run_id", type=str, default=None, help="Output namespace. Default=timestamp.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--analyze_only", action="store_true", help="Skip running and only summarize existing rows.")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    return parser.parse_args()


def make_jobs(args: argparse.Namespace, run_id: str) -> List[JobSpec]:
    datasets = parse_csv_tokens(args.datasets)
    seeds = list(DEFAULT_SEEDS) if args.seeds is None else parse_int_csv(args.seeds)
    profiles = parse_csv_tokens(args.profiles)
    jobs: List[JobSpec] = []

    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not found in BEST_CFG.")
        base_cfg = dict(BEST_CFG[dataset])
        for profile in profiles:
            prof = profile_overrides(dataset, profile)
            for ab in ABLATIONS:
                for seed in seeds:
                    params = dict(base_cfg)
                    params.update(prof)
                    params["debug_module3_diag"] = True
                    params["diag_batches"] = 1

                    if args.epochs is not None:
                        params["epochs"] = int(args.epochs)
                    if args.early_stop_patience is not None:
                        params["early_stop_patience"] = int(args.early_stop_patience)
                    if args.learning_rate is not None:
                        params["learning_rate"] = float(args.learning_rate)

                    abl = ab["name"]
                    model_variant = f"{dataset}_m13diag_{profile}_{abl}"
                    save_dir = Path("checkpoints") / "m13_diag" / run_id / dataset / f"seed{seed}" / f"{profile}_{abl}"
                    log_dir = Path("logs") / "m13_diag" / run_id / dataset / f"seed{seed}" / f"{profile}_{abl}"

                    if (not args.rerun_existing) and (save_dir / "test_results.json").exists():
                        continue

                    save_dir.mkdir(parents=True, exist_ok=True)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    job = JobSpec(
                        dataset=dataset,
                        seed=int(seed),
                        profile=profile,
                        ablation=abl,
                        model_variant=model_variant,
                        save_dir=save_dir,
                        log_dir=log_dir,
                        params=params,
                        flags=dict(ab["flags"]),
                        cmd=[],
                    )
                    job.cmd = build_command(job)
                    jobs.append(job)
    return jobs


def print_job(job: JobSpec, gpu_id: int) -> None:
    print(
        f"[PLAN] dataset={job.dataset} seed={job.seed} profile={job.profile} "
        f"ablation={job.ablation} gpu={gpu_id}"
    )
    print("       CMD:", " ".join(job.cmd))


def run_jobs(args: argparse.Namespace, jobs: Sequence[JobSpec], run_id: str) -> None:
    gpus = parse_gpu_ids(args.gpus)
    if not gpus:
        raise ValueError("No GPUs provided.")
    max_concurrent = calc_effective_max_concurrent(args.max_concurrent, gpus, args.max_per_gpu)
    max_per_gpu = max(1, int(args.max_per_gpu))
    poll_interval = max(1, int(args.poll_interval))

    running: List[Tuple[subprocess.Popen, int, JobSpec]] = []
    gpu_load: Dict[int, int] = {gid: 0 for gid in gpus}
    rr = 0
    idx = 0

    while idx < len(jobs) or running:
        alive: List[Tuple[subprocess.Popen, int, JobSpec]] = []
        for proc, gpu, job in running:
            ret = proc.poll()
            if ret is None:
                alive.append((proc, gpu, job))
                continue

            gpu_load[gpu] = max(0, gpu_load.get(gpu, 0) - 1)
            row = collect_result(job, ret)
            append_result_row(RESULT_CSV, row)
            print(
                f"[DONE] dataset={job.dataset} seed={job.seed} profile={job.profile} "
                f"ablation={job.ablation} exit={ret} auc={row.get('test_auc')}"
            )
        running = alive

        while idx < len(jobs) and len(running) < max_concurrent:
            gpu_id, rr = pick_gpu_with_slot_round_robin(gpus, gpu_load, max_per_gpu, rr)
            if gpu_id is None:
                break
            job = jobs[idx]
            print_job(job, gpu_id)
            if args.dry_run:
                idx += 1
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.Popen(job.cmd, env=env)
            running.append((proc, gpu_id, job))
            gpu_load[gpu_id] = gpu_load.get(gpu_id, 0) + 1
            idx += 1

        if running:
            time.sleep(poll_interval)
        elif idx < len(jobs):
            time.sleep(1)

    if args.dry_run:
        print("[DRY-RUN] no process launched.")


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.analyze_only:
        rows = load_result_rows(RESULT_CSV, run_id=args.run_id or "")
        summary = write_summary(SUMMARY_CSV, rows)
        print(f"[ANALYZE-ONLY] rows={len(rows)}, summary_rows={len(summary)}")
        print(f"Diagnosis CSV: {RESULT_CSV}")
        print(f"Summary CSV:   {SUMMARY_CSV}")
        return

    jobs = make_jobs(args, run_id=run_id)
    print(f"Run ID: {run_id}")
    print(f"Jobs: {len(jobs)}")
    print(f"Diagnosis CSV: {RESULT_CSV}")
    print(f"Summary CSV:   {SUMMARY_CSV}")

    if jobs:
        run_jobs(args, jobs, run_id=run_id)
    else:
        print("[INFO] No new jobs to run (all checkpoints exist or filters excluded all jobs).")

    # Summarize rows for current run namespace.
    rows = load_result_rows(RESULT_CSV, run_id=run_id)
    summary = write_summary(SUMMARY_CSV, rows)
    print(f"[SUMMARY] rows={len(rows)}, summary_rows={len(summary)}")
    for r in summary:
        print(
            f"[SUMMARY] dataset={r['dataset']} seed={r['seed']} profile={r['profile']} "
            f"delta_m1={r['delta_module1_full_minus_no1']} "
            f"delta_m3={r['delta_module3_full_minus_no3']} "
            f"reason={r['diagnosis_reason']}"
        )


if __name__ == "__main__":
    main()
