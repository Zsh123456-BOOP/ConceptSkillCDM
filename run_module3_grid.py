#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run module3-focused grid search based on BEST_CFG.

Design goals:
1) Use BEST_CFG from `best_configs.py` as the base config for each dataset.
2) For each grid point, always run paired variants: `full` and `no_module3`.
3) Reuse multi-GPU scheduling with `max_concurrent` and `max_per_gpu` limits.
4) Write summary rows to `results/module3_grid_results.csv`.
5) Support `--dry_run` to print commands without executing.

Examples:
python run_module3_grid.py --datasets assist_09,junyi --seeds 42 --gpus 0 --max_concurrent 1 --max_per_gpu 1 --epochs 15
python run_module3_grid.py --datasets assist_09,junyi --seeds 42 --gpus 0,1,2,3 --max_concurrent 4 --max_per_gpu 1 --poll_interval 15
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from best_configs import BEST_CFG, DEFAULT_SEEDS
from gpu_utils import (
    calc_effective_max_concurrent,
    parse_gpu_ids,
    parse_int_csv,
    pick_gpu_with_slot_round_robin,
)

import main as main_entry


NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
RESULT_CSV = Path("results") / "module3_grid_results.csv"
ALLOWED_VARIANTS = {"full", "no_module3"}
VARIANT_ALIAS = {"full": "full", "no_module3": "no_module3", "no3": "no_module3"}


@dataclass
class GridPoint:
    tag: str
    overrides: Dict[str, Any]


@dataclass
class JobSpec:
    dataset: str
    seed: int
    grid_tag: str
    variant: str  # full | no_module3
    model_variant: str
    save_dir: Path
    log_dir: Path
    params: Dict[str, Any]
    cmd: List[str]


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run module3-focused grid based on BEST_CFG.")
    parser.add_argument("--datasets", type=str, default="assist_09,junyi")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds. Default uses DEFAULT_SEEDS.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Epochs for quick diagnosis (default=15). If not explicitly provided and BEST_CFG has epochs, BEST_CFG is used.",
    )
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids.")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_interval", type=int, default=15)
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    parser.add_argument("--ablation_set", type=str, default="model", help="Currently only supports model.")
    parser.add_argument(
        "--only_variants",
        type=str,
        default="full,no_module3",
        help="Comma-separated subset of variants. Allowed: full,no_module3(no3 alias).",
    )
    return parser.parse_args()


def epochs_was_explicitly_set(argv: Sequence[str]) -> bool:
    for token in argv:
        if token == "--epochs" or token.startswith("--epochs="):
            return True
    return False


def parse_csv_tokens(text: str) -> List[str]:
    return [tok.strip() for tok in str(text).split(",") if tok.strip()]


def normalize_variants(text: str) -> List[str]:
    out: List[str] = []
    for token in parse_csv_tokens(text):
        key = VARIANT_ALIAS.get(token.lower())
        if key is None:
            raise ValueError(f"Unknown variant '{token}'. Allowed: {sorted(ALLOWED_VARIANTS)}")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("No variants selected after parsing --only_variants.")
    return out


def get_main_arg_dests() -> Set[str]:
    parser = main_entry.parse_args()
    dests: Set[str] = set()
    for action in parser._actions:
        if action.dest and action.dest != "help":
            dests.add(action.dest)
    return dests


def append_arg(cmd: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{key}")
        return
    cmd.extend([f"--{key}", str(value)])


def build_grid_points(base_cfg: Dict[str, Any]) -> Tuple[List[GridPoint], List[str]]:
    # Direction-2: remove proto_relax style points by default (prototype stays off unless explicitly enabled).
    points: List[GridPoint] = [
        GridPoint("baseline", {}),
        GridPoint("sparse001", {"lambda_sparse": 0.01}),
        GridPoint("gate03_bm30", {"fusion_gate_max": 0.30, "fusion_gate_bias_init": -3.0}),
        GridPoint("clip15", {"residual_clip_t": 1.5}),
        GridPoint(
            "sparse_gate_clip",
            {"lambda_sparse": 0.01, "fusion_gate_max": 0.30, "fusion_gate_bias_init": -3.0, "residual_clip_t": 1.5},
        ),
    ]
    skip_msgs: List[str] = []

    base_dropout = float(base_cfg.get("dropout", 0.0) or 0.0)
    if base_dropout >= 0.2:
        points.append(GridPoint("drop02", {"dropout": 0.2}))
    else:
        skip_msgs.append(
            f"base dropout={base_dropout:.4f} < 0.2, skip drop02 point (no reverse increase)."
        )

    return points, skip_msgs


def build_shared_params(
    dataset: str,
    base_cfg: Dict[str, Any],
    grid_overrides: Dict[str, Any],
    epochs_override: Optional[int],
) -> Dict[str, Any]:
    params = dict(base_cfg)
    params.update(grid_overrides)

    if epochs_override is not None:
        params["epochs"] = int(epochs_override)
    elif params.get("epochs") is None:
        params["epochs"] = 15

    # Fail-fast: no_module3 with ablate_module2=True will become invalid (no prediction path).
    if bool(params.get("ablate_module2", False)):
        raise ValueError(
            f"[{dataset}] BEST_CFG has ablate_module2=True, incompatible with no_module3 pair runs."
        )

    # Enforce "full" semantics for module3 baseline while keeping prototype off by default.
    params["ablate_module3"] = False
    params["ablate_skill_encoder"] = False
    params["disable_q_aligned_residual"] = False

    # Prototype stays disabled unless user explicitly overrides in cfg/grid.
    params.setdefault("enable_soft_prototype", False)
    params["disable_soft_prototype"] = True
    params["ablate_soft_prototype"] = True
    params["use_soft_prototype_main_path"] = False
    params["num_prototypes"] = 0
    params["proto_lambda"] = float(params.get("proto_lambda", 0.0) or 0.0)
    params["lambda_proto_div"] = 0.0
    params["lambda_proto_usage"] = 0.0

    # Conservative fusion defaults.
    params.setdefault("fusion_gate_max", 0.4)
    params.setdefault("fusion_gate_bias_init", -2.5)
    params.setdefault("residual_clip_t", 2.0)

    return params


def build_variant_params(shared_params: Dict[str, Any], variant: str) -> Dict[str, Any]:
    params = dict(shared_params)
    if variant == "no_module3":
        params["ablate_module3"] = True
        # Remove explicit submodule toggles to avoid conflicts; ablate_module3 is the source of truth.
        params.pop("use_mf_branch", None)
        params.pop("use_soft_prototype", None)
    elif variant == "full":
        params["ablate_module3"] = False
    else:
        raise ValueError(f"Unsupported variant '{variant}'.")
    return params


def validate_params(dataset: str, params: Dict[str, Any], main_arg_dests: Set[str]) -> None:
    unknown = sorted(k for k in params.keys() if k not in main_arg_dests)
    if unknown:
        raise ValueError(f"[{dataset}] Unknown keys in BEST_CFG/grid params for main.py: {unknown}")
    if bool(params.get("ablate_module2", False)) and bool(params.get("ablate_module3", False)):
        raise ValueError(f"[{dataset}] Invalid params: ablate_module2=True and ablate_module3=True.")


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
        "--debug_module3_diag",
        "--diag_batches",
        "2",
    ]

    skip_keys = {"dataset_name", "model_variant", "save_dir", "log_dir", "seed", "generate_diagnosis", "gpus"}
    for key in sorted(job.params.keys()):
        if key in skip_keys:
            continue
        append_arg(cmd, key, job.params[key])
    return cmd


def extract_float(line: str, key: str) -> Optional[float]:
    m = re.search(rf"{re.escape(key)}=(?P<v>{NUM_RE})", line)
    if not m:
        return None
    return float(m.group("v"))


def latest_train_log(log_dir: Path) -> Optional[Path]:
    if not log_dir.exists():
        return None
    candidates = sorted(log_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def parse_diag_from_log(log_file: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "reg_bce_ratio": None,
        "delta_over_irt": None,
        "mf_abs_mean": None,
        "residual_abs_mean": None,
        "gate_mean": None,
        "graph_entropy_ratio": None,
        "alpha_std": None,
    }
    if not log_file.exists():
        return out

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[Reg Terms]" in line:
                val_pos = line.rfind("| Val:")
                section = line[val_pos:] if val_pos >= 0 else line
                val_ratio = extract_float(section, "reg_bce_ratio")
                if val_ratio is not None:
                    out["reg_bce_ratio"] = val_ratio

            if "[Diag][M3] Epoch" in line or "[Diag] Epoch" in line:
                mf = extract_float(line, "mf_abs_mean")
                residual = extract_float(line, "residual_abs_mean")
                gate = extract_float(line, "gate_mean")
                irt = extract_float(line, "irt_abs_mean")
                delta = extract_float(line, "delta_abs_mean")
                ger = extract_float(line, "graph_entropy_ratio")
                alpha_std = extract_float(line, "alpha_std")
                dor = extract_float(line, "delta_over_irt")

                if mf is not None:
                    out["mf_abs_mean"] = mf
                if residual is not None:
                    out["residual_abs_mean"] = residual
                if gate is not None:
                    out["gate_mean"] = gate
                if ger is not None:
                    out["graph_entropy_ratio"] = ger
                if alpha_std is not None:
                    out["alpha_std"] = alpha_std
                if dor is not None:
                    out["delta_over_irt"] = dor
                elif delta is not None and irt is not None:
                    out["delta_over_irt"] = float(delta) / (abs(float(irt)) + 1e-12)

    return out


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_result(job: JobSpec, exit_code: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": job.dataset,
        "seed": job.seed,
        "grid_tag": job.grid_tag,
        "variant": "no3" if job.variant == "no_module3" else "full",
        "model_variant": job.model_variant,
        "save_dir": str(job.save_dir),
        "log_dir": str(job.log_dir),
        "exit_code": int(exit_code),
        "status": "ok" if exit_code == 0 else "failed",
    }

    test_json = read_json(job.save_dir / "test_results.json") or {}
    metrics = test_json.get("metrics", {}) if isinstance(test_json, dict) else {}
    row["test_auc"] = metrics.get("auc")
    row["test_acc"] = metrics.get("acc")
    row["test_rmse"] = metrics.get("rmse")
    row["best_val_auc"] = test_json.get("best_val_auc") if isinstance(test_json, dict) else None
    row["model_epoch"] = test_json.get("model_epoch") if isinstance(test_json, dict) else None

    history_json = read_json(job.save_dir / "training_history.json") or {}
    if isinstance(history_json, dict):
        row["best_epoch"] = history_json.get("best_epoch")
        if row.get("best_val_auc") is None:
            row["best_val_auc"] = history_json.get("best_val_auc")

    if row.get("best_epoch") is None:
        row["best_epoch"] = row.get("model_epoch")

    log_file = latest_train_log(job.log_dir)
    row["log_file"] = str(log_file) if log_file else ""
    if log_file is not None:
        row.update(parse_diag_from_log(log_file))

    row["overrides_json"] = json.dumps(job.params, ensure_ascii=False, sort_keys=True)
    return row


def append_result_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "seed",
        "grid_tag",
        "variant",
        "model_variant",
        "test_auc",
        "test_acc",
        "test_rmse",
        "best_val_auc",
        "best_epoch",
        "model_epoch",
        "reg_bce_ratio",
        "delta_over_irt",
        "mf_abs_mean",
        "residual_abs_mean",
        "gate_mean",
        "graph_entropy_ratio",
        "alpha_std",
        "status",
        "exit_code",
        "save_dir",
        "log_dir",
        "log_file",
        "overrides_json",
    ]

    need_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def make_jobs(
    datasets: Sequence[str],
    seeds: Sequence[int],
    variants: Sequence[str],
    epochs_override: Optional[int],
    main_arg_dests: Set[str],
) -> List[JobSpec]:
    jobs: List[JobSpec] = []

    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' is missing in BEST_CFG.")
        base_cfg = dict(BEST_CFG[dataset])
        grid_points, skip_msgs = build_grid_points(base_cfg)
        for msg in skip_msgs:
            print(f"[GRID-SKIP] dataset={dataset}: {msg}")

        for seed in seeds:
            for gp in grid_points:
                shared = build_shared_params(
                    dataset=dataset,
                    base_cfg=base_cfg,
                    grid_overrides=gp.overrides,
                    epochs_override=epochs_override,
                )
                for variant in variants:
                    params = build_variant_params(shared, variant)
                    validate_params(dataset, params, main_arg_dests)

                    suffix = "no3" if variant == "no_module3" else "full"
                    model_variant = f"{dataset}_m3grid_{gp.tag}_{suffix}"
                    save_dir = Path("checkpoints") / f"{dataset}_m3grid" / f"seed{seed}" / f"{gp.tag}_{suffix}"
                    log_dir = Path("logs") / f"{dataset}_m3grid" / f"seed{seed}" / f"{gp.tag}_{suffix}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    log_dir.mkdir(parents=True, exist_ok=True)

                    job = JobSpec(
                        dataset=dataset,
                        seed=int(seed),
                        grid_tag=gp.tag,
                        variant=variant,
                        model_variant=model_variant,
                        save_dir=save_dir,
                        log_dir=log_dir,
                        params=params,
                        cmd=[],
                    )
                    job.cmd = build_command(job)

                    has_ablate_flag = "--ablate_module3" in job.cmd
                    if variant == "full" and has_ablate_flag:
                        raise RuntimeError(f"[{model_variant}] full command unexpectedly contains --ablate_module3.")
                    if variant == "no_module3" and not has_ablate_flag:
                        raise RuntimeError(f"[{model_variant}] no_module3 command misses --ablate_module3.")

                    jobs.append(job)
    return jobs


def print_job_brief(job: JobSpec, gpu_id: int) -> None:
    short = (
        f"[PLAN] dataset={job.dataset} seed={job.seed} tag={job.grid_tag} "
        f"variant={job.variant} gpu={gpu_id} model_variant={job.model_variant}"
    )
    print(short)
    print("       CMD:", shlex.join(job.cmd))


def run_dry(jobs: Sequence[JobSpec], gpus: Sequence[int]) -> None:
    if not gpus:
        raise ValueError("No GPUs available for dry-run display.")
    for idx, job in enumerate(jobs):
        gpu_id = gpus[idx % len(gpus)]
        print_job_brief(job, gpu_id)


def run_jobs(
    jobs: Sequence[JobSpec],
    gpus: List[int],
    max_concurrent: int,
    max_per_gpu: int,
    poll_interval: int,
) -> None:
    running: List[Tuple[subprocess.Popen, int, JobSpec]] = []
    gpu_load: Dict[int, int] = {gid: 0 for gid in gpus}
    rr = 0
    next_job_idx = 0

    while next_job_idx < len(jobs) or running:
        new_running: List[Tuple[subprocess.Popen, int, JobSpec]] = []
        for proc, gpu, job in running:
            ret = proc.poll()
            if ret is None:
                new_running.append((proc, gpu, job))
                continue

            gpu_load[gpu] = max(0, gpu_load.get(gpu, 0) - 1)
            row = collect_result(job, ret)
            append_result_row(RESULT_CSV, row)
            print(
                f"[DONE] dataset={job.dataset} seed={job.seed} tag={job.grid_tag} "
                f"variant={job.variant} gpu={gpu} exit={ret} auc={row.get('test_auc')}"
            )
        running = new_running

        while next_job_idx < len(jobs) and len(running) < max_concurrent:
            gpu_id, rr = pick_gpu_with_slot_round_robin(gpus, gpu_load, max_per_gpu, rr)
            if gpu_id is None:
                break

            job = jobs[next_job_idx]
            print_job_brief(job, gpu_id)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.Popen(job.cmd, env=env)

            running.append((proc, gpu_id, job))
            gpu_load[gpu_id] = gpu_load.get(gpu_id, 0) + 1
            next_job_idx += 1

        if running:
            time.sleep(max(1, int(poll_interval)))
        elif next_job_idx < len(jobs):
            time.sleep(1)


def main() -> None:
    args = parse_cli()
    if args.ablation_set != "model":
        raise ValueError(f"--ablation_set currently only supports 'model', got '{args.ablation_set}'.")

    datasets = parse_csv_tokens(args.datasets)
    if not datasets:
        raise ValueError("No datasets provided.")

    seeds = list(DEFAULT_SEEDS) if args.seeds is None else parse_int_csv(args.seeds)
    if not seeds:
        raise ValueError("No seeds provided.")

    variants = normalize_variants(args.only_variants)
    gpus = parse_gpu_ids(args.gpus)
    if not gpus:
        raise ValueError("No GPUs provided. Example: --gpus 0 or --gpus 0,1,2,3")

    max_concurrent = calc_effective_max_concurrent(args.max_concurrent, gpus, args.max_per_gpu)
    max_per_gpu = max(1, int(args.max_per_gpu))
    poll_interval = max(1, int(args.poll_interval))
    main_arg_dests = get_main_arg_dests()
    epochs_override = int(args.epochs) if epochs_was_explicitly_set(sys.argv[1:]) else None

    jobs = make_jobs(
        datasets=datasets,
        seeds=seeds,
        variants=variants,
        epochs_override=epochs_override,
        main_arg_dests=main_arg_dests,
    )

    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Variants: {variants}")
    print(
        f"GPUs: {gpus}, max_concurrent={args.max_concurrent}, "
        f"max_per_gpu={max_per_gpu}, effective_max={max_concurrent}"
    )
    print(f"Jobs: {len(jobs)}")
    print(f"Result CSV: {RESULT_CSV}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        run_dry(jobs, gpus)
        return

    run_jobs(
        jobs=jobs,
        gpus=gpus,
        max_concurrent=max_concurrent,
        max_per_gpu=max_per_gpu,
        poll_interval=poll_interval,
    )


if __name__ == "__main__":
    main()






