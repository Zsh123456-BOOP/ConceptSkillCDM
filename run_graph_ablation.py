#!/usr/bin/env python
"""Run the clean Graph-IRT ablation suite.

The runner intentionally exposes only interventions with a single, auditable
meaning.  Historical ``no_A``/``no_E`` aliases are not accepted.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiment_configs import DEFAULT_SEEDS, EXPERIMENT_CONFIGS
from gpu_utils import (
    calc_effective_max_concurrent,
    parse_gpu_ids,
    parse_int_csv,
    pick_gpu_with_slot_round_robin,
)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    overrides: Dict[str, Any]


@dataclass(frozen=True)
class JobSpec:
    dataset: str
    seed: int
    ablation: AblationSpec
    save_dir: Path
    log_dir: Path
    params: Dict[str, Any]
    cmd: List[str]


ABLATIONS: Dict[str, AblationSpec] = {
    "full": AblationSpec("full", {}),
    "no_message_passing": AblationSpec(
        "no_message_passing", {"graph_propagation_alpha": 0.0}
    ),
    "item_only": AblationSpec("item_only", {"graph_prior_mode": "item_only"}),
    "exposure_only": AblationSpec(
        "exposure_only", {"graph_prior_mode": "exposure_only"}
    ),
    "degree_random": AblationSpec(
        "degree_random", {"graph_prior_mode": "degree_random"}
    ),
}


def _csv_tokens(value: str) -> List[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _append_cli_arg(cmd: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{key}")
        return
    cmd.extend((f"--{key}", str(value)))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Graph-IRT ablations.")
    parser.add_argument("--datasets", default="assist_09,assist_17,junyi")
    parser.add_argument("--ablations", default=",".join(ABLATIONS))
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_interval", type=int, default=10)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--generate_diagnosis", action="store_true")
    parser.add_argument(
        "--student_concept_interaction",
        choices=("none", "hadamard", "low_rank"),
        default=None,
        help="Override the interaction mode for every full/ablation job.",
    )
    parser.add_argument(
        "--student_concept_interaction_scale",
        type=float,
        default=None,
        help=(
            "Override the interaction scale for every full/ablation job; "
            "must be positive when low_rank is active."
        ),
    )
    parser.add_argument(
        "--student_concept_interaction_ratio_cap",
        type=float,
        default=None,
        help="Override the per-student interaction RMS ratio cap; 0 disables it.",
    )
    parser.add_argument(
        "--student_concept_interaction_rank",
        type=int,
        default=None,
        help="Override the low-rank interaction width for every full/ablation job.",
    )
    parser.add_argument(
        "--student_concept_interaction_init_std",
        type=float,
        default=None,
        help=(
            "Override the low-rank factor initialization std for every full/ablation job; "
            "must be positive when low_rank is active."
        ),
    )
    return parser.parse_args(argv)


def _selected_ablations(value: str) -> List[AblationSpec]:
    names = _csv_tokens(value)
    unknown = sorted(set(names) - set(ABLATIONS))
    if unknown:
        raise ValueError(
            f"Unknown ablation(s): {unknown}. Available={sorted(ABLATIONS)}"
        )
    if not names:
        raise ValueError("At least one ablation is required.")
    return [ABLATIONS[name] for name in names]


def make_jobs(args: argparse.Namespace, run_id: Optional[str] = None) -> List[JobSpec]:
    datasets = _csv_tokens(args.datasets)
    seeds = list(DEFAULT_SEEDS) if args.seeds is None else parse_int_csv(args.seeds)
    ablations = _selected_ablations(args.ablations)
    session = run_id or args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    interaction_scale = args.student_concept_interaction_scale
    if interaction_scale is not None:
        interaction_scale = float(interaction_scale)
        if not math.isfinite(interaction_scale) or not 0.0 <= interaction_scale <= 4.0:
            raise ValueError(
                "student_concept_interaction_scale must be finite and in [0, 4], "
                f"got {args.student_concept_interaction_scale!r}"
            )
    interaction_rank = args.student_concept_interaction_rank
    if interaction_rank is not None and int(interaction_rank) <= 0:
        raise ValueError(
            "student_concept_interaction_rank must be positive, "
            f"got {args.student_concept_interaction_rank!r}"
        )
    interaction_ratio_cap = args.student_concept_interaction_ratio_cap
    if interaction_ratio_cap is not None:
        interaction_ratio_cap = float(interaction_ratio_cap)
        if not math.isfinite(interaction_ratio_cap) or not 0.0 <= interaction_ratio_cap <= 4.0:
            raise ValueError(
                "student_concept_interaction_ratio_cap must be finite and in [0, 4], "
                f"got {args.student_concept_interaction_ratio_cap!r}"
            )
    interaction_init_std = args.student_concept_interaction_init_std
    if interaction_init_std is not None:
        interaction_init_std = float(interaction_init_std)
        if not math.isfinite(interaction_init_std) or not 0.0 <= interaction_init_std <= 1.0:
            raise ValueError(
                "student_concept_interaction_init_std must be finite and in [0, 1], "
                f"got {args.student_concept_interaction_init_std!r}"
            )
    jobs: List[JobSpec] = []

    for dataset in datasets:
        if dataset not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Dataset {dataset!r} is not present in EXPERIMENT_CONFIGS.")
        for ablation in ablations:
            for seed in seeds:
                params = dict(EXPERIMENT_CONFIGS[dataset])
                params.update(ablation.overrides)
                if args.student_concept_interaction is not None:
                    params["student_concept_interaction"] = args.student_concept_interaction
                if args.student_concept_interaction_scale is not None:
                    params["student_concept_interaction_scale"] = interaction_scale
                if args.student_concept_interaction_ratio_cap is not None:
                    params["student_concept_interaction_ratio_cap"] = interaction_ratio_cap
                if args.student_concept_interaction_rank is not None:
                    params["student_concept_interaction_rank"] = int(interaction_rank)
                if args.student_concept_interaction_init_std is not None:
                    params["student_concept_interaction_init_std"] = interaction_init_std
                if (
                    params.get("student_concept_interaction", "none") == "low_rank"
                    and float(params.get("student_concept_interaction_scale", 1.0)) == 0.0
                ):
                    raise ValueError(
                        "student_concept_interaction_scale must be positive for low_rank "
                        "to avoid a disabled interaction"
                    )
                if (
                    params.get("student_concept_interaction", "none") == "low_rank"
                    and float(params.get("student_concept_interaction_init_std", 0.1)) == 0.0
                ):
                    raise ValueError(
                        "student_concept_interaction_init_std must be positive for low_rank "
                        "to avoid zero-gradient factors"
                    )
                params.pop("num_gpus", None)
                params.pop("seed", None)
                params["model_variant"] = ablation.name

                tag = f"{dataset}_graph_irt_{ablation.name}_seed{seed}_{session}"
                save_dir = Path("checkpoints") / tag
                log_dir = Path("logs") / tag
                cmd = [
                    sys.executable,
                    "main.py",
                    "--dataset_name",
                    dataset,
                    "--seed",
                    str(seed),
                    "--save_dir",
                    str(save_dir),
                    "--log_dir",
                    str(log_dir),
                ]
                for key, value in params.items():
                    _append_cli_arg(cmd, key, value)
                if args.generate_diagnosis:
                    cmd.extend(("--generate_diagnosis", "True"))
                else:
                    cmd.extend(("--generate_diagnosis", "False"))

                jobs.append(
                    JobSpec(
                        dataset=dataset,
                        seed=int(seed),
                        ablation=ablation,
                        save_dir=save_dir,
                        log_dir=log_dir,
                        params=params,
                        cmd=cmd,
                    )
                )
    return jobs


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    jobs = make_jobs(args)
    gpus = parse_gpu_ids(args.gpus)
    if not gpus:
        raise ValueError("No GPU ids supplied. Use --gpus 0 (or another visible id).")
    max_concurrent = calc_effective_max_concurrent(
        max(1, int(args.max_concurrent)), gpus, max(1, int(args.max_per_gpu))
    )

    print(
        f"Graph-IRT jobs={len(jobs)}, ablations={_csv_tokens(args.ablations)}, "
        f"seeds={args.seeds or DEFAULT_SEEDS}, dry_run={args.dry_run}"
    )
    if args.dry_run:
        for job in jobs:
            print(" ".join(job.cmd))
        return

    running: List[tuple[subprocess.Popen, int, JobSpec]] = []
    gpu_load = {gpu: 0 for gpu in gpus}
    next_job = 0
    next_gpu_index = 0
    failures: List[str] = []
    try:
        while next_job < len(jobs) or running:
            active: List[tuple[subprocess.Popen, int, JobSpec]] = []
            for process, gpu, job in running:
                code = process.poll()
                if code is None:
                    active.append((process, gpu, job))
                    continue
                gpu_load[gpu] = max(0, gpu_load[gpu] - 1)
                if code != 0:
                    failures.append(
                        f"{job.dataset}/{job.ablation.name}/seed{job.seed}: exit {code}"
                    )
            running = active

            if failures and not running:
                raise RuntimeError("Graph-IRT jobs failed: " + "; ".join(failures))

            while not failures and next_job < len(jobs) and len(running) < max_concurrent:
                gpu, next_gpu_index = pick_gpu_with_slot_round_robin(
                    gpus,
                    gpu_load,
                    max(1, int(args.max_per_gpu)),
                    next_gpu_index,
                )
                if gpu is None:
                    break
                job = jobs[next_job]
                job.save_dir.mkdir(parents=True, exist_ok=True)
                job.log_dir.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                print(
                    f"[LAUNCH] {job.dataset}/{job.ablation.name}/seed{job.seed} gpu={gpu}\n"
                    f"         {' '.join(job.cmd)}"
                )
                process = subprocess.Popen(job.cmd, env=env)
                running.append((process, gpu, job))
                gpu_load[gpu] += 1
                next_job += 1

            if running:
                time.sleep(max(1, int(args.poll_interval)))
    except KeyboardInterrupt:
        for process, _, _ in running:
            process.terminate()
        for process, _, _ in running:
            process.wait()
        raise


if __name__ == "__main__":
    main()
