#!/usr/bin/env python
"""Run the clean Graph-IRT ablation suite.

The runner intentionally exposes only interventions with a single, auditable
meaning.  Historical ``no_A``/``no_E`` aliases are not accepted.
"""

from __future__ import annotations

import argparse
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
    "no_item_matching": AblationSpec(
        "no_item_matching", {"enable_item_matching": False}
    ),
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
    jobs: List[JobSpec] = []

    for dataset in datasets:
        if dataset not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Dataset {dataset!r} is not present in EXPERIMENT_CONFIGS.")
        for ablation in ablations:
            for seed in seeds:
                params = dict(EXPERIMENT_CONFIGS[dataset])
                params.update(ablation.overrides)
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
