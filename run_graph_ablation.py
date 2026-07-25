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
from src.config import TRAIN_EVIDENCE_MODES


@dataclass(frozen=True)
class AblationSpec:
    name: str
    overrides: Dict[str, Any]


@dataclass(frozen=True)
class JobSpec:
    dataset: str
    seed: int
    ablation: AblationSpec
    train_evidence_mode: str
    save_dir: Path
    log_dir: Path
    params: Dict[str, Any]
    cmd: List[str]


ABLATIONS: Dict[str, AblationSpec] = {
    "full": AblationSpec("full", {}),
    "gec_residual": AblationSpec("gec_residual", {}),
    "gec_relation_residual": AblationSpec(
        "gec_relation_residual",
        {
            # One global adapter protocol for every dataset.  Candidate early
            # stopping is separate from the final parent/candidate selection.
            "epochs": 30,
            "min_epochs": 3,
            "patience": 3,
            "early_stop_patience": 6,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
        },
    ),
    "gec_branch_gate": AblationSpec(
        "gec_branch_gate",
        {
            "epochs": 30,
            "min_epochs": 3,
            "patience": 3,
            "early_stop_patience": 6,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "max_train_batches": 200,
        },
    ),
    "no_response_evidence": AblationSpec("no_response_evidence", {}),
    "no_evidence_anchor": AblationSpec("no_evidence_anchor", {}),
    "no_evidence_propagation": AblationSpec("no_evidence_propagation", {}),
    "pairwise_auc": AblationSpec("pairwise_auc", {}),
    "no_message_passing": AblationSpec(
        "no_message_passing", {"graph_propagation_alpha": 0.0}
    ),
    "no_graph_calibration": AblationSpec(
        "no_graph_calibration", {"graph_propagation_alpha": 0.0}
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
    parser.add_argument(
        "--ablations",
        default=",".join(
            name
            for name in ABLATIONS
            if name
            not in {
                "gec_residual",
                "gec_relation_residual",
                "gec_branch_gate",
            }
        ),
        help=(
            "Comma-separated variants. Residual adapters are opt-in because "
            "they require --warm_start_run_id."
        ),
    )
    parser.add_argument(
        "--train_evidence_modes",
        default="excluded",
        help=(
            "Comma-separated training-query evidence modes. "
            f"Available={','.join(TRAIN_EVIDENCE_MODES)}"
        ),
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_interval", type=int, default=10)
    parser.add_argument("--run_id", default=None)
    parser.add_argument(
        "--warm_start_run_id",
        default=None,
        help=(
            "Run id containing paired full/v1 checkpoints. Required when "
            "training a residual adapter."
        ),
    )
    parser.add_argument(
        "--run_mode",
        choices=("train", "test"),
        default="train",
        help=(
            "Explicit experiment phase. train is validation-only model selection; "
            "test evaluates checkpoints from an existing --run_id and never trains them."
        ),
    )
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
    train_evidence_modes = _csv_tokens(args.train_evidence_modes)
    unknown_modes = sorted(set(train_evidence_modes) - set(TRAIN_EVIDENCE_MODES))
    if unknown_modes:
        raise ValueError(
            "Unknown training evidence mode(s): "
            f"{unknown_modes}. Available={list(TRAIN_EVIDENCE_MODES)}"
        )
    if not train_evidence_modes:
        raise ValueError("At least one training evidence mode is required.")
    requested_run_id = run_id or args.run_id
    if args.run_mode == "test" and not requested_run_id:
        raise ValueError(
            "--run_mode test requires --run_id for an existing validation-selected run."
        )
    session = requested_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    residual_names = {
        "gec_residual",
        "gec_relation_residual",
        "gec_branch_gate",
    }
    residual_requested = any(
        ablation.name in residual_names for ablation in ablations
    )
    if args.run_mode == "train" and residual_requested and not args.warm_start_run_id:
        raise ValueError(
            "training a residual adapter requires --warm_start_run_id"
        )
    if (
        args.run_mode == "train"
        and args.warm_start_run_id
        and not residual_requested
    ):
        raise ValueError(
            "--warm_start_run_id is only valid when training a residual adapter"
        )
    jobs: List[JobSpec] = []

    for dataset in datasets:
        if dataset not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Dataset {dataset!r} is not present in EXPERIMENT_CONFIGS.")
        for ablation in ablations:
            for train_evidence_mode in train_evidence_modes:
                for seed in seeds:
                    params = dict(EXPERIMENT_CONFIGS[dataset])
                    params.update(ablation.overrides)
                    params.pop("num_gpus", None)
                    params.pop("seed", None)
                    params["model_variant"] = ablation.name
                    params["train_evidence_mode"] = train_evidence_mode
                    if (
                        args.run_mode == "train"
                        and ablation.name in residual_names
                    ):
                        base_tag = (
                            f"{dataset}_graph_irt_full_"
                            f"{train_evidence_mode}_seed{seed}_"
                            f"{args.warm_start_run_id}"
                        )
                        params["warm_start_checkpoint"] = str(
                            Path("checkpoints") / base_tag / "best_model.pth"
                        )

                    tag = (
                        f"{dataset}_graph_irt_{ablation.name}_"
                        f"{train_evidence_mode}_seed{seed}_{session}"
                    )
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
                        "--run_mode",
                        str(args.run_mode),
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
                            train_evidence_mode=train_evidence_mode,
                            save_dir=save_dir,
                            log_dir=log_dir,
                            params=params,
                            cmd=cmd,
                        )
                    )
    return jobs


def _require_existing_test_checkpoints(jobs: Sequence[JobSpec]) -> None:
    """Fail before launching if a test phase has no sealed training checkpoint."""
    missing = [
        str(job.save_dir / "best_model.pth")
        for job in jobs
        if not (job.save_dir / "best_model.pth").is_file()
    ]
    if missing:
        preview = "\n  ".join(missing[:10])
        suffix = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            "Test phase only evaluates existing checkpoints; missing:\n  "
            f"{preview}{suffix}"
        )


def _require_fresh_training_dirs(jobs: Sequence[JobSpec]) -> None:
    """Refuse a training launch that could overwrite an earlier run."""
    conflicts = [
        str(path)
        for job in jobs
        for path in (job.save_dir, job.log_dir)
        if path.exists()
    ]
    if conflicts:
        preview = "\n  ".join(conflicts[:10])
        suffix = (
            f"\n  ... and {len(conflicts) - 10} more"
            if len(conflicts) > 10
            else ""
        )
        raise FileExistsError(
            "Training run directories already exist; choose a new --run_id:\n  "
            f"{preview}{suffix}"
        )


def _require_residual_warm_starts(jobs: Sequence[JobSpec]) -> None:
    """Fail before launch unless every residual job has its paired v1 parent."""
    missing = [
        str(job.params.get("warm_start_checkpoint", ""))
        for job in jobs
        if job.ablation.name
        in {"gec_residual", "gec_relation_residual", "gec_branch_gate"}
        and not Path(str(job.params.get("warm_start_checkpoint", ""))).is_file()
    ]
    if missing:
        preview = "\n  ".join(missing[:10])
        suffix = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            "Residual training requires paired full/v1 checkpoints:\n  "
            f"{preview}{suffix}"
        )


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
        f"Graph-IRT phase={args.run_mode}, jobs={len(jobs)}, "
        f"ablations={_csv_tokens(args.ablations)}, "
        f"train_evidence_modes={_csv_tokens(args.train_evidence_modes)}, "
        f"seeds={args.seeds or DEFAULT_SEEDS}, dry_run={args.dry_run}"
    )
    if args.dry_run:
        for job in jobs:
            print(" ".join(job.cmd))
        return

    if args.run_mode == "test":
        _require_existing_test_checkpoints(jobs)
    else:
        _require_fresh_training_dirs(jobs)
        _require_residual_warm_starts(jobs)

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
                        f"{job.dataset}/{job.ablation.name}/"
                        f"{job.train_evidence_mode}/seed{job.seed}: exit {code}"
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
                if args.run_mode == "train":
                    job.save_dir.mkdir(parents=True, exist_ok=True)
                job.log_dir.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                print(
                    f"[LAUNCH] {job.dataset}/{job.ablation.name}/"
                    f"{job.train_evidence_mode}/seed{job.seed} gpu={gpu}\n"
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
