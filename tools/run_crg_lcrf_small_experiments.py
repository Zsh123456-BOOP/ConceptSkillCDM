#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run paper-style CRG/LCRF small experiments from trained checkpoints.

This orchestrator is inference-only.  It does not train new models.  It bundles
the four paper evidence steps:

1. dataset phenomenon profile,
2. CRG held-out reachability retrieval,
3. CRG support corruption,
4. LCRF counterfactual and case-study export.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("assist_09", "junyi", "assist_17", "nips34")


DEFAULT_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "assist_09": {
        "full": "checkpoints/abce_diag/recover_ed553d3_assist09_gpu2_20260518_140623/assist_09/seed42/best_full",
        "no_A": "checkpoints/abce_diag/recover_ed553d3_assist09_gpu2_20260518_140623/assist_09/seed42/best_no_A",
        "no_E": "checkpoints/abce_diag/recover_ed553d3_assist09_gpu2_20260518_140623/assist_09/seed42/best_no_E",
    },
    "junyi": {
        "full": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/junyi/seed42/best_full",
        "no_A": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/junyi/seed42/best_no_A",
        "no_E": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/junyi/seed42/best_no_E",
    },
    "assist_17": {
        "full": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/assist_17/seed42/best_full",
        "no_A": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/assist_17/seed42/best_no_A",
        "no_E": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/assist_17/seed42/best_no_E",
    },
    "nips34": {
        "full": "checkpoints/mechanism/recover_ed553d3_nips34_gpu3_20260519_044400/phase2/nips34/full",
        "no_A": "checkpoints/mechanism/recover_ed553d3_nips34_gpu3_20260519_044400/phase2/nips34/no_A",
        "no_E": "checkpoints/mechanism/recover_ed553d3_nips34_gpu3_20260519_044400/phase2/nips34/no_E",
    },
}


def _csv_tokens(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _load_checkpoint_map(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    data = json.loads(json.dumps(DEFAULT_CHECKPOINTS))
    if path is None:
        return data
    raw = json.loads(path.read_text(encoding="utf-8"))
    for dataset, mapping in raw.items():
        item = data.setdefault(dataset, {})
        item.update({str(k): str(v) for k, v in dict(mapping).items()})
    return data


def _exists_checkpoint(save_dir: Path) -> bool:
    return (save_dir / "best_model.pth").exists()


def _run(
    cmd: Sequence[str],
    *,
    log_file: Path,
    env: Mapping[str, str],
    dry_run: bool,
) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(x) for x in cmd)
    print(f"[RUN] {printable}")
    print(f"[LOG] {log_file}")
    if dry_run:
        return 0
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(list(cmd), cwd=ROOT, env=dict(env), stdout=f, stderr=subprocess.STDOUT)
    print(f"[DONE] exit={proc.returncode}")
    return int(proc.returncode)


def _run_optional_plot(
    cmd: Sequence[str],
    *,
    input_dir: Path,
    log_file: Path,
    env: Mapping[str, str],
    dry_run: bool,
) -> int:
    if not input_dir.exists():
        print(f"[SKIP] plot input missing: {input_dir}")
        return 0
    return _run(cmd, log_file=log_file, env=env, dry_run=dry_run)


def _append_status(path: Path, rows: Iterable[MutableMapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old: List[Dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        import csv

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            old = list(csv.DictReader(f))
    rows = [dict(row) for row in rows]
    keys: List[str] = []
    for row in [*old, *rows]:
        for key in row:
            if key not in keys:
                keys.append(key)
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old)
        writer.writerows(rows)


def _checkpoint_dir(mapping: Mapping[str, str], name: str) -> Optional[Path]:
    value = mapping.get(name)
    if not value:
        return None
    path = ROOT / value
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=f"crg_lcrf_small_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--checkpoint_json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--top_cases", type=int, default=3)
    parser.add_argument("--candidate_pool", type=int, default=80)
    parser.add_argument("--skip_profile", action="store_true")
    parser.add_argument("--skip_retrieval", action="store_true")
    parser.add_argument("--skip_corruption", action="store_true")
    parser.add_argument("--skip_cases", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    datasets = _csv_tokens(args.datasets)
    output_root = Path(args.output_root) if args.output_root else Path("results") / args.run_id
    log_root = Path("logs") / args.run_id
    status_csv = output_root / "small_experiment_status.csv"
    ckpts = _load_checkpoint_map(args.checkpoint_json)
    env = os.environ.copy()

    rows: List[Dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "checkpoint_manifest.json").write_text(
        json.dumps({d: ckpts.get(d, {}) for d in datasets}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.skip_profile:
        out_dir = output_root / "data_phenomenon"
        ret = _run(
            [
                sys.executable,
                "tools/profile_ae_data_readiness.py",
                "--datasets",
                ",".join(datasets),
                "--out_csv",
                str(out_dir / "crg_lcrf_data_readiness.csv"),
                "--out_json",
                str(out_dir / "crg_lcrf_data_readiness.json"),
            ],
            log_file=log_root / "data_phenomenon.log",
            env=env,
            dry_run=args.dry_run,
        )
        rows.append({"step": "data_phenomenon", "dataset": ",".join(datasets), "exit_code": ret, "output_dir": str(out_dir)})
        _append_status(status_csv, rows[-1:])

    for dataset in datasets:
        if not args.skip_retrieval:
            out_dir = output_root / "crg_retrieval" / dataset
            ret = _run(
                [
                    sys.executable,
                    "tools/analyze_a_support_evidence.py",
                    "--dataset_name",
                    dataset,
                    "--output_dir",
                    str(out_dir),
                    "--device",
                    args.device,
                    "--batch_size",
                    str(args.batch_size),
                    "--num_workers",
                    str(args.num_workers),
                ],
                log_file=log_root / dataset / "crg_retrieval.log",
                env=env,
                dry_run=args.dry_run,
            )
            rows.append({"step": "crg_retrieval", "dataset": dataset, "exit_code": ret, "output_dir": str(out_dir)})
            _append_status(status_csv, rows[-1:])

        mapping = ckpts.get(dataset, {})
        full_dir = _checkpoint_dir(mapping, "full")
        no_a_dir = _checkpoint_dir(mapping, "no_A")
        no_e_dir = _checkpoint_dir(mapping, "no_E")

        if not args.skip_corruption:
            out_dir = output_root / "crg_support_corruption" / dataset
            if full_dir is None or not _exists_checkpoint(full_dir):
                print(f"[SKIP] {dataset} CRG corruption: missing full checkpoint")
                ret = 99
            else:
                ret = _run(
                    [
                        sys.executable,
                        "tools/analyze_a_support_corruption.py",
                        "--dataset_name",
                        dataset,
                        "--output_dir",
                        str(out_dir),
                        "--full_save_dir",
                        str(full_dir),
                        "--device",
                        args.device,
                        "--batch_size",
                        str(args.batch_size),
                        "--num_workers",
                        str(args.num_workers),
                        "--corruption_fracs",
                        "0",
                        "0.25",
                        "0.50",
                        "0.75",
                        "1.0",
                        "--random_trials",
                        "5",
                    ],
                    log_file=log_root / dataset / "crg_support_corruption.log",
                    env=env,
                    dry_run=args.dry_run,
                )
            rows.append({"step": "crg_support_corruption", "dataset": dataset, "exit_code": ret, "output_dir": str(out_dir)})
            _append_status(status_csv, rows[-1:])

        if not args.skip_cases:
            out_dir = output_root / "lcrf_case_studies" / dataset
            if not all(path is not None and _exists_checkpoint(path) for path in (full_dir, no_a_dir, no_e_dir)):
                print(f"[SKIP] {dataset} LCRF cases: missing full/no_A/no_E checkpoint")
                ret = 99
            else:
                ret = _run(
                    [
                        sys.executable,
                        "tools/export_ae_case_studies.py",
                        "--dataset_name",
                        dataset,
                        "--full_dir",
                        str(full_dir),
                        "--no_a_dir",
                        str(no_a_dir),
                        "--no_e_dir",
                        str(no_e_dir),
                        "--output_dir",
                        str(out_dir),
                        "--top_cases",
                        str(args.top_cases),
                        "--candidate_pool",
                        str(args.candidate_pool),
                        "--batch_size",
                        str(args.batch_size),
                        "--num_workers",
                        str(args.num_workers),
                        "--device",
                        args.device,
                        "--e_counterfactuals",
                    ],
                    log_file=log_root / dataset / "lcrf_case_studies.log",
                    env=env,
                    dry_run=args.dry_run,
                )
            rows.append({"step": "lcrf_case_studies", "dataset": dataset, "exit_code": ret, "output_dir": str(out_dir)})
            _append_status(status_csv, rows[-1:])

    if not args.skip_plots:
        for dataset in datasets:
            specs = [
                (
                    "crg_retrieval_plot",
                    output_root / "crg_retrieval" / dataset,
                    [
                        "Rscript",
                        "tools/plot_a_support_evidence.R",
                        str(output_root / "crg_retrieval" / dataset),
                        str(output_root / "crg_retrieval" / dataset / "figures"),
                    ],
                ),
                (
                    "crg_support_corruption_plot",
                    output_root / "crg_support_corruption" / dataset,
                    [
                        "Rscript",
                        "tools/plot_a_support_corruption.R",
                        str(output_root / "crg_support_corruption" / dataset),
                        str(output_root / "crg_support_corruption" / dataset / "figures"),
                    ],
                ),
                (
                    "lcrf_case_studies_plot",
                    output_root / "lcrf_case_studies" / dataset,
                    [
                        "Rscript",
                        "tools/plot_ae_case_studies.R",
                        str(output_root / "lcrf_case_studies" / dataset),
                        str(output_root / "lcrf_case_studies" / dataset / "figures"),
                    ],
                ),
            ]
            for step, in_dir, cmd in specs:
                ret = _run_optional_plot(
                    cmd,
                    input_dir=in_dir,
                    log_file=log_root / dataset / f"{step}.log",
                    env=env,
                    dry_run=args.dry_run,
                )
                rows.append({"step": step, "dataset": dataset, "exit_code": ret, "output_dir": str(in_dir)})
                _append_status(status_csv, rows[-1:])

    summary = {
        "run_id": args.run_id,
        "datasets": datasets,
        "output_root": str(output_root),
        "status_csv": str(status_csv),
        "rows": rows,
    }
    (output_root / "small_experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
