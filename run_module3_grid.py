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
from itertools import product
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
    parser.add_argument(
        "--grid_points",
        type=int,
        default=16,
        help=(
            "Per-dataset grid-point budget. "
            "Approx runs = len(datasets) * len(seeds) * grid_points * len(variants). "
            "Default 16 => about 64 runs for 2 datasets with full/no_module3."
        ),
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--rerun_existing",
        action="store_true",
        help="Force rerun even if the same (dataset,seed,grid_tag,variant) already succeeded in results/module3_grid_results.csv.",
    )
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


def variant_csv_name(variant: str) -> str:
    return "no3" if variant == "no_module3" else "full"


def variant_internal_name(csv_variant: str) -> Optional[str]:
    raw = str(csv_variant).strip().lower()
    if raw == "full":
        return "full"
    if raw in {"no3", "no_module3"}:
        return "no_module3"
    return None


def make_job_key(dataset: str, seed: int, grid_tag: str, variant: str) -> Tuple[str, int, str, str]:
    return (str(dataset), int(seed), str(grid_tag), str(variant))


def _uniq_keep_order(values: Sequence[float]) -> List[float]:
    out: List[float] = []
    seen: Set[float] = set()
    for v in values:
        key = round(float(v), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(v))
    return out


def _fmt_float_tag(v: float) -> str:
    txt = f"{float(v):.3f}".rstrip("0").rstrip(".")
    txt = txt.replace("-", "m").replace(".", "p")
    return txt


def _point_key(overrides: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    def _norm(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, 6)
        return v

    return tuple((k, _norm(v)) for k, v in sorted(overrides.items(), key=lambda x: x[0]))


def build_grid_points(base_cfg: Dict[str, Any], grid_points: int) -> Tuple[List[GridPoint], List[str]]:
    """
    Focused mini-grid for module3 rescue:
    - prioritize lambda_sparse / graph-regularization / fusion gate / residual clip / dropout
    - keep prototype path off by default (handled in shared params)
    - default budget tuned so 2 datasets * 1 seed * 2 variants * 16 points ~= 64 runs
    """
    budget = max(1, int(grid_points))
    skip_msgs: List[str] = []

    base_sparse = float(base_cfg.get("lambda_sparse", 1.0) or 1.0)
    base_dropout = float(base_cfg.get("dropout", 0.0) or 0.0)
    base_gate_max = float(base_cfg.get("fusion_gate_max", 1.0) or 1.0)
    base_gate_bias = float(base_cfg.get("fusion_gate_bias_init", -1.1) or -1.1)
    base_clip_t = float(base_cfg.get("residual_clip_t", 2.0) or 2.0)
    base_hmax = float(base_cfg.get("graph_entropy_max", 0.85) or 0.85)
    base_diag = float(base_cfg.get("lambda_graph_diag", 0.10) or 0.10)
    base_uniform = float(base_cfg.get("lambda_graph_uniform", 0.04) or 0.04)
    base_margin = float(base_cfg.get("graph_uniform_margin", 0.10) or 0.10)
    base_warmup = int(base_cfg.get("graph_reg_warmup_epochs", 1) or 1)

    sparse_vals = _uniq_keep_order([base_sparse, 0.5, 0.3, 0.1, 0.03, 0.01, 0.003])
    gate_bias_vals = _uniq_keep_order([base_gate_bias, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0])
    gate_max_vals = _uniq_keep_order([base_gate_max, 0.3, 0.4, 0.5, 0.6, 0.7])
    clip_vals = _uniq_keep_order([base_clip_t, 1.5, 2.0, 2.5, 3.0, 4.0])
    graph_hmax_vals = _uniq_keep_order([base_hmax, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60])
    graph_diag_vals = _uniq_keep_order([base_diag, 0.08, 0.10, 0.12, 0.15])
    graph_uniform_vals = _uniq_keep_order([base_uniform, 0.02, 0.04, 0.06, 0.08])
    graph_margin_vals = _uniq_keep_order([base_margin, 0.08, 0.10, 0.12, 0.15])
    graph_warmup_vals = list(dict.fromkeys([base_warmup, 0, 1, 2]))

    dropout_vals: List[float] = [base_dropout]
    for delta in (0.05, 0.10, 0.15):
        cand = round(max(0.05, base_dropout - delta), 3)
        if cand + 1e-9 < base_dropout:
            dropout_vals.append(cand)
    dropout_vals = _uniq_keep_order(dropout_vals)
    if len(dropout_vals) == 1:
        skip_msgs.append(f"base dropout={base_dropout:.4f} already small; no lower-dropout points added.")

    points: List[GridPoint] = []
    seen: Set[Tuple[Tuple[str, Any], ...]] = set()

    def add_point(tag: str, overrides: Dict[str, Any]) -> None:
        key = _point_key(overrides)
        if key in seen:
            return
        seen.add(key)
        points.append(GridPoint(tag, overrides))

    # baseline
    add_point("baseline", {})

    # curated high-priority points (keep early order useful when budget is small)
    priority_points: List[Tuple[str, Dict[str, Any]]] = [
        ("prio_sparse_0p3", {"lambda_sparse": 0.3}),
        ("prio_hmax_0p7", {"graph_entropy_max": 0.7}),
        ("prio_gdiag_0p12", {"lambda_graph_diag": 0.12}),
        ("prio_gwarm_0", {"graph_reg_warmup_epochs": 0}),
        (
            "prio_s0p1_h0p7",
            {"lambda_sparse": 0.1, "graph_entropy_max": 0.7},
        ),
        (
            "prio_s0p1_h0p7_gd0p12",
            {"lambda_sparse": 0.1, "graph_entropy_max": 0.7, "lambda_graph_diag": 0.12},
        ),
        (
            "prio_s0p03_h0p6_gd0p15_gw0",
            {
                "lambda_sparse": 0.03,
                "graph_entropy_max": 0.6,
                "lambda_graph_diag": 0.15,
                "graph_reg_warmup_epochs": 0,
            },
        ),
        (
            "prio_s0p1_bm1p5",
            {"lambda_sparse": 0.1, "fusion_gate_bias_init": -1.5},
        ),
        (
            "prio_s0p03_bm1p1_g1p0",
            {"lambda_sparse": 0.03, "fusion_gate_bias_init": -1.1, "fusion_gate_max": 1.0},
        ),
    ]
    for tag, overrides in priority_points:
        add_point(tag, overrides)

    # single-axis points (high-priority diagnostics)
    for s in sparse_vals[1:]:
        add_point(f"sparse_{_fmt_float_tag(s)}", {"lambda_sparse": s})
    for h in graph_hmax_vals[1:]:
        add_point(f"hmax_{_fmt_float_tag(h)}", {"graph_entropy_max": h})
    for d in graph_diag_vals[1:]:
        add_point(f"gdiag_{_fmt_float_tag(d)}", {"lambda_graph_diag": d})
    for u in graph_uniform_vals[1:]:
        add_point(f"guni_{_fmt_float_tag(u)}", {"lambda_graph_uniform": u})
    for m in graph_margin_vals[1:]:
        add_point(f"gmargin_{_fmt_float_tag(m)}", {"graph_uniform_margin": m})
    for w in graph_warmup_vals:
        if int(w) == int(base_warmup):
            continue
        add_point(f"gwarm_{int(w)}", {"graph_reg_warmup_epochs": int(w)})
    for b in gate_bias_vals[1:]:
        add_point(f"gbias_{_fmt_float_tag(b)}", {"fusion_gate_bias_init": b})
    for g in gate_max_vals[1:]:
        add_point(f"gmax_{_fmt_float_tag(g)}", {"fusion_gate_max": g})
    for c in clip_vals[1:]:
        add_point(f"clip_{_fmt_float_tag(c)}", {"residual_clip_t": c})
    for d in dropout_vals[1:]:
        add_point(f"drop_{_fmt_float_tag(d)}", {"dropout": d})

    # pairwise points: sparse + graph range constraints
    for s in (0.5, 0.3, 0.1, 0.03):
        for h in (0.80, 0.70, 0.60):
            add_point(
                f"s{_fmt_float_tag(s)}_h{_fmt_float_tag(h)}",
                {"lambda_sparse": s, "graph_entropy_max": h},
            )
        for d in (0.10, 0.12, 0.15):
            add_point(
                f"s{_fmt_float_tag(s)}_gd{_fmt_float_tag(d)}",
                {"lambda_sparse": s, "lambda_graph_diag": d},
            )
        for w in (0, 1):
            add_point(
                f"s{_fmt_float_tag(s)}_gw{int(w)}",
                {"lambda_sparse": s, "graph_reg_warmup_epochs": int(w)},
            )

    # pairwise points: sparse + gate bias (ease conservative gate)
    for s in (0.3, 0.1, 0.03, 0.01):
        for b in (-2.5, -2.0, -1.5):
            add_point(
                f"s{_fmt_float_tag(s)}_b{_fmt_float_tag(b)}",
                {"lambda_sparse": s, "fusion_gate_bias_init": b},
            )

    # pairwise points: sparse + gate max (allow stronger residual if needed)
    for s in (0.3, 0.1, 0.03, 0.01):
        for g in (0.5, 0.6, 0.7):
            add_point(
                f"s{_fmt_float_tag(s)}_g{_fmt_float_tag(g)}",
                {"lambda_sparse": s, "fusion_gate_max": g},
            )

    # pairwise points: sparse + lower dropout (assist_09 rescue-oriented)
    for s in (0.3, 0.1, 0.03, 0.01):
        for d in dropout_vals[1:]:
            add_point(
                f"s{_fmt_float_tag(s)}_d{_fmt_float_tag(d)}",
                {"lambda_sparse": s, "dropout": d},
            )

    # pairwise points: sparse + clip threshold
    for s in (0.3, 0.1, 0.03, 0.01):
        for c in (1.5, 2.5, 3.0):
            add_point(
                f"s{_fmt_float_tag(s)}_c{_fmt_float_tag(c)}",
                {"lambda_sparse": s, "residual_clip_t": c},
            )

    # triple points: sparse + graph constraints + warmup
    graph_triples = [
        (0.3, 0.80, 0.10, 1),
        (0.1, 0.80, 0.10, 1),
        (0.1, 0.70, 0.12, 1),
        (0.1, 0.70, 0.12, 0),
        (0.03, 0.70, 0.12, 1),
        (0.03, 0.70, 0.12, 0),
        (0.03, 0.60, 0.15, 1),
        (0.03, 0.60, 0.15, 0),
    ]
    for s, h, d, w in graph_triples:
        add_point(
            f"s{_fmt_float_tag(s)}_h{_fmt_float_tag(h)}_gd{_fmt_float_tag(d)}_gw{int(w)}",
            {
                "lambda_sparse": s,
                "graph_entropy_max": h,
                "lambda_graph_diag": d,
                "graph_reg_warmup_epochs": int(w),
            },
        )

    # triple points: sparse + relaxed gate + stronger cap
    triple_candidates = [
        (0.1, -2.0, 0.5, 2.5),
        (0.03, -2.0, 0.6, 2.5),
        (0.01, -1.5, 0.6, 3.0),
        (0.03, -1.5, 0.5, 3.0),
        (0.3, -2.5, 0.5, 2.5),
        (0.1, -2.5, 0.6, 3.0),
        (0.01, -2.0, 0.7, 3.0),
        (0.03, -1.0, 0.7, 4.0),
    ]
    for s, b, g, c in triple_candidates:
        add_point(
            f"s{_fmt_float_tag(s)}_b{_fmt_float_tag(b)}_g{_fmt_float_tag(g)}_c{_fmt_float_tag(c)}",
            {
                "lambda_sparse": s,
                "fusion_gate_bias_init": b,
                "fusion_gate_max": g,
                "residual_clip_t": c,
            },
        )

    # Auto-fill stage:
    # If hand-crafted points are insufficient for a requested budget (e.g. 64),
    # expand with deterministic cartesian combinations until budget is reached.
    if len(points) < budget:
        for s, h, dreg, w, b, g, c, d in product(
            sparse_vals,
            graph_hmax_vals,
            graph_diag_vals,
            graph_warmup_vals,
            gate_bias_vals,
            gate_max_vals,
            clip_vals,
            dropout_vals,
        ):
            overrides: Dict[str, Any] = {}
            if abs(s - base_sparse) > 1e-9:
                overrides["lambda_sparse"] = s
            if abs(h - base_hmax) > 1e-9:
                overrides["graph_entropy_max"] = h
            if abs(dreg - base_diag) > 1e-9:
                overrides["lambda_graph_diag"] = dreg
            if int(w) != int(base_warmup):
                overrides["graph_reg_warmup_epochs"] = int(w)
            if abs(b - base_gate_bias) > 1e-9:
                overrides["fusion_gate_bias_init"] = b
            if abs(g - base_gate_max) > 1e-9:
                overrides["fusion_gate_max"] = g
            if abs(c - base_clip_t) > 1e-9:
                overrides["residual_clip_t"] = c
            if abs(d - base_dropout) > 1e-9:
                overrides["dropout"] = d

            if not overrides:
                continue

            tag_parts: List[str] = ["auto"]
            if "lambda_sparse" in overrides:
                tag_parts.append(f"s{_fmt_float_tag(overrides['lambda_sparse'])}")
            if "graph_entropy_max" in overrides:
                tag_parts.append(f"h{_fmt_float_tag(overrides['graph_entropy_max'])}")
            if "lambda_graph_diag" in overrides:
                tag_parts.append(f"gd{_fmt_float_tag(overrides['lambda_graph_diag'])}")
            if "graph_reg_warmup_epochs" in overrides:
                tag_parts.append(f"gw{int(overrides['graph_reg_warmup_epochs'])}")
            if "fusion_gate_bias_init" in overrides:
                tag_parts.append(f"b{_fmt_float_tag(overrides['fusion_gate_bias_init'])}")
            if "fusion_gate_max" in overrides:
                tag_parts.append(f"g{_fmt_float_tag(overrides['fusion_gate_max'])}")
            if "residual_clip_t" in overrides:
                tag_parts.append(f"c{_fmt_float_tag(overrides['residual_clip_t'])}")
            if "dropout" in overrides:
                tag_parts.append(f"d{_fmt_float_tag(overrides['dropout'])}")
            add_point("_".join(tag_parts), overrides)

            if len(points) >= budget:
                break

    if len(points) > budget:
        skip_msgs.append(f"generated {len(points)} points, truncating to budget={budget}.")
        points = points[:budget]
    elif len(points) < budget:
        skip_msgs.append(f"generated {len(points)} unique points (< budget={budget}).")

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
    params.setdefault("fusion_gate_max", 1.0)
    params.setdefault("fusion_gate_bias_init", -1.1)
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
    debug_diag = bool(job.params.get("debug_module3_diag", True))
    diag_batches = max(1, int(job.params.get("diag_batches", 2)))
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
    if debug_diag:
        cmd.extend(["--debug_module3_diag", "--diag_batches", str(diag_batches)])

    skip_keys = {
        "dataset_name",
        "model_variant",
        "save_dir",
        "log_dir",
        "seed",
        "generate_diagnosis",
        "gpus",
        "debug_module3_diag",
        "diag_batches",
    }
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


def load_completed_job_keys(path: Path) -> Set[Tuple[str, int, str, str]]:
    done: Set[Tuple[str, int, str, str]] = set()
    if not path.exists():
        return done

    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("status", "")).strip().lower()
            if status != "ok":
                continue
            dataset = row.get("dataset")
            seed_raw = row.get("seed")
            grid_tag = row.get("grid_tag")
            variant_raw = row.get("variant")
            variant = variant_internal_name(variant_raw)
            if not dataset or seed_raw is None or not grid_tag or variant is None:
                continue
            try:
                seed = int(seed_raw)
            except Exception:
                continue
            done.add(make_job_key(dataset, seed, grid_tag, variant))
    return done


def collect_result(job: JobSpec, exit_code: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": job.dataset,
        "seed": job.seed,
        "grid_tag": job.grid_tag,
        "variant": variant_csv_name(job.variant),
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
    grid_points: int,
    main_arg_dests: Set[str],
    completed_keys: Optional[Set[Tuple[str, int, str, str]]] = None,
    skip_completed: bool = True,
) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    completed_keys = completed_keys or set()
    skipped_done = 0
    skipped_ckpt = 0

    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' is missing in BEST_CFG.")
        base_cfg = dict(BEST_CFG[dataset])
        grid_list, skip_msgs = build_grid_points(base_cfg, grid_points=grid_points)
        for msg in skip_msgs:
            print(f"[GRID-SKIP] dataset={dataset}: {msg}")

        for seed in seeds:
            for gp in grid_list:
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

                    job_key = make_job_key(dataset, int(seed), gp.tag, variant)
                    if skip_completed and job_key in completed_keys:
                        skipped_done += 1
                        continue
                    # Fallback skip: successful checkpoint exists even if CSV row is missing.
                    if skip_completed:
                        test_json = read_json(save_dir / "test_results.json")
                        metrics = test_json.get("metrics", {}) if isinstance(test_json, dict) else {}
                        if isinstance(metrics, dict) and metrics.get("auc") is not None:
                            skipped_ckpt += 1
                            completed_keys.add(job_key)
                            continue

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
    if skip_completed and (skipped_done > 0 or skipped_ckpt > 0):
        print(
            f"[RESUME] skipped completed jobs: from_csv={skipped_done}, "
            f"from_checkpoint={skipped_ckpt}"
        )
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
    completed_keys = set() if args.rerun_existing else load_completed_job_keys(RESULT_CSV)

    jobs = make_jobs(
        datasets=datasets,
        seeds=seeds,
        variants=variants,
        epochs_override=epochs_override,
        grid_points=args.grid_points,
        main_arg_dests=main_arg_dests,
        completed_keys=completed_keys,
        skip_completed=not args.rerun_existing,
    )

    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Variants: {variants}")
    print(f"Grid points per dataset: {max(1, int(args.grid_points))}")
    print(
        f"GPUs: {gpus}, max_concurrent={args.max_concurrent}, "
        f"max_per_gpu={max_per_gpu}, effective_max={max_concurrent}"
    )
    print(f"Jobs: {len(jobs)} (approx target = datasets*seeds*grid_points*variants)")
    print(f"Result CSV: {RESULT_CSV}")
    print(f"Dry run: {args.dry_run}")
    print(f"Rerun existing: {args.rerun_existing}")

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
