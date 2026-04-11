#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_abce_ablation.py

Historical name kept for compatibility, but the current runner supports
fine-grained ABDE ablation:

Component mapping:
- A: global concept graph      -> --ablate_concept_graph
- B: MF/Q residual branch      -> --ablate_skill_encoder
- D: 2PL-IRT diagnosis head    -> --ablate_module2
- E: personal graph mixing     -> --use_personal_graph (off by dropping this flag)

Outputs:
- results/abce_ablation_diagnosis.csv
- results/abce_ablation_summary.csv
- results/abce_ablation_summary_mean.csv

Profiles:
- best: baseline best config
- b_rescue: conservative B rescue
- e_rescue: warmup + anti-collapse E rescue
- all_rescue: combine B/E rescue knobs
"""

from __future__ import annotations

import argparse
import csv
import io
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


RESULT_CSV = Path("results") / "abce_ablation_diagnosis.csv"
SUMMARY_CSV = Path("results") / "abce_ablation_summary.csv"
MEAN_SUMMARY_CSV = Path("results") / "abce_ablation_summary_mean.csv"
NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
ROW_START_RE = re.compile(
    r'(?<![\r\n])(?=(assist_09|assist_17|junyi),\d+,(best|ae_dominant|b_rescue|e_rescue|all_rescue),(full|no_A|no_B|no_D|no_E|no_AE|B_q_only|B_no_q),)'
)


@dataclass
class AblationSpec:
    name: str
    flags: Dict[str, Any]
    overrides: Dict[str, Any]
    drop_keys: Tuple[str, ...] = ()


@dataclass
class JobSpec:
    dataset: str
    seed: int
    profile: str
    ablation: AblationSpec
    model_variant: str
    save_dir: Path
    log_dir: Path
    params: Dict[str, Any]
    cmd: List[str]


BASE_SINGLE_ABLATIONS: Tuple[AblationSpec, ...] = (
    AblationSpec(name="full", flags={}, overrides={}),
    AblationSpec(name="no_A", flags={"ablate_concept_graph": True}, overrides={"num_gnn_layers": 0}),
    AblationSpec(
        name="no_E",
        flags={},
        overrides={"lambda_sparse_personal": 0.0, "lambda_alpha": 0.0},
        drop_keys=("use_personal_graph",),
    ),
    AblationSpec(name="no_B", flags={"ablate_skill_encoder": True}, overrides={}),
    AblationSpec(name="no_D", flags={"ablate_module2": True}, overrides={}),
)

BASE_SINGLE_PLUS_EXTRA: Tuple[AblationSpec, ...] = (
    AblationSpec(
        name="no_AE",
        flags={"ablate_concept_graph": True},
        overrides={"num_gnn_layers": 0, "lambda_sparse_personal": 0.0, "lambda_alpha": 0.0},
        drop_keys=("use_personal_graph",),
    ),
    AblationSpec(name="B_q_only", flags={"disable_b_id_adapter": True, "disable_b_bias": True}, overrides={}),
    AblationSpec(name="B_no_q", flags={"disable_q_conditioning": True}, overrides={}),
)


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
        "alpha_bias_std": None,
        "gate_mean": None,
        "delta_over_irt": None,
        "mf_abs_mean": None,
        "irt_abs_mean": None,
        "q_interaction_abs_mean": None,
        "id_adapter_abs_mean": None,
        "bias_abs_mean": None,
        "B_q_share": None,
        "B_id_share": None,
        "B_bias_share": None,
        "student_q_norm": None,
        "student_id_adapter_norm": None,
        "item_q_norm": None,
        "item_id_adapter_norm": None,
        "personal_matrix_delta": None,
        "personal_matrix_student_std": None,
        "personal_delta_pre_softmax_norm": None,
        "personal_delta_student_std": None,
        "alpha_head_std": None,
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
                    "alpha_bias_std",
                    "gate_mean",
                    "delta_over_irt",
                    "mf_abs_mean",
                    "irt_abs_mean",
                    "q_interaction_abs_mean",
                    "id_adapter_abs_mean",
                    "bias_abs_mean",
                    "B_q_share",
                    "B_id_share",
                    "B_bias_share",
                    "student_q_norm",
                    "student_id_adapter_norm",
                    "item_q_norm",
                    "item_id_adapter_norm",
                    "personal_matrix_delta",
                    "personal_matrix_student_std",
                    "personal_delta_pre_softmax_norm",
                    "personal_delta_student_std",
                    "alpha_head_std",
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

def pick_base_ablations(component_set: str) -> List[AblationSpec]:
    if component_set == "single":
        return list(BASE_SINGLE_ABLATIONS)
    if component_set == "single_plus":
        return list(BASE_SINGLE_ABLATIONS) + list(BASE_SINGLE_PLUS_EXTRA)
    raise ValueError(f"Unknown component_set: {component_set}")


def build_command(job: JobSpec, generate_diagnosis: bool) -> List[str]:
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
        "True" if generate_diagnosis else "False",
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

    for k, v in job.ablation.flags.items():
        append_arg(cmd, k, v)

    return cmd


def collect_result(job: JobSpec, exit_code: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": job.dataset,
        "seed": job.seed,
        "profile": job.profile,
        "ablation": job.ablation.name,
        "model_variant": job.model_variant,
        "status": "failed",
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

    hist = read_json(job.save_dir / "training_history.json") or {}
    row["best_epoch"] = hist.get("best_epoch") if isinstance(hist, dict) else None

    args_json = read_json(job.save_dir / "args.json") or {}
    for key in (
        "enable_module1",
        "enable_module2",
        "enable_module3",
        "use_concept_graph",
        "use_personal_graph",
        "use_mf_branch",
    ):
        row[f"effective_{key}"] = args_json.get(key)

    log_file = latest_train_log(job.log_dir)
    row["log_file"] = str(log_file) if log_file else ""
    row.update(parse_log_metrics(log_file))

    if exit_code == 0:
        row["status"] = "ok"
    elif row.get("test_auc") is not None:
        row["status"] = "metrics_ok"

    row["params_json"] = json.dumps(job.params, ensure_ascii=False, sort_keys=True)
    row["flags_json"] = json.dumps(job.ablation.flags, ensure_ascii=False, sort_keys=True)
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
        "effective_enable_module1",
        "effective_enable_module2",
        "effective_enable_module3",
        "effective_use_concept_graph",
        "effective_use_personal_graph",
        "effective_use_mf_branch",
        "reg_bce_ratio",
        "graph_entropy_ratio",
        "alpha_std",
        "alpha_bias_std",
        "gate_mean",
        "delta_over_irt",
        "mf_abs_mean",
        "irt_abs_mean",
        "q_interaction_abs_mean",
        "id_adapter_abs_mean",
        "bias_abs_mean",
        "B_q_share",
        "B_id_share",
        "B_bias_share",
        "student_q_norm",
        "student_id_adapter_norm",
        "item_q_norm",
        "item_id_adapter_norm",
        "personal_matrix_delta",
        "personal_matrix_student_std",
        "personal_delta_pre_softmax_norm",
        "personal_delta_student_std",
        "alpha_head_std",
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

    if not need_header and path.stat().st_size > 0:
        with open(path, "rb") as fb:
            fb.seek(-1, os.SEEK_END)
            last_byte = fb.read(1)
        if last_byte not in (b"\n", b"\r"):
            with open(path, "a", encoding="utf-8", newline="") as f:
                f.write("\n")

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _repair_glued_csv_rows(text: str) -> Tuple[str, int]:
    repaired, count = ROW_START_RE.subn("\n", text)
    return repaired, int(count)


def load_result_rows(path: Path, run_id: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        text = f.read()
    repaired_text, repaired_count = _repair_glued_csv_rows(text)
    if repaired_count > 0:
        print(f"[WARN] Auto-repaired {repaired_count} glued row boundary/boundaries in {path}.")
    for row in csv.DictReader(io.StringIO(repaired_text)):
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


def classify_delta(delta: Optional[float], threshold: float = 0.003) -> str:
    if delta is None:
        return "untested"
    if delta > threshold:
        return "useful"
    if delta < -threshold:
        return "harmful_or_unstable"
    return "neutral"


def _first_non_none(rows: Sequence[Dict[str, Any]], key: str) -> Any:
    for r in rows:
        if key in r and r[key] not in ("", None):
            return r[key]
    return None


def _majority_value(rows: Sequence[Dict[str, Any]], key: str) -> Any:
    counter: Dict[Any, int] = {}
    for r in rows:
        v = r.get(key)
        if v in ("", None):
            continue
        counter[v] = counter.get(v, 0) + 1
    if not counter:
        return _first_non_none(rows, key)
    return max(counter.items(), key=lambda kv: kv[1])[0]


def diagnose_reason(
    full_row: Dict[str, Any],
    delta_a: Optional[float],
    delta_b: Optional[float],
    delta_d: Optional[float],
    delta_e: Optional[float],
) -> str:
    reasons: List[str] = []
    ger = try_float(full_row.get("graph_entropy_ratio"))
    alpha_std = try_float(full_row.get("alpha_std"))
    alpha_bias_std = try_float(full_row.get("alpha_bias_std"))
    gate = try_float(full_row.get("gate_mean"))
    dor = try_float(full_row.get("delta_over_irt"))
    b_q_share = try_float(full_row.get("B_q_share"))
    b_id_share = try_float(full_row.get("B_id_share"))
    personal_matrix_delta = try_float(full_row.get("personal_matrix_delta"))
    personal_matrix_student_std = try_float(full_row.get("personal_matrix_student_std"))
    personal_delta_pre_softmax_norm = try_float(full_row.get("personal_delta_pre_softmax_norm"))
    personal_delta_student_std = try_float(full_row.get("personal_delta_student_std"))
    alpha_head_std = try_float(full_row.get("alpha_head_std"))

    if ger is not None and ger > 0.98:
        reasons.append("graph-uniform-risk")
    if alpha_std is not None and alpha_std < 1e-6:
        reasons.append("personal-alpha-collapse")
    if alpha_bias_std is not None and alpha_bias_std < 1e-6:
        reasons.append("personal-bias-collapse")
    if gate is not None and gate < 0.5:
        reasons.append("mf-gate-low")
    if gate is not None and gate > 0.9:
        reasons.append("mf-gate-very-high")
    if dor is not None and dor < 0.05:
        reasons.append("residual-delta-low")
    if b_q_share is not None and b_q_share < 0.45:
        reasons.append("B-q-share-low")
    if b_id_share is not None and b_id_share > 0.35:
        reasons.append("B-id-share-high")
    if personal_matrix_delta is not None and personal_matrix_delta < 0.01:
        reasons.append("personal-matrix-delta-low")
    if personal_matrix_student_std is not None and personal_matrix_student_std < 0.001:
        reasons.append("personal-matrix-student-flat")
    if personal_delta_pre_softmax_norm is not None and personal_delta_pre_softmax_norm < 0.01:
        reasons.append("personal-delta-norm-low")
    if personal_delta_student_std is not None and personal_delta_student_std < 0.001:
        reasons.append("personal-delta-student-flat")
    if alpha_head_std is not None and alpha_head_std < 0.001:
        reasons.append("alpha-head-collapse")

    if delta_a is not None and abs(delta_a) < 0.002:
        reasons.append("A-delta-small")
    if delta_b is not None and abs(delta_b) < 0.002:
        reasons.append("B-delta-small")
    if delta_d is not None and abs(delta_d) < 0.002:
        reasons.append("D-delta-small")
    if delta_e is not None and abs(delta_e) < 0.002:
        reasons.append("E-delta-small")

    return ";".join(reasons)


def write_summary(
    path: Path,
    mean_path: Path,
    rows: List[Dict[str, Any]],
    threshold: float = 0.003,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        if str(r.get("status", "")).lower() not in {"ok", "metrics_ok"}:
            continue
        key = (str(r.get("dataset", "")), str(r.get("seed", "")), str(r.get("profile", "")))
        grouped.setdefault(key, {})
        grouped[key][str(r.get("ablation", ""))] = r

    summary_rows: List[Dict[str, Any]] = []
    for (dataset, seed, profile), mp in sorted(grouped.items()):
        full = mp.get("full")
        if full is None:
            continue

        full_auc = try_float(full.get("test_auc"))
        no_a_auc = try_float(mp.get("no_A", {}).get("test_auc"))
        no_b_auc = try_float(mp.get("no_B", {}).get("test_auc"))
        no_d_auc = try_float(mp.get("no_D", {}).get("test_auc"))
        no_e_auc = try_float(mp.get("no_E", {}).get("test_auc"))

        delta_a = (full_auc - no_a_auc) if (full_auc is not None and no_a_auc is not None) else None
        delta_b = (full_auc - no_b_auc) if (full_auc is not None and no_b_auc is not None) else None
        delta_d = (full_auc - no_d_auc) if (full_auc is not None and no_d_auc is not None) else None
        delta_e = (full_auc - no_e_auc) if (full_auc is not None and no_e_auc is not None) else None

        comp_state = {
            "A": classify_delta(delta_a, threshold),
            "B": classify_delta(delta_b, threshold),
            "D": classify_delta(delta_d, threshold),
            "E": classify_delta(delta_e, threshold),
        }
        keep = [k for k, v in comp_state.items() if v == "useful"]
        drop = [k for k, v in comp_state.items() if v == "harmful_or_unstable"]
        tune = [k for k, v in comp_state.items() if v == "neutral"]

        row = {
            "dataset": dataset,
            "seed": seed,
            "profile": profile,
            "full_auc": full_auc,
            "no_A_auc": no_a_auc,
            "no_B_auc": no_b_auc,
            "no_D_auc": no_d_auc,
            "no_E_auc": no_e_auc,
            "delta_A_full_minus_noA": delta_a,
            "delta_B_full_minus_noB": delta_b,
            "delta_D_full_minus_noD": delta_d,
            "delta_E_full_minus_noE": delta_e,
            "full_effective_use_concept_graph": full.get("effective_use_concept_graph"),
            "full_effective_use_personal_graph": full.get("effective_use_personal_graph"),
            "full_effective_use_mf_branch": full.get("effective_use_mf_branch"),
            "full_graph_entropy_ratio": try_float(full.get("graph_entropy_ratio")),
            "full_alpha_std": try_float(full.get("alpha_std")),
            "full_alpha_bias_std": try_float(full.get("alpha_bias_std")),
            "full_gate_mean": try_float(full.get("gate_mean")),
            "full_delta_over_irt": try_float(full.get("delta_over_irt")),
            "full_B_q_share": try_float(full.get("B_q_share")),
            "full_B_id_share": try_float(full.get("B_id_share")),
            "full_B_bias_share": try_float(full.get("B_bias_share")),
            "full_q_interaction_abs_mean": try_float(full.get("q_interaction_abs_mean")),
            "full_id_adapter_abs_mean": try_float(full.get("id_adapter_abs_mean")),
            "full_bias_abs_mean": try_float(full.get("bias_abs_mean")),
            "full_personal_matrix_delta": try_float(full.get("personal_matrix_delta")),
            "full_personal_matrix_student_std": try_float(full.get("personal_matrix_student_std")),
            "full_personal_delta_pre_softmax_norm": try_float(full.get("personal_delta_pre_softmax_norm")),
            "full_personal_delta_student_std": try_float(full.get("personal_delta_student_std")),
            "full_alpha_head_std": try_float(full.get("alpha_head_std")),
            "full_warn_graph_uniform_count": full.get("warn_graph_uniform_count"),
            "full_warn_alpha_collapse_count": full.get("warn_alpha_collapse_count"),
            "full_warn_module3_count": full.get("warn_module3_count"),
            "full_module_activity_epoch10": full.get("module_activity_epoch10", ""),
            "state_A": comp_state["A"],
            "state_B": comp_state["B"],
            "state_D": comp_state["D"],
            "state_E": comp_state["E"],
            "suggest_keep": ",".join(keep),
            "suggest_drop": ",".join(drop),
            "suggest_tune": ",".join(tune),
            "diagnosis_reason": diagnose_reason(full, delta_a, delta_b, delta_d, delta_e),
        }
        summary_rows.append(row)

    summary_fields = [
        "dataset",
        "seed",
        "profile",
        "full_auc",
        "no_A_auc",
        "no_B_auc",
        "no_D_auc",
        "no_E_auc",
        "delta_A_full_minus_noA",
        "delta_B_full_minus_noB",
        "delta_D_full_minus_noD",
        "delta_E_full_minus_noE",
        "full_effective_use_concept_graph",
        "full_effective_use_personal_graph",
        "full_effective_use_mf_branch",
        "full_graph_entropy_ratio",
        "full_alpha_std",
        "full_alpha_bias_std",
        "full_gate_mean",
        "full_delta_over_irt",
        "full_B_q_share",
        "full_B_id_share",
        "full_B_bias_share",
        "full_q_interaction_abs_mean",
        "full_id_adapter_abs_mean",
        "full_bias_abs_mean",
        "full_personal_matrix_delta",
        "full_personal_matrix_student_std",
        "full_personal_delta_pre_softmax_norm",
        "full_personal_delta_student_std",
        "full_alpha_head_std",
        "full_warn_graph_uniform_count",
        "full_warn_alpha_collapse_count",
        "full_warn_module3_count",
        "full_module_activity_epoch10",
        "state_A",
        "state_B",
        "state_D",
        "state_E",
        "suggest_keep",
        "suggest_drop",
        "suggest_tune",
        "diagnosis_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    mean_rows: List[Dict[str, Any]] = []
    mean_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in summary_rows:
        mean_group.setdefault((r["dataset"], r["profile"]), []).append(r)

    for (dataset, profile), grp in sorted(mean_group.items()):
        n = len(grp)
        deltas_a = [try_float(x.get("delta_A_full_minus_noA")) for x in grp]
        deltas_b = [try_float(x.get("delta_B_full_minus_noB")) for x in grp]
        deltas_d = [try_float(x.get("delta_D_full_minus_noD")) for x in grp]
        deltas_e = [try_float(x.get("delta_E_full_minus_noE")) for x in grp]
        full_aucs = [try_float(x.get("full_auc")) for x in grp]

        def mean_valid(vals: List[Optional[float]]) -> Optional[float]:
            arr = [v for v in vals if v is not None]
            if not arr:
                return None
            return float(sum(arr) / len(arr))

        def useful_ratio(vals: List[Optional[float]]) -> Optional[float]:
            arr = [v for v in vals if v is not None]
            if not arr:
                return None
            return float(sum(1 for v in arr if v > threshold) / len(arr))

        mean_a = mean_valid(deltas_a)
        mean_b = mean_valid(deltas_b)
        mean_d = mean_valid(deltas_d)
        mean_e = mean_valid(deltas_e)
        keep = []
        if mean_a is not None and mean_a > threshold:
            keep.append("A")
        if mean_b is not None and mean_b > threshold:
            keep.append("B")
        if mean_d is not None and mean_d > threshold:
            keep.append("D")
        if mean_e is not None and mean_e > threshold:
            keep.append("E")

        row = {
            "dataset": dataset,
            "profile": profile,
            "n_seeds": n,
            "full_auc_mean": mean_valid(full_aucs),
            "delta_A_mean": mean_a,
            "delta_B_mean": mean_b,
            "delta_D_mean": mean_d,
            "delta_E_mean": mean_e,
            "A_useful_seed_ratio": useful_ratio(deltas_a),
            "B_useful_seed_ratio": useful_ratio(deltas_b),
            "D_useful_seed_ratio": useful_ratio(deltas_d),
            "E_useful_seed_ratio": useful_ratio(deltas_e),
            "suggest_keep_by_mean": ",".join(keep),
            "state_A_majority": _majority_value(grp, "state_A"),
            "state_B_majority": _majority_value(grp, "state_B"),
            "state_D_majority": _majority_value(grp, "state_D"),
            "state_E_majority": _majority_value(grp, "state_E"),
        }
        mean_rows.append(row)

    mean_fields = [
        "dataset",
        "profile",
        "n_seeds",
        "full_auc_mean",
        "delta_A_mean",
        "delta_B_mean",
        "delta_D_mean",
        "delta_E_mean",
        "A_useful_seed_ratio",
        "B_useful_seed_ratio",
        "D_useful_seed_ratio",
        "E_useful_seed_ratio",
        "suggest_keep_by_mean",
        "state_A_majority",
        "state_B_majority",
        "state_D_majority",
        "state_E_majority",
    ]
    with open(mean_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mean_fields)
        writer.writeheader()
        for r in mean_rows:
            writer.writerow(r)

    return summary_rows, mean_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B/D/E sub-module ablation with diagnosis summary.")
    parser.add_argument("--datasets", type=str, default="assist_09,junyi")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds. Default uses DEFAULT_SEEDS.")
    parser.add_argument(
        "--profiles",
        type=str,
        default=None,
        help="Optional filter: best,ae_dominant,b_rescue,e_rescue,all_rescue",
    )
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_interval", type=int, default=10)
    parser.add_argument("--run_id", type=str, default=None, help="Output namespace. Default=timestamp.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--analyze_only", action="store_true", help="Skip running and only summarize existing rows.")
    parser.add_argument("--generate_diagnosis", action="store_true", help="Enable post-training diagnosis artifacts.")
    parser.add_argument("--component_set", type=str, default="single", choices=["single", "single_plus"])
    parser.add_argument("--ablations", type=str, default=None, help="Optional filter: full,no_A,no_B,no_D,no_E,no_AE,B_q_only,B_no_q")
    parser.add_argument("--delta_threshold", type=float, default=0.003, help="Threshold for useful/neutral/harmful labels.")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    return parser.parse_args()


def _profile_overrides(profile: str, dataset: str, args: argparse.Namespace) -> Dict[str, Any]:
    if profile == "best":
        return {}
    if profile == "ae_dominant":
        if dataset == "assist_09":
            return {
                "fusion_gate_max": 0.35,
                "fusion_gate_bias_init": -2.2,
                "residual_scale_init": 0.08,
                "graph_identity_residual": 0.25,
                "mf_warmup_epochs": 5,
                "lambda_delta_ratio": 0.08,
                "delta_ratio_target": 0.12,
                "lambda_b_id_budget": 0.06,
                "b_id_budget_target": 0.20,
                "personal_warmup_epochs": 8,
                "personal_max_alpha": 0.45,
                "personal_delta_scale": 1.10,
                "lambda_alpha_min": 0.05,
                "alpha_min_target": 0.02,
            }
        return {
            "fusion_gate_max": 0.25,
            "fusion_gate_bias_init": -2.4,
            "residual_scale_init": 0.06,
            "graph_identity_residual": 0.10,
            "mf_warmup_epochs": 6,
            "lambda_delta_ratio": 0.08,
            "delta_ratio_target": 0.08,
            "lambda_b_id_budget": 0.06,
            "b_id_budget_target": 0.18,
            "personal_warmup_epochs": 6,
            "personal_max_alpha": 0.40,
            "personal_delta_scale": 1.05,
            "lambda_alpha_min": 0.05,
            "alpha_min_target": 0.02,
        }
    if profile == "b_rescue":
        if dataset == "assist_09":
            return {
                "mf_warmup_epochs": 4,
                "lambda_delta_ratio": 0.0,
                "delta_ratio_target": 0.20,
                "fusion_gate_max": 1.0,
                "fusion_gate_bias_init": -1.05,
                "residual_scale_init": 0.24,
            }
        return {
            "mf_warmup_epochs": 5,
            "lambda_delta_ratio": 0.0,
            "delta_ratio_target": 0.15,
            "fusion_gate_max": 0.85,
            "fusion_gate_bias_init": -1.4,
            "residual_scale_init": 0.18,
        }
    if profile == "e_rescue":
        return {
            "use_personal_graph": True,
            "personal_max_alpha": 0.45,
            "personal_delta_scale": 1.00,
            "personal_warmup_epochs": 10,
            "lambda_alpha_min": 0.08,
            "alpha_min_target": 0.03,
        }
    if profile == "all_rescue":
        merged: Dict[str, Any] = {}
        for rescue_name in ("b_rescue", "e_rescue"):
            merged.update(_profile_overrides(rescue_name, dataset, args))
        return merged
    raise ValueError(f"Unknown profile '{profile}'.")


def _selected_profiles(all_profiles: List[str], filters: Optional[str]) -> List[str]:
    if not filters:
        return all_profiles
    names = set(parse_csv_tokens(filters))
    chosen = [p for p in all_profiles if p in names]
    missing = names - set(all_profiles)
    if missing:
        raise ValueError(f"Unknown profile names: {sorted(missing)}")
    if not chosen:
        raise ValueError("No profiles selected after filtering.")
    return chosen


def _selected_ablations(all_abls: List[AblationSpec], filters: Optional[str]) -> List[AblationSpec]:
    if not filters:
        return all_abls
    names = set(parse_csv_tokens(filters))
    chosen = [a for a in all_abls if a.name in names]
    missing = names - set(a.name for a in all_abls)
    if missing:
        raise ValueError(f"Unknown ablation names: {sorted(missing)}")
    if not chosen:
        raise ValueError("No ablations selected after filtering.")
    return chosen


def make_jobs(args: argparse.Namespace, run_id: str) -> List[JobSpec]:
    datasets = parse_csv_tokens(args.datasets)
    seeds = list(DEFAULT_SEEDS) if args.seeds is None else parse_int_csv(args.seeds)
    base_abls = pick_base_ablations(args.component_set)
    base_abls = _selected_ablations(base_abls, args.ablations)

    jobs: List[JobSpec] = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not found in BEST_CFG.")
        base_cfg = dict(BEST_CFG[dataset])
        profiles = ["best", "ae_dominant", "b_rescue", "e_rescue", "all_rescue"]
        profiles = _selected_profiles(profiles, args.profiles)

        for profile in profiles:
            profile_cfg = _profile_overrides(profile, dataset, args)
            profile_abls = list(base_abls)

            for ab in profile_abls:
                for seed in seeds:
                    params = dict(base_cfg)
                    params.update(profile_cfg)
                    params.update(ab.overrides)
                    params["debug_module3_diag"] = True
                    params["diag_batches"] = 1

                    if args.epochs is not None:
                        params["epochs"] = int(args.epochs)
                    if args.early_stop_patience is not None:
                        params["early_stop_patience"] = int(args.early_stop_patience)
                    if args.learning_rate is not None:
                        params["learning_rate"] = float(args.learning_rate)

                    for k in set(ab.drop_keys) | set(ab.flags.keys()):
                        if k in params:
                            del params[k]

                    variant = f"{dataset}_abce_{profile}_{ab.name}"
                    save_dir = Path("checkpoints") / "abce_diag" / run_id / dataset / f"seed{seed}" / f"{profile}_{ab.name}"
                    log_dir = Path("logs") / "abce_diag" / run_id / dataset / f"seed{seed}" / f"{profile}_{ab.name}"

                    if (not args.rerun_existing) and (save_dir / "test_results.json").exists():
                        continue

                    save_dir.mkdir(parents=True, exist_ok=True)
                    log_dir.mkdir(parents=True, exist_ok=True)

                    job = JobSpec(
                        dataset=dataset,
                        seed=int(seed),
                        profile=profile,
                        ablation=ab,
                        model_variant=variant,
                        save_dir=save_dir,
                        log_dir=log_dir,
                        params=params,
                        cmd=[],
                    )
                    job.cmd = build_command(job, generate_diagnosis=bool(args.generate_diagnosis))
                    jobs.append(job)
    return jobs


def print_job(job: JobSpec, gpu_id: int) -> None:
    print(
        f"[PLAN] dataset={job.dataset} seed={job.seed} profile={job.profile} "
        f"ablation={job.ablation.name} gpu={gpu_id}"
    )
    print("       CMD:", " ".join(job.cmd))


def run_jobs(args: argparse.Namespace, jobs: Sequence[JobSpec]) -> None:
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
                f"ablation={job.ablation.name} exit={ret} auc={row.get('test_auc')}"
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
        summary_rows, mean_rows = write_summary(
            SUMMARY_CSV,
            MEAN_SUMMARY_CSV,
            rows,
            threshold=float(args.delta_threshold),
        )
        print(f"[ANALYZE-ONLY] rows={len(rows)}, summary_rows={len(summary_rows)}, mean_rows={len(mean_rows)}")
        print(f"Diagnosis CSV: {RESULT_CSV}")
        print(f"Summary CSV:   {SUMMARY_CSV}")
        print(f"Mean CSV:      {MEAN_SUMMARY_CSV}")
        return

    jobs = make_jobs(args, run_id=run_id)
    print(f"Run ID: {run_id}")
    print(f"Jobs: {len(jobs)}")
    print(f"Diagnosis CSV: {RESULT_CSV}")
    print(f"Summary CSV:   {SUMMARY_CSV}")
    print(f"Mean CSV:      {MEAN_SUMMARY_CSV}")

    if jobs:
        run_jobs(args, jobs)
    else:
        print("[INFO] No new jobs to run (all checkpoints exist or filters excluded all jobs).")

    rows = load_result_rows(RESULT_CSV, run_id=run_id)
    summary_rows, mean_rows = write_summary(
        SUMMARY_CSV,
        MEAN_SUMMARY_CSV,
        rows,
        threshold=float(args.delta_threshold),
    )
    print(f"[SUMMARY] rows={len(rows)}, summary_rows={len(summary_rows)}, mean_rows={len(mean_rows)}")
    for r in summary_rows:
        print(
            f"[SUMMARY] dataset={r['dataset']} seed={r['seed']} profile={r['profile']} "
            f"dA={r['delta_A_full_minus_noA']} dB={r['delta_B_full_minus_noB']} "
            f"dD={r['delta_D_full_minus_noD']} "
            f"dE={r['delta_E_full_minus_noE']} "
            f"keep={r['suggest_keep']} reason={r['diagnosis_reason']}"
        )


if __name__ == "__main__":
    main()
