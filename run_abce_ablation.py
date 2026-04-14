#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_abce_ablation.py

历史文件名保留，但当前 runner 已收缩为 AE-only：
- `full`：A + E + 固定预测头 D
- `no_A`：关闭全局概念图 A
- `no_E`：关闭个性化图 E
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
    r"(?<![\r\n])(?=(assist_09|assist_17|junyi),\d+,(best|ae_dominant),(full|no_A|no_E),)"
)
BOOLEAN_OPTIONAL_KEYS = {
    "use_personal_graph",
    "personal_disable_direct_bias",
    "personal_disable_student_global_context",
    "personal_support_only",
    "share_concept_embeddings",
}
STRUCTURAL_SWITCH_KEYS = (
    "share_concept_embeddings",
    "personal_disable_direct_bias",
    "personal_direct_bias_scale",
    "personal_alpha_bias_scale",
    "personal_reg_warmup_epochs",
    "personal_disable_student_global_context",
    "personal_local_hops",
    "personal_support_only",
    "use_personal_graph",
    "use_concept_graph",
    "graph_identity_residual",
    "graph_propagation_alpha",
    "graph_readout_1hop_scale",
    "graph_readout_2hop_scale",
    "personal_delta_scale",
    "personal_warmup_epochs",
    "lambda_alpha_min",
    "alpha_min_target",
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
    AblationSpec(name="no_A", flags={"ablate_concept_graph": True}, overrides={}),
    AblationSpec(
        name="no_E",
        flags={},
        overrides={
            "use_personal_graph": False,
            "lambda_sparse_personal": 0.0,
            "lambda_alpha": 0.0,
            "lambda_alpha_min": 0.0,
            "alpha_min_target": 0.0,
        },
    ),
)


def parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def append_arg(cmd: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if key in BOOLEAN_OPTIONAL_KEYS:
            cmd.append(f"--{key}" if value else f"--no-{key}")
        elif value:
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
        "alpha_state_path_absmean": None,
        "alpha_id_path_absmean": None,
        "alpha_bias_path_absmean": None,
        "irt_abs_mean": None,
        "personal_matrix_delta": None,
        "personal_matrix_student_std": None,
        "personal_delta_pre_softmax_norm": None,
        "personal_delta_student_std": None,
        "alpha_head_std": None,
        "relation_identity_delta": None,
        "knowledge_state_graph_delta": None,
        "knowledge_state_personal_delta": None,
        "personal_bad_row_count": None,
        "personal_fallback_row_count": None,
        "personal_student_mix": None,
        "personal_student_adapter_scale": None,
        "personal_logits_absmax": None,
        "local_row_ratio": None,
        "personal_support_density": None,
        "readout_query_delta": None,
        "warn_graph_uniform_count": 0,
        "warn_alpha_collapse_count": 0,
        "warn_personal_count": 0,
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

            if "[Diag][AE] Epoch" in line:
                for key in (
                    "irt_abs_mean",
                    "alpha_std",
                    "alpha_bias_std",
                    "personal_matrix_delta",
                    "personal_matrix_student_std",
                    "personal_delta_pre_softmax_norm",
                    "personal_delta_student_std",
                    "alpha_head_std",
                    "alpha_state_path",
                    "alpha_id_path",
                    "alpha_bias_path",
                    "personal_bad_rows",
                    "personal_fallback_rows",
                    "personal_student_mix",
                    "personal_student_adapter",
                    "personal_logits_absmax",
                    "local_row_ratio",
                    "support_density",
                    "readout_query_delta",
                ):
                    v = extract_float(line, key)
                    if v is not None:
                        mapped = {
                            "alpha_state_path": "alpha_state_path_absmean",
                            "alpha_id_path": "alpha_id_path_absmean",
                            "alpha_bias_path": "alpha_bias_path_absmean",
                            "personal_bad_rows": "personal_bad_row_count",
                            "personal_fallback_rows": "personal_fallback_row_count",
                            "personal_student_adapter": "personal_student_adapter_scale",
                            "support_density": "personal_support_density",
                        }.get(key, key)
                        out[mapped] = v

            if "[Diag][A] Epoch" in line or "[Diag][Graph] Epoch" in line:
                for key, target in (
                    ("entropy_ratio", "graph_entropy_ratio"),
                    ("relation_identity_delta", "relation_identity_delta"),
                    ("knowledge_state_graph_delta", "knowledge_state_graph_delta"),
                    ("knowledge_state_personal_delta", "knowledge_state_personal_delta"),
                ):
                    v = extract_float(line, key)
                    if v is not None:
                        out[target] = v

            if "[Diag Warning][Graph]" in line:
                out["warn_graph_uniform_count"] += 1
            if "alpha_std has been near zero" in line:
                out["warn_alpha_collapse_count"] += 1
            if "[Diag Warning][E]" in line:
                out["warn_personal_count"] += 1

            if "[Module Activity] Epoch 10:" in line:
                out["module_activity_epoch10"] = line.strip()

    return out


def _expected_runtime_flags(job: JobSpec) -> Dict[str, Any]:
    use_concept_graph = not bool(job.ablation.flags.get("ablate_concept_graph", False))
    use_personal_graph = bool(job.params.get("use_personal_graph", job.ablation.name != "no_E"))
    return {
        "enable_module1": True,
        "use_concept_graph": use_concept_graph,
        "use_personal_graph": use_personal_graph,
    }


def _validate_runtime_flags(job: JobSpec, runtime_facts: Dict[str, Any]) -> Tuple[bool, str]:
    expected = _expected_runtime_flags(job)
    for key, expected_value in expected.items():
        actual = runtime_facts.get(key)
        if actual is None:
            return False, f"missing_{key}"
        if bool(actual) != bool(expected_value):
            return False, f"{key}_expected_{expected_value}_got_{actual}"
    return True, ""


def _failure_from_metadata(
    *,
    exit_code: int,
    save_dir: Path,
    failure_json: Optional[Dict[str, Any]],
    log_file: Optional[Path],
) -> Tuple[str, str]:
    if failure_json:
        reason = str(failure_json.get("reason") or "runtime_exception")
        stage = str(failure_json.get("stage") or "")
        message = str(failure_json.get("message") or "")
        if reason.startswith("nonfinite_"):
            return reason, stage or "train_or_val"
        if "Ablation mismatch" in message:
            return "runtime_guardrail_fail", stage or "startup"
        return reason, stage

    if not (save_dir / "best_model.pth").exists():
        return "checkpoint_missing", ""

    if log_file and log_file.exists():
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if "Ablation mismatch" in text:
            return "runtime_guardrail_fail", "startup"
        if "Checkpoint/model architecture mismatch" in text or "State dict mismatch detected" in text:
            return "checkpoint_mismatch", "inference"
        if "nonfinite_" in text.lower() or "nan" in text.lower():
            return "nonfinite_loss", ""

    if exit_code != 0:
        return "runtime_exception", ""
    return "", ""


def pick_base_ablations(component_set: str) -> List[AblationSpec]:
    if component_set != "single":
        raise ValueError(f"Unknown component_set: {component_set}")
    return list(BASE_SINGLE_ABLATIONS)


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
    for key in sorted(job.params.keys()):
        if key in skip_keys:
            continue
        if key.startswith("ablate_"):
            continue
        append_arg(cmd, key, job.params[key])

    for key, value in job.ablation.flags.items():
        append_arg(cmd, key, value)

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
    failure_json = read_json(job.save_dir / "failure_reason.json") or {}
    metrics = test_json.get("metrics", {}) if isinstance(test_json, dict) else {}
    row["test_auc"] = metrics.get("auc")
    row["test_acc"] = metrics.get("acc")
    row["test_rmse"] = metrics.get("rmse")
    row["best_val_auc"] = test_json.get("best_val_auc")
    row["model_epoch"] = test_json.get("model_epoch")
    row["test_total_rows"] = test_json.get("test_total_rows")
    row["test_seen_rows"] = test_json.get("test_seen_rows")
    row["test_seen_coverage"] = test_json.get("test_seen_coverage")
    row["clean_baseline"] = test_json.get("train_only_split_hygiene")
    runtime_facts = test_json.get("runtime_facts", {}) if isinstance(test_json, dict) else {}
    config_switches = test_json.get("config_switches", {}) if isinstance(test_json, dict) else {}
    monitor = test_json.get("monitor", {}) if isinstance(test_json, dict) else {}
    ae_diagnostics = test_json.get("ae_diagnostics", {}) if isinstance(test_json, dict) else {}

    hist = read_json(job.save_dir / "training_history.json") or {}
    row["best_epoch"] = hist.get("best_epoch") if isinstance(hist, dict) else None

    args_json = read_json(job.save_dir / "args.json") or {}
    for key in ("enable_module1", "use_concept_graph", "use_personal_graph"):
        row[f"effective_{key}"] = runtime_facts.get(key, args_json.get(key))
    for key in STRUCTURAL_SWITCH_KEYS:
        row[key] = config_switches.get(key, args_json.get(key, job.params.get(key)))
    for key, value in runtime_facts.items():
        row[f"runtime_{key}"] = value
    for key, value in monitor.items():
        row[key] = value

    log_file = latest_train_log(job.log_dir)
    row["log_file"] = str(log_file) if log_file else ""
    row.update(parse_log_metrics(log_file))
    for key, value in ae_diagnostics.items():
        if key in row and row[key] in ("", None):
            row[key] = value

    if exit_code == 0:
        row["status"] = "ok"
    elif row.get("test_auc") is not None:
        row["status"] = "metrics_ok"

    failure_reason, failure_stage = _failure_from_metadata(
        exit_code=exit_code,
        save_dir=job.save_dir,
        failure_json=failure_json,
        log_file=log_file,
    )
    if row["status"] in {"ok", "metrics_ok"} and row.get("test_auc") is not None:
        failure_reason, failure_stage = "", ""
    row["failure_reason"] = failure_reason
    row["failure_stage"] = failure_stage

    ablation_valid, ablation_invalid_reason = _validate_runtime_flags(job, runtime_facts)
    row["ablation_valid"] = ablation_valid
    row["ablation_invalid_reason"] = ablation_invalid_reason
    if row["status"] in {"ok", "metrics_ok"} and not ablation_valid:
        row["status"] = "invalid"

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
        "test_total_rows",
        "test_seen_rows",
        "test_seen_coverage",
        "clean_baseline",
        "effective_enable_module1",
        "effective_use_concept_graph",
        "effective_use_personal_graph",
        "reg_bce_ratio",
        "graph_entropy_ratio",
        "alpha_std",
        "alpha_bias_std",
        "alpha_state_path_absmean",
        "alpha_id_path_absmean",
        "alpha_bias_path_absmean",
        "irt_abs_mean",
        "personal_matrix_delta",
        "personal_matrix_student_std",
        "personal_delta_pre_softmax_norm",
        "personal_delta_student_std",
        "alpha_head_std",
        "relation_identity_delta",
        "knowledge_state_graph_delta",
        "knowledge_state_personal_delta",
        "personal_bad_row_count",
        "personal_fallback_row_count",
        "personal_student_mix",
        "personal_student_adapter_scale",
        "personal_logits_absmax",
        "local_row_ratio",
        "personal_support_density",
        "readout_query_delta",
        "share_concept_embeddings",
        "personal_disable_direct_bias",
        "personal_direct_bias_scale",
        "personal_alpha_bias_scale",
        "personal_reg_warmup_epochs",
        "personal_disable_student_global_context",
        "personal_local_hops",
        "personal_support_only",
        "graph_propagation_alpha",
        "graph_readout_1hop_scale",
        "graph_readout_2hop_scale",
        "scheduler_monitor",
        "scheduler_mode",
        "best_monitor",
        "best_mode",
        "early_stop_monitor",
        "early_stop_mode",
        "warn_graph_uniform_count",
        "warn_alpha_collapse_count",
        "warn_personal_count",
        "module_activity_epoch10",
        "failure_reason",
        "failure_stage",
        "ablation_valid",
        "ablation_invalid_reason",
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
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _repair_glued_csv_rows(text: str) -> Tuple[str, int]:
    repaired, count = ROW_START_RE.subn("\n", text)
    return repaired, int(count)


def load_result_rows(path: Path, run_id: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        text = f.read()
    repaired_text, repaired_count = _repair_glued_csv_rows(text)
    if repaired_count > 0:
        print(f"[WARN] Auto-repaired {repaired_count} glued row boundary/boundaries in {path}.")
    rows: List[Dict[str, Any]] = []
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


def _majority_value(rows: Sequence[Dict[str, Any]], key: str) -> Any:
    counter: Dict[Any, int] = {}
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        counter[value] = counter.get(value, 0) + 1
    if not counter:
        return None
    return max(counter.items(), key=lambda kv: kv[1])[0]


def diagnose_reason(full_row: Dict[str, Any], delta_a: Optional[float], delta_e: Optional[float]) -> str:
    reasons: List[str] = []
    ger = try_float(full_row.get("graph_entropy_ratio"))
    alpha_std = try_float(full_row.get("alpha_std"))
    alpha_bias_std = try_float(full_row.get("alpha_bias_std"))
    alpha_state_path = try_float(full_row.get("alpha_state_path_absmean"))
    alpha_id_path = try_float(full_row.get("alpha_id_path_absmean"))
    personal_matrix_delta = try_float(full_row.get("personal_matrix_delta"))
    personal_matrix_student_std = try_float(full_row.get("personal_matrix_student_std"))
    personal_delta_pre_softmax_norm = try_float(full_row.get("personal_delta_pre_softmax_norm"))
    personal_delta_student_std = try_float(full_row.get("personal_delta_student_std"))
    alpha_head_std = try_float(full_row.get("alpha_head_std"))
    relation_identity_delta = try_float(full_row.get("relation_identity_delta"))
    knowledge_state_graph_delta = try_float(full_row.get("knowledge_state_graph_delta"))

    if ger is not None and ger > 0.98:
        reasons.append("graph-uniform-risk")
    if alpha_std is not None and alpha_std < 1e-6:
        reasons.append("personal-alpha-collapse")
    if alpha_bias_std is not None and alpha_bias_std < 1e-6:
        reasons.append("personal-bias-collapse")
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
    if alpha_state_path is not None and alpha_id_path is not None and alpha_id_path > max(alpha_state_path, 1e-6):
        reasons.append("alpha-id-dominant")
    if relation_identity_delta is not None and relation_identity_delta < 0.02:
        reasons.append("A-near-identity")
    if knowledge_state_graph_delta is not None and knowledge_state_graph_delta < 0.02:
        reasons.append("A-state-delta-low")
    if delta_a is not None and abs(delta_a) < 0.002:
        reasons.append("A-delta-small")
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
    for row in rows:
        if str(row.get("status", "")).lower() not in {"ok", "metrics_ok"}:
            continue
        if str(row.get("ablation_valid", "")).lower() == "false":
            continue
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("profile", "")))
        grouped.setdefault(key, {})
        grouped[key][str(row.get("ablation", ""))] = row

    summary_rows: List[Dict[str, Any]] = []
    for (dataset, seed, profile), mp in sorted(grouped.items()):
        full = mp.get("full")
        if full is None:
            continue

        full_auc = try_float(full.get("test_auc"))
        no_a_auc = try_float(mp.get("no_A", {}).get("test_auc"))
        no_e_auc = try_float(mp.get("no_E", {}).get("test_auc"))

        delta_a = (full_auc - no_a_auc) if (full_auc is not None and no_a_auc is not None) else None
        delta_e = (full_auc - no_e_auc) if (full_auc is not None and no_e_auc is not None) else None

        comp_state = {
            "A": classify_delta(delta_a, threshold),
            "E": classify_delta(delta_e, threshold),
        }
        keep = [key for key, value in comp_state.items() if value == "useful"]
        drop = [key for key, value in comp_state.items() if value == "harmful_or_unstable"]
        tune = [key for key, value in comp_state.items() if value == "neutral"]

        row = {
            "dataset": dataset,
            "seed": seed,
            "profile": profile,
            "full_auc": full_auc,
            "no_A_auc": no_a_auc,
            "no_E_auc": no_e_auc,
            "delta_A_full_minus_noA": delta_a,
            "delta_E_full_minus_noE": delta_e,
            "full_effective_use_concept_graph": full.get("effective_use_concept_graph"),
            "full_effective_use_personal_graph": full.get("effective_use_personal_graph"),
            "full_clean_baseline": full.get("clean_baseline"),
            "full_test_total_rows": try_float(full.get("test_total_rows")),
            "full_test_seen_rows": try_float(full.get("test_seen_rows")),
            "full_test_seen_coverage": try_float(full.get("test_seen_coverage")),
            "full_share_concept_embeddings": full.get("share_concept_embeddings"),
            "full_personal_disable_direct_bias": full.get("personal_disable_direct_bias"),
            "full_personal_direct_bias_scale": try_float(full.get("personal_direct_bias_scale")),
            "full_personal_alpha_bias_scale": try_float(full.get("personal_alpha_bias_scale")),
            "full_personal_reg_warmup_epochs": try_float(full.get("personal_reg_warmup_epochs")),
            "full_personal_disable_student_global_context": full.get("personal_disable_student_global_context"),
            "full_personal_local_hops": try_float(full.get("personal_local_hops")),
            "full_personal_support_only": full.get("personal_support_only"),
            "full_graph_propagation_alpha": try_float(full.get("graph_propagation_alpha")),
            "full_graph_readout_1hop_scale": try_float(full.get("graph_readout_1hop_scale")),
            "full_graph_readout_2hop_scale": try_float(full.get("graph_readout_2hop_scale")),
            "full_ablation_valid": full.get("ablation_valid"),
            "full_graph_entropy_ratio": try_float(full.get("graph_entropy_ratio")),
            "full_alpha_std": try_float(full.get("alpha_std")),
            "full_alpha_bias_std": try_float(full.get("alpha_bias_std")),
            "full_alpha_state_path_absmean": try_float(full.get("alpha_state_path_absmean")),
            "full_alpha_id_path_absmean": try_float(full.get("alpha_id_path_absmean")),
            "full_personal_matrix_delta": try_float(full.get("personal_matrix_delta")),
            "full_personal_matrix_student_std": try_float(full.get("personal_matrix_student_std")),
            "full_personal_delta_pre_softmax_norm": try_float(full.get("personal_delta_pre_softmax_norm")),
            "full_personal_delta_student_std": try_float(full.get("personal_delta_student_std")),
            "full_alpha_head_std": try_float(full.get("alpha_head_std")),
            "full_relation_identity_delta": try_float(full.get("relation_identity_delta")),
            "full_knowledge_state_graph_delta": try_float(full.get("knowledge_state_graph_delta")),
            "full_knowledge_state_personal_delta": try_float(full.get("knowledge_state_personal_delta")),
            "full_warn_graph_uniform_count": full.get("warn_graph_uniform_count"),
            "full_warn_alpha_collapse_count": full.get("warn_alpha_collapse_count"),
            "full_warn_personal_count": full.get("warn_personal_count"),
            "full_module_activity_epoch10": full.get("module_activity_epoch10", ""),
            "state_A": comp_state["A"],
            "state_E": comp_state["E"],
            "suggest_keep": ",".join(keep),
            "suggest_drop": ",".join(drop),
            "suggest_tune": ",".join(tune),
            "diagnosis_reason": diagnose_reason(full, delta_a, delta_e),
        }
        summary_rows.append(row)

    summary_fields = [
        "dataset",
        "seed",
        "profile",
        "full_auc",
        "no_A_auc",
        "no_E_auc",
        "delta_A_full_minus_noA",
        "delta_E_full_minus_noE",
        "full_effective_use_concept_graph",
        "full_effective_use_personal_graph",
        "full_clean_baseline",
        "full_test_total_rows",
        "full_test_seen_rows",
        "full_test_seen_coverage",
        "full_share_concept_embeddings",
        "full_personal_disable_direct_bias",
        "full_personal_direct_bias_scale",
        "full_personal_alpha_bias_scale",
        "full_personal_reg_warmup_epochs",
        "full_personal_disable_student_global_context",
        "full_personal_local_hops",
        "full_personal_support_only",
        "full_graph_propagation_alpha",
        "full_graph_readout_1hop_scale",
        "full_graph_readout_2hop_scale",
        "full_ablation_valid",
        "full_graph_entropy_ratio",
        "full_alpha_std",
        "full_alpha_bias_std",
        "full_alpha_state_path_absmean",
        "full_alpha_id_path_absmean",
        "full_personal_matrix_delta",
        "full_personal_matrix_student_std",
        "full_personal_delta_pre_softmax_norm",
        "full_personal_delta_student_std",
        "full_alpha_head_std",
        "full_relation_identity_delta",
        "full_knowledge_state_graph_delta",
        "full_knowledge_state_personal_delta",
        "full_warn_graph_uniform_count",
        "full_warn_alpha_collapse_count",
        "full_warn_personal_count",
        "full_module_activity_epoch10",
        "state_A",
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
        for row in summary_rows:
            writer.writerow(row)

    mean_rows: List[Dict[str, Any]] = []
    mean_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in summary_rows:
        mean_group.setdefault((row["dataset"], row["profile"]), []).append(row)

    for (dataset, profile), grp in sorted(mean_group.items()):
        deltas_a = [try_float(x.get("delta_A_full_minus_noA")) for x in grp]
        deltas_e = [try_float(x.get("delta_E_full_minus_noE")) for x in grp]
        full_aucs = [try_float(x.get("full_auc")) for x in grp]

        def mean_valid(values: List[Optional[float]]) -> Optional[float]:
            arr = [v for v in values if v is not None]
            return float(sum(arr) / len(arr)) if arr else None

        def useful_ratio(values: List[Optional[float]]) -> Optional[float]:
            arr = [v for v in values if v is not None]
            if not arr:
                return None
            return float(sum(1 for v in arr if v > threshold) / len(arr))

        mean_a = mean_valid(deltas_a)
        mean_e = mean_valid(deltas_e)
        keep = []
        if mean_a is not None and mean_a > threshold:
            keep.append("A")
        if mean_e is not None and mean_e > threshold:
            keep.append("E")

        mean_rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "n_seeds": len(grp),
                "full_auc_mean": mean_valid(full_aucs),
                "delta_A_mean": mean_a,
                "delta_E_mean": mean_e,
                "full_clean_baseline_majority": _majority_value(grp, "full_clean_baseline"),
                "full_test_seen_coverage_mean": mean_valid(
                    [try_float(x.get("full_test_seen_coverage")) for x in grp]
                ),
                "A_useful_seed_ratio": useful_ratio(deltas_a),
                "E_useful_seed_ratio": useful_ratio(deltas_e),
                "suggest_keep_by_mean": ",".join(keep),
                "state_A_majority": _majority_value(grp, "state_A"),
                "state_E_majority": _majority_value(grp, "state_E"),
            }
        )

    mean_fields = [
        "dataset",
        "profile",
        "n_seeds",
        "full_auc_mean",
        "delta_A_mean",
        "delta_E_mean",
        "full_clean_baseline_majority",
        "full_test_seen_coverage_mean",
        "A_useful_seed_ratio",
        "E_useful_seed_ratio",
        "suggest_keep_by_mean",
        "state_A_majority",
        "state_E_majority",
    ]
    with open(mean_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mean_fields)
        writer.writeheader()
        for row in mean_rows:
            writer.writerow(row)

    return summary_rows, mean_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AE-only ablations with diagnosis summary.")
    parser.add_argument("--datasets", type=str, default="assist_09,junyi")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds. Default uses DEFAULT_SEEDS.")
    parser.add_argument("--profiles", type=str, default=None, help="Optional filter: best,ae_dominant")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_interval", type=int, default=10)
    parser.add_argument("--run_id", type=str, default=None, help="Output namespace. Default=timestamp.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--analyze_only", action="store_true", help="Skip running and only summarize existing rows.")
    parser.add_argument("--generate_diagnosis", action="store_true", help="Enable post-training diagnosis artifacts.")
    parser.add_argument("--component_set", type=str, default="single", choices=["single"])
    parser.add_argument("--ablations", type=str, default=None, help="Optional filter: full,no_A,no_E")
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
                "graph_identity_residual": 0.05,
                "graph_propagation_alpha": 0.22,
                "graph_readout_1hop_scale": 0.45,
                "graph_readout_2hop_scale": 0.18,
                "personal_warmup_epochs": 6,
                "personal_reg_warmup_epochs": 6,
                "personal_max_alpha": 0.45,
                "personal_delta_scale": 7.0,
                "lambda_alpha_min": 0.10,
                "alpha_min_target": 0.05,
                "personal_alpha_bias_scale": 0.03,
                "personal_direct_bias_scale": 0.0,
                "personal_disable_direct_bias": True,
                "personal_disable_student_global_context": True,
                "personal_local_hops": 1,
                "personal_support_only": True,
                "share_concept_embeddings": True,
            }
        return {
            "graph_identity_residual": 0.05,
            "graph_propagation_alpha": 0.28,
            "graph_readout_1hop_scale": 0.34,
            "graph_readout_2hop_scale": 0.12,
            "personal_warmup_epochs": 4,
            "personal_reg_warmup_epochs": 4,
            "personal_max_alpha": 0.40,
            "personal_delta_scale": 5.0,
            "lambda_alpha_min": 0.08,
            "alpha_min_target": 0.04,
            "personal_alpha_bias_scale": 0.015,
            "personal_direct_bias_scale": 0.0,
            "personal_disable_direct_bias": True,
            "personal_disable_student_global_context": True,
            "personal_local_hops": 1,
            "personal_support_only": True,
            "share_concept_embeddings": True,
        }
    raise ValueError(f"Unknown profile '{profile}'.")


def _selected_profiles(all_profiles: List[str], filters: Optional[str]) -> List[str]:
    if not filters:
        return all_profiles
    names = set(parse_csv_tokens(filters))
    chosen = [profile for profile in all_profiles if profile in names]
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
    chosen = [ab for ab in all_abls if ab.name in names]
    missing = names - set(ab.name for ab in all_abls)
    if missing:
        raise ValueError(f"Unknown ablation names: {sorted(missing)}")
    if not chosen:
        raise ValueError("No ablations selected after filtering.")
    return chosen


def make_jobs(args: argparse.Namespace, run_id: str) -> List[JobSpec]:
    datasets = parse_csv_tokens(args.datasets)
    seeds = list(DEFAULT_SEEDS) if args.seeds is None else parse_int_csv(args.seeds)
    base_abls = _selected_ablations(pick_base_ablations(args.component_set), args.ablations)
    profiles = _selected_profiles(["best", "ae_dominant"], args.profiles)

    jobs: List[JobSpec] = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not found in BEST_CFG.")
        base_cfg = dict(BEST_CFG[dataset])

        for profile in profiles:
            profile_cfg = _profile_overrides(profile, dataset, args)
            for ablation in base_abls:
                for seed in seeds:
                    params = dict(base_cfg)
                    params.update(profile_cfg)
                    params.update(ablation.overrides)
                    params["debug_graph_diag"] = True
                    params["diag_batches"] = 1

                    if args.epochs is not None:
                        params["epochs"] = int(args.epochs)
                    if args.early_stop_patience is not None:
                        params["early_stop_patience"] = int(args.early_stop_patience)
                    if args.learning_rate is not None:
                        params["learning_rate"] = float(args.learning_rate)

                    for key in set(ablation.drop_keys) | set(ablation.flags.keys()):
                        params.pop(key, None)

                    model_variant = f"{dataset}_abce_{profile}_{ablation.name}"
                    save_dir = Path("checkpoints") / "abce_diag" / run_id / dataset / f"seed{seed}" / f"{profile}_{ablation.name}"
                    log_dir = Path("logs") / "abce_diag" / run_id / dataset / f"seed{seed}" / f"{profile}_{ablation.name}"

                    if (not args.rerun_existing) and (save_dir / "test_results.json").exists():
                        continue

                    save_dir.mkdir(parents=True, exist_ok=True)
                    log_dir.mkdir(parents=True, exist_ok=True)

                    job = JobSpec(
                        dataset=dataset,
                        seed=int(seed),
                        profile=profile,
                        ablation=ablation,
                        model_variant=model_variant,
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
    for row in summary_rows:
        print(
            f"[SUMMARY] dataset={row['dataset']} seed={row['seed']} profile={row['profile']} "
            f"dA={row['delta_A_full_minus_noA']} dE={row['delta_E_full_minus_noE']} "
            f"keep={row['suggest_keep']} reason={row['diagnosis_reason']}"
        )


if __name__ == "__main__":
    main()
