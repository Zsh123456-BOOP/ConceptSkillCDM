#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run staged ERS/SLPR mechanism experiments and plot them with R."""

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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from best_configs import BEST_CFG  # noqa: E402
from gpu_utils import calc_effective_max_concurrent, parse_gpu_ids, pick_gpu_with_slot_round_robin  # noqa: E402
from run_abce_ablation import AblationSpec, JobSpec, _build_job_env, build_command, collect_result  # noqa: E402


DEFAULT_DATASETS = (
    "assist_09",
    "junyi",
    "assist_17",
    "frcsub",
    "nips34",
    "ednet_kt1_clean15_sample5000",
    "assist_12_clean15_item50",
)

DEFAULT_VARIANTS = (
    "full",
    "no_A",
    "no_E",
    "A_item_only",
    "A_seq_only",
    "A_uniform",
    "A_self_only",
    "E_prior_only",
    "E_frozen_alpha",
)


FAIR_NEUTRAL_E_OVERRIDES = {
    # Keep use_personal_graph=True so the predictor/head path is identical to
    # full, but make the local posterior and query writeback an identity
    # control. This is the fair A substrate test: only A evidence changes.
    "personal_delta_scale": 0.0,
    "personal_query_correction_scale": 0.0,
    "personal_query_message_gain": 0.0,
    "personal_mastery_prior_scale": 0.0,
    "personal_recent_mastery_prior_scale": 0.0,
    "personal_item_support_mass": 0.0,
    "ae_posterior_prior_scale": 0.0,
    "lambda_personal_kl": 0.0,
    "lambda_personal_query_residual": 0.0,
    "lambda_alpha_min": 0.0,
    "alpha_min_target": 0.0,
}


FAIR_GLOBAL_POSTERIOR_OVERRIDES = {
    # Keep A, stat priors, AE residuals, and the E container active. Only force
    # the student-conditioned posterior to equal the global support and remove
    # the query-state writeback.
    "personal_delta_scale": 0.0,
    "personal_query_correction_scale": 0.0,
    "personal_query_message_gain": 0.0,
    "ae_posterior_prior_scale": 0.0,
    "lambda_personal_kl": 0.0,
    "lambda_personal_query_residual": 0.0,
}


PHASE1_PERSONAL_ACTIVE_DEFAULTS = {
    # The BEST_CFG files preserve some high-AUC runs where the old E writeback
    # path was intentionally muted.  Phase1 is a mechanism test, so full-like E
    # variants must exercise the posterior and bounded query writeback.
    "personal_delta_scale": 1.0,
    "personal_query_correction_scale": 0.15,
    "personal_query_message_gain": 1.0,
}

PHASE1_MAX_PERSONAL_WARMUP_EPOCHS = 2
PHASE1_MAX_AE_LR_MULT = 5.0

NEUTRAL_E_VARIANTS = {
    "no_E",
    "no_E_fair",
    "A_fused_neutralE",
    "A_item_neutralE",
    "A_seq_neutralE",
    "A_uniform_neutralE",
    "A_self_neutralE",
    "E_global_posterior",
    "E_prior_only",
    "E_theta025_global",
    "E_theta050_global",
    "E_mastery05_global",
    "E_mastery10_global",
    "E_readout_global",
    "E_theta_readout_global",
    "E_mastery10_gate8_global",
    "E_mastery10_gate8_pp2_global",
}

POSTERIOR_ONLY_E_VARIANTS = {
    "E_posterior_only",
    "E_theta025_posterior_only",
    "E_theta050_posterior_only",
    "E_mastery05_posterior_only",
    "E_mastery10_posterior_only",
    "E_readout_posterior_only",
    "E_theta_readout_posterior_only",
    "E_mastery10_gate8_posterior_only",
    "E_mastery10_gate8_pp2_posterior_only",
}

QUERY_ONLY_E_VARIANTS = {
    "E_query_only",
}


def _as_float(params: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _raise_if_personal_writes_all_zero(params: Dict[str, Any], variant: str) -> None:
    if not bool(params.get("use_personal_graph", True)):
        return
    if variant in NEUTRAL_E_VARIANTS:
        return
    active = any(_as_float(params, key) > 0.0 for key in PHASE1_PERSONAL_ACTIVE_DEFAULTS)
    if not active:
        raise ValueError(
            f"{variant} keeps use_personal_graph=True but all E write paths are zero. "
            "Use no_E/neutral controls for an intentionally inactive E."
        )


def _ensure_positive(params: Dict[str, Any], key: str) -> None:
    if _as_float(params, key) <= 0.0:
        params[key] = PHASE1_PERSONAL_ACTIVE_DEFAULTS[key]


def _cap_phase1_personal_warmup(params: Dict[str, Any], phase_epochs: int) -> None:
    max_warmup = min(PHASE1_MAX_PERSONAL_WARMUP_EPOCHS, max(0, int(phase_epochs) // 2))
    for key in ("personal_warmup_epochs", "personal_reg_warmup_epochs"):
        if key in params:
            params[key] = min(int(params.get(key, max_warmup)), max_warmup)


def _apply_phase1_personal_activation(params: Dict[str, Any], variant: str, phase_epochs: int) -> None:
    if _as_float(params, "ae_lr_mult", 1.0) > PHASE1_MAX_AE_LR_MULT:
        params["ae_lr_mult"] = PHASE1_MAX_AE_LR_MULT
    if not bool(params.get("use_personal_graph", True)):
        return
    _cap_phase1_personal_warmup(params, phase_epochs)
    if variant in NEUTRAL_E_VARIANTS:
        return
    if variant in POSTERIOR_ONLY_E_VARIANTS:
        _ensure_positive(params, "personal_delta_scale")
        _raise_if_personal_writes_all_zero(params, variant)
        return
    if variant in QUERY_ONLY_E_VARIANTS:
        _ensure_positive(params, "personal_query_correction_scale")
        _ensure_positive(params, "personal_query_message_gain")
        _raise_if_personal_writes_all_zero(params, variant)
        return
    for key in PHASE1_PERSONAL_ACTIVE_DEFAULTS:
        _ensure_positive(params, key)
    _raise_if_personal_writes_all_zero(params, variant)


def _with_relation_theta(base: Optional[Dict[str, Any]] = None, *, scale: float) -> Dict[str, Any]:
    overrides = dict(base or {})
    overrides["relation_theta_scale"] = float(scale)
    return overrides


def _with_mastery_prior(
    base: Optional[Dict[str, Any]] = None,
    *,
    mastery: float,
    recent: float,
    posterior_scale: float = 1.0,
    theta_scale: float = 0.0,
    count_smoothing: float = 0.0,
) -> Dict[str, Any]:
    overrides = dict(base or {})
    overrides["personal_mastery_prior_scale"] = float(mastery)
    overrides["personal_recent_mastery_prior_scale"] = float(recent)
    overrides["ae_posterior_prior_scale"] = float(posterior_scale)
    overrides["ae_posterior_theta_scale"] = float(theta_scale)
    overrides["personal_mastery_count_smoothing"] = float(count_smoothing)
    return overrides

RESULT_FIELD_HINTS = (
    "run_id",
    "phase",
    "dataset",
    "seed",
    "variant",
    "profile",
    "status",
    "exit_code",
    "test_auc",
    "test_acc",
    "test_rmse",
    "best_val_auc",
    "best_epoch",
    "model_epoch",
    "ablation_valid",
    "ablation_invalid_reason",
    "graph_prior_mode",
    "shuffle_student_concept_priors",
    "personal_freeze_alpha_gate",
    "disable_item_prior",
    "disable_sequence_prior",
    "effective_use_concept_graph",
    "effective_use_personal_graph",
    "support_candidate_size_mean",
    "support_final_size_mean",
    "support_item_survival_rate",
    "support_seq_survival_rate",
    "support_item_seq_overlap",
    "support_self_retention_rate",
    "graph_entropy_ratio",
    "relation_identity_delta",
    "knowledge_state_graph_delta",
    "knowledge_state_personal_delta",
    "alpha_std",
    "alpha_delta_absmean",
    "query_row_posterior_kl",
    "query_row_posterior_delta_abs",
    "query_row_personal_message_delta",
    "roadmap_macro_logit_abs_mean",
    "roadmap_difficulty_logit_abs_mean",
    "roadmap_reliability_logit_abs_mean",
    "roadmap_item_logit_abs_mean",
    "roadmap_sequence_logit_abs_mean",
    "roadmap_route_difficulty_delta_abs_mean",
    "roadmap_route_reliability_delta_abs_mean",
    "roadmap_item_difficulty_delta_abs_mean",
    "roadmap_sequence_difficulty_delta_abs_mean",
    "tutor_local_navigation_logit_abs_mean",
    "tutor_current_mastery_logit_abs_mean",
    "tutor_current_recent_logit_abs_mean",
    "tutor_route_mastery_logit_abs_mean",
    "tutor_route_recent_logit_abs_mean",
    "tutor_gap_penalty_logit_abs_mean",
    "tutor_query_reliability_mean",
    "tutor_route_reliability_mean",
    "tutor_personal_route_mass_mean",
    "tutor_personal_route_delta_abs_mean",
    "personal_mastery_reliability_mean",
    "personal_recent_mastery_reliability_mean",
    "personal_support_density",
    "personal_to_graph_query_ratio",
    "log_file",
    "save_dir",
    "log_dir",
    "params_json",
    "flags_json",
)


@dataclass
class MechanismJob:
    phase: str
    variant: str
    job: JobSpec
    params: Dict[str, Any]


def parse_csv_tokens(text: str) -> List[str]:
    return [token.strip() for token in str(text).split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default="", help="Run id. Default uses timestamp.")
    parser.add_argument("--phase", choices=["phase1", "phase2", "both"], default="both")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--gpus", default="1,3")
    parser.add_argument("--max_concurrent", type=int, default=0, help="Default: one job per GPU.")
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_interval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--phase1_epochs", type=int, default=6)
    parser.add_argument("--phase1_max_train_batches", type=int, default=120)
    parser.add_argument("--phase1_max_val_batches", type=int, default=60)
    parser.add_argument("--phase1_max_test_batches", type=int, default=60)
    parser.add_argument("--phase2_epochs", type=int, default=45)
    parser.add_argument("--phase2_patience", type=int, default=8)
    parser.add_argument("--limit_jobs", type=int, default=0, help="Optional total cap for smoke testing.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--generate_diagnosis", action="store_true")
    parser.add_argument("--skip_r", action="store_true")
    return parser.parse_args()


def _variant_spec(name: str) -> AblationSpec:
    if name == "full":
        return AblationSpec(name=name, flags={}, overrides={})
    if name == "no_A":
        return AblationSpec(name=name, flags={"ablate_concept_graph": True}, overrides={})
    if name == "no_E":
        return AblationSpec(
            name=name,
            flags={},
            overrides={
                "use_personal_graph": False,
                "lambda_sparse_personal": 0.0,
                "lambda_alpha": 0.0,
                "lambda_alpha_min": 0.0,
                "alpha_min_target": 0.0,
            },
        )
    if name == "full_fair":
        return AblationSpec(name=name, flags={}, overrides={})
    if name == "no_A_fair":
        return AblationSpec(
            name=name,
            flags={"ablate_concept_graph": True},
            overrides=dict(FAIR_NEUTRAL_E_OVERRIDES),
        )
    if name == "no_E_fair":
        return AblationSpec(name=name, flags={}, overrides=dict(FAIR_GLOBAL_POSTERIOR_OVERRIDES))
    if name == "A_item_only":
        return AblationSpec(name=name, flags={}, overrides={"graph_prior_mode": "item_only"})
    if name == "A_seq_only":
        return AblationSpec(name=name, flags={}, overrides={"graph_prior_mode": "seq_only"})
    if name == "A_uniform":
        return AblationSpec(name=name, flags={}, overrides={"graph_prior_mode": "uniform"})
    if name == "A_self_only":
        return AblationSpec(name=name, flags={}, overrides={"graph_prior_mode": "self_only"})
    if name == "A_fused_neutralE":
        return AblationSpec(name=name, flags={}, overrides=dict(FAIR_NEUTRAL_E_OVERRIDES))
    if name == "A_item_neutralE":
        overrides = dict(FAIR_NEUTRAL_E_OVERRIDES)
        overrides["graph_prior_mode"] = "item_only"
        return AblationSpec(name=name, flags={}, overrides=overrides)
    if name == "A_seq_neutralE":
        overrides = dict(FAIR_NEUTRAL_E_OVERRIDES)
        overrides["graph_prior_mode"] = "seq_only"
        return AblationSpec(name=name, flags={}, overrides=overrides)
    if name == "A_uniform_neutralE":
        overrides = dict(FAIR_NEUTRAL_E_OVERRIDES)
        overrides["graph_prior_mode"] = "uniform"
        return AblationSpec(name=name, flags={}, overrides=overrides)
    if name == "A_self_neutralE":
        overrides = dict(FAIR_NEUTRAL_E_OVERRIDES)
        overrides["graph_prior_mode"] = "self_only"
        return AblationSpec(name=name, flags={}, overrides=overrides)
    if name == "E_prior_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides={
                "personal_delta_scale": 0.0,
                "personal_mastery_prior_scale": 0.0,
                "personal_recent_mastery_prior_scale": 0.0,
                "personal_item_support_mass": 0.0,
            },
        )
    if name == "E_frozen_alpha":
        return AblationSpec(
            name=name,
            flags={},
            overrides={
                "personal_alpha_budget": 0.0,
                "personal_alpha_bias_scale": 0.0,
                "personal_freeze_alpha_gate": True,
                "personal_warmup_epochs": 0,
                "personal_reg_warmup_epochs": 0,
            },
        )
    if name == "E_full_fair":
        return AblationSpec(name=name, flags={}, overrides={})
    if name == "E_global_posterior":
        return AblationSpec(name=name, flags={}, overrides=dict(FAIR_GLOBAL_POSTERIOR_OVERRIDES))
    if name == "E_shuffle_student":
        return AblationSpec(name=name, flags={}, overrides={"shuffle_student_concept_priors": True})
    if name == "E_posterior_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides={
                "personal_query_correction_scale": 0.0,
                "personal_query_message_gain": 0.0,
                "lambda_personal_query_residual": 0.0,
            },
        )
    if name == "E_query_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides={
                "personal_delta_scale": 0.0,
                "ae_posterior_prior_scale": 0.0,
                "lambda_personal_kl": 0.0,
            },
        )
    if name == "E_theta025_full":
        return AblationSpec(name=name, flags={}, overrides=_with_relation_theta(scale=0.25))
    if name == "E_theta025_global":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_relation_theta(FAIR_GLOBAL_POSTERIOR_OVERRIDES, scale=0.25),
        )
    if name == "E_theta025_posterior_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_relation_theta(
                {
                    "personal_query_correction_scale": 0.0,
                    "personal_query_message_gain": 0.0,
                    "lambda_personal_query_residual": 0.0,
                },
                scale=0.25,
            ),
        )
    if name == "E_theta050_full":
        return AblationSpec(name=name, flags={}, overrides=_with_relation_theta(scale=0.50))
    if name == "E_theta050_global":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_relation_theta(FAIR_GLOBAL_POSTERIOR_OVERRIDES, scale=0.50),
        )
    if name == "E_theta050_posterior_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_relation_theta(
                {
                    "personal_query_correction_scale": 0.0,
                    "personal_query_message_gain": 0.0,
                    "lambda_personal_query_residual": 0.0,
                },
                scale=0.50,
            ),
        )
    if name == "E_mastery05_full":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(mastery=0.5, recent=0.25),
        )
    if name == "E_mastery05_global":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(FAIR_GLOBAL_POSTERIOR_OVERRIDES, mastery=0.5, recent=0.25),
        )
    if name == "E_mastery05_posterior_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                {
                    "personal_query_correction_scale": 0.0,
                    "personal_query_message_gain": 0.0,
                    "lambda_personal_query_residual": 0.0,
                },
                mastery=0.5,
                recent=0.25,
            ),
        )
    if name == "E_mastery10_full":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(mastery=1.0, recent=0.5),
        )
    if name == "E_mastery10_global":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(FAIR_GLOBAL_POSTERIOR_OVERRIDES, mastery=1.0, recent=0.5),
        )
    if name == "E_mastery10_posterior_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                {
                    "personal_query_correction_scale": 0.0,
                    "personal_query_message_gain": 0.0,
                    "lambda_personal_query_residual": 0.0,
                },
                mastery=1.0,
                recent=0.5,
            ),
        )
    if name == "E_readout_full":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(mastery=1.0, recent=0.5, posterior_scale=2.0, theta_scale=1.0),
        )
    if name == "E_readout_global":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                FAIR_GLOBAL_POSTERIOR_OVERRIDES,
                mastery=1.0,
                recent=0.5,
                posterior_scale=2.0,
                theta_scale=1.0,
            ),
        )
    if name == "E_readout_posterior_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                {
                    "personal_query_correction_scale": 0.0,
                    "personal_query_message_gain": 0.0,
                    "lambda_personal_query_residual": 0.0,
                },
                mastery=1.0,
                recent=0.5,
                posterior_scale=2.0,
                theta_scale=1.0,
            ),
        )
    if name == "E_theta_readout_full":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(mastery=0.0, recent=0.0, posterior_scale=0.0, theta_scale=2.0),
        )
    if name == "E_theta_readout_global":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                FAIR_GLOBAL_POSTERIOR_OVERRIDES,
                mastery=0.0,
                recent=0.0,
                posterior_scale=0.0,
                theta_scale=2.0,
            ),
        )
    if name == "E_theta_readout_posterior_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                {
                    "personal_query_correction_scale": 0.0,
                    "personal_query_message_gain": 0.0,
                    "lambda_personal_query_residual": 0.0,
                },
                mastery=0.0,
                recent=0.0,
                posterior_scale=0.0,
                theta_scale=2.0,
            ),
        )
    if name == "E_mastery10_gate8_full":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(mastery=1.0, recent=0.5, count_smoothing=8.0),
        )
    if name == "E_mastery10_gate8_global":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                FAIR_GLOBAL_POSTERIOR_OVERRIDES,
                mastery=1.0,
                recent=0.5,
                count_smoothing=8.0,
            ),
        )
    if name == "E_mastery10_gate8_posterior_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                {
                    "personal_query_correction_scale": 0.0,
                    "personal_query_message_gain": 0.0,
                    "lambda_personal_query_residual": 0.0,
                },
                mastery=1.0,
                recent=0.5,
                count_smoothing=8.0,
            ),
        )
    if name == "E_mastery10_gate8_pp2_full":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                mastery=1.0,
                recent=0.5,
                posterior_scale=2.0,
                count_smoothing=8.0,
            ),
        )
    if name == "E_mastery10_gate8_pp2_global":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                FAIR_GLOBAL_POSTERIOR_OVERRIDES,
                mastery=1.0,
                recent=0.5,
                posterior_scale=2.0,
                count_smoothing=8.0,
            ),
        )
    if name == "E_mastery10_gate8_pp2_posterior_only":
        return AblationSpec(
            name=name,
            flags={},
            overrides=_with_mastery_prior(
                {
                    "personal_query_correction_scale": 0.0,
                    "personal_query_message_gain": 0.0,
                    "lambda_personal_query_residual": 0.0,
                },
                mastery=1.0,
                recent=0.5,
                posterior_scale=2.0,
                count_smoothing=8.0,
            ),
        )
    raise ValueError(f"Unknown mechanism variant: {name}")


def _phase_params(phase: str, args: argparse.Namespace) -> Dict[str, Any]:
    if phase == "phase1":
        return {
            "epochs": int(args.phase1_epochs),
            "early_stop_patience": 3,
            "patience": 3,
            "max_train_batches": int(args.phase1_max_train_batches),
            "max_val_batches": int(args.phase1_max_val_batches),
            "max_test_batches": int(args.phase1_max_test_batches),
            "debug_graph_diag": True,
            "diag_batches": 1,
            "num_workers": int(args.num_workers),
        }
    if phase == "phase2":
        return {
            "epochs": int(args.phase2_epochs),
            "early_stop_patience": int(args.phase2_patience),
            "patience": int(args.phase2_patience),
            "debug_graph_diag": True,
            "diag_batches": 1,
            "num_workers": int(args.num_workers),
        }
    raise ValueError(f"Unknown phase: {phase}")


def _base_params(dataset: str, args: argparse.Namespace) -> Dict[str, Any]:
    params = dict(BEST_CFG.get(dataset, {}))
    params.pop("seed", None)
    params.pop("model_variant", None)
    params["num_workers"] = int(args.num_workers)
    return params


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _supplement_row(row: Dict[str, Any], job: MechanismJob, run_id: str) -> Dict[str, Any]:
    out = dict(row)
    out["run_id"] = run_id
    out["phase"] = job.phase
    out["variant"] = job.variant
    args_json = _read_json(job.job.save_dir / "args.json")
    diag = _read_json(job.job.save_dir / "diag_best.json") or _read_json(job.job.save_dir / "diag_last.json")
    test_json = _read_json(job.job.save_dir / "test_results.json")
    config_switches = test_json.get("config_switches", {}) if isinstance(test_json, dict) else {}
    for key in ("graph_prior_mode", "disable_item_prior", "disable_sequence_prior"):
        out[key] = config_switches.get(key, args_json.get(key, job.params.get(key, "")))
    for key, value in diag.items():
        out.setdefault(key, value)
        if out.get(key) in ("", None):
            out[key] = value
    return out


def _fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    keys = set(RESULT_FIELD_HINTS)
    for row in rows:
        keys.update(row.keys())
    ordered = [key for key in RESULT_FIELD_HINTS if key in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def append_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows: List[Dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            old_rows = list(csv.DictReader(f))
    fieldnames = _fieldnames([*old_rows, *rows])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerows(rows)


def update_status(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_jobs(args: argparse.Namespace, run_id: str) -> List[MechanismJob]:
    phases = ["phase1", "phase2"] if args.phase == "both" else [args.phase]
    datasets = parse_csv_tokens(args.datasets)
    variants = parse_csv_tokens(args.variants)
    jobs: List[MechanismJob] = []
    for phase in phases:
        for dataset in datasets:
            base = _base_params(dataset, args)
            base.update(_phase_params(phase, args))
            for variant in variants:
                spec = _variant_spec(variant)
                params = dict(base)
                params.update(spec.overrides)
                if phase == "phase1":
                    _apply_phase1_personal_activation(
                        params,
                        variant,
                        int(args.phase1_epochs),
                    )
                save_dir = Path("checkpoints") / "mechanism" / run_id / phase / dataset / variant
                log_dir = Path("logs") / "mechanism" / run_id / phase / dataset / variant
                if (not args.rerun_existing) and (save_dir / "test_results.json").exists():
                    continue
                job = JobSpec(
                    dataset=dataset,
                    seed=int(args.seed),
                    profile=phase,
                    ablation=spec,
                    model_variant=f"{dataset}_{phase}_{variant}",
                    save_dir=save_dir,
                    log_dir=log_dir,
                    params=params,
                    cmd=[],
                )
                job.cmd = build_command(job, generate_diagnosis=bool(args.generate_diagnosis))
                jobs.append(MechanismJob(phase=phase, variant=variant, job=job, params=params))
    if args.limit_jobs and args.limit_jobs > 0:
        jobs = jobs[: int(args.limit_jobs)]
    return jobs


def run_postprocess(result_csv: Path, args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    out_dir = result_csv.parent
    eval_cmd = [
        sys.executable,
        "tools/evaluate_mechanism_results.py",
        str(result_csv),
        "--out_dir",
        str(out_dir),
    ]
    print("[POST] ", " ".join(eval_cmd))
    subprocess.run(eval_cmd, check=False)
    if args.skip_r:
        return
    rscript = "Rscript"
    plot_dir = out_dir / "figures"
    r_cmd = [rscript, "tools/plot_mechanism_results.R", str(result_csv), str(plot_dir)]
    print("[POST] ", " ".join(r_cmd))
    subprocess.run(r_cmd, check=False)


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    run_id = args.run_id or f"mechanism_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_csv = Path("results") / run_id / "mechanism_results.csv"
    status_json = Path("results") / run_id / "mechanism_status.json"

    gpus = parse_gpu_ids(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs provided.")
    max_concurrent = args.max_concurrent or len(gpus)
    max_concurrent = calc_effective_max_concurrent(max_concurrent, gpus, args.max_per_gpu)
    jobs = build_jobs(args, run_id)

    print(f"Run ID: {run_id}")
    print(f"Jobs: {len(jobs)}")
    print(f"GPUs: {gpus} max_concurrent={max_concurrent} max_per_gpu={args.max_per_gpu}")
    print(f"Result CSV: {result_csv}")
    if args.dry_run:
        for item in jobs:
            print(f"[DRY] {item.phase}/{item.job.dataset}/{item.variant}: {' '.join(item.job.cmd)}")
        update_status(status_json, {"run_id": run_id, "dry_run": True, "total_jobs": len(jobs)})
        return

    running: List[Tuple[subprocess.Popen, int, MechanismJob]] = []
    gpu_load: Dict[int, int] = {gid: 0 for gid in gpus}
    rr = 0
    idx = 0
    completed = 0
    launched = 0
    poll_interval = max(2, int(args.poll_interval))

    while idx < len(jobs) or running:
        alive: List[Tuple[subprocess.Popen, int, MechanismJob]] = []
        completed_rows: List[Dict[str, Any]] = []
        for proc, gpu, mech_job in running:
            ret = proc.poll()
            if ret is None:
                alive.append((proc, gpu, mech_job))
                continue
            gpu_load[gpu] = max(0, gpu_load.get(gpu, 0) - 1)
            row = collect_result(mech_job.job, ret)
            row = _supplement_row(row, mech_job, run_id)
            completed_rows.append(row)
            completed += 1
            print(
                f"[DONE] {mech_job.phase}/{mech_job.job.dataset}/{mech_job.variant} "
                f"gpu={gpu} exit={ret} test_auc={row.get('test_auc')} best_val={row.get('best_val_auc')}"
            )
        running = alive
        append_rows(result_csv, completed_rows)

        while idx < len(jobs) and len(running) < max_concurrent:
            gpu_id, rr = pick_gpu_with_slot_round_robin(gpus, gpu_load, args.max_per_gpu, rr)
            if gpu_id is None:
                break
            mech_job = jobs[idx]
            print(f"[PLAN] {mech_job.phase}/{mech_job.job.dataset}/{mech_job.variant} gpu={gpu_id}")
            print("       CMD:", " ".join(mech_job.job.cmd))
            env = _build_job_env(os.environ.copy(), gpu_id)
            proc = subprocess.Popen(mech_job.job.cmd, env=env)
            running.append((proc, gpu_id, mech_job))
            gpu_load[gpu_id] = gpu_load.get(gpu_id, 0) + 1
            launched += 1
            idx += 1

        update_status(
            status_json,
            {
                "run_id": run_id,
                "total_jobs": len(jobs),
                "completed_jobs": completed,
                "running_jobs": len(running),
                "launched_jobs": launched,
                "gpus": gpus,
                "result_csv": str(result_csv),
            },
        )
        if running:
            time.sleep(poll_interval)
        elif idx < len(jobs):
            time.sleep(1)

    update_status(
        status_json,
        {
            "run_id": run_id,
            "total_jobs": len(jobs),
            "completed_jobs": completed,
            "running_jobs": 0,
            "launched_jobs": launched,
            "gpus": gpus,
            "result_csv": str(result_csv),
        },
    )
    print(f"[DONE] completed={completed} launched={launched} total={len(jobs)}")
    run_postprocess(result_csv, args)


if __name__ == "__main__":
    main()
