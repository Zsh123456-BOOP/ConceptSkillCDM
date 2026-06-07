#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compact tuned overrides for AE-focused training/ablation.

Most CRG/LCRF constants are fixed dataset defaults in ``src.config``.  This
file only records values that differ from those defaults for the selected
best recipe.  Use ``get_effective_config`` when a caller really needs the full
main.py parameter dictionary.
"""

from typing import Any, Dict

from src.config import DATASET_DEFAULTS


MAIN_IMPLICIT_DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "epochs": 100,
    "knowledge_dim": 128,
    "num_relation_heads": 4,
    "num_gnn_layers": 2,
    "weight_decay": 1e-5,
    "patience": 5,
    "num_workers": 4,
    "save_interval": 10,
    "min_stu_interactions": 15,
    "min_exer_interactions": 0,
    "min_poison_count": 0,
    "graph_prior_mode": "evidence",
    "personal_max_alpha": 0.35,
}


MAIN_ASSIST09_OVERRIDES: Dict[str, Any] = {
    "batch_size": 128,
    "learning_rate": 3e-4,
    "dropout": 0.20,
    "early_stop_patience": 3,
    "no_cuda": False,
    "knowledge_dim": 64,
    "num_relation_heads": 2,
    "lambda_sparse": 0.2,
    "graph_entropy_min": 0.25,
    "graph_entropy_max": 0.75,
    "lambda_graph_diag": 0.02,
    "lambda_graph_uniform": 0.01,
    "graph_reg_warmup_epochs": 4,
    "model_variant": "gpd_base",
}


MAIN_JUNYI_OVERRIDES: Dict[str, Any] = {
    "batch_size": 256,
    "early_stop_patience": 3,
    "patience": 1,
    "no_cuda": False,
    "model_variant": "gpd_base",
}


MAIN_ASSIST17_OVERRIDES: Dict[str, Any] = {
    "learning_rate": 0.001,
    "dropout": 0.25,
    "early_stop_patience": 3,
    "no_cuda": False,
    "personal_rank": 8,
    "personal_student_dim": 64,
    "personal_alpha_budget": 0.10,
    "ae_logit_residual_scale": 1.00,
    "ae_logit_residual_clip": 6.00,
    "ae_irt_logit_scale": 0.20,
    "ae_interaction_logit_scale": 1.00,
    "ae_logit_dim": 64,
    "ae_lr_mult": 5.0,
    "ae_stat_prior_scale": 1.0,
    "model_variant": "gpd_base",
}


EDNET_GAP_OVERRIDES: Dict[str, Any] = {
    "early_stop_patience": 8,
    "patience": 8,
    "no_cuda": False,
    "graph_propagation_alpha": 0.06,
    "graph_query_readout_scale": 0.06,
    "graph_query_readout_2hop_scale": 0.02,
    "lambda_personal_kl": 0.01,
    "lambda_personal_query_residual": 0.02,
    "disable_sequence_prior": False,
    "disable_item_prior": False,
    "personal_warmup_epochs": 4,
    "personal_reg_warmup_epochs": 4,
    "lambda_alpha_min": 0.03,
    "alpha_min_target": 0.02,
    "personal_mastery_prior_scale": 1.2,
    "personal_recent_mastery_prior_scale": 0.6,
    "personal_query_correction_scale": 0.25,
    "personal_query_correction_max_ratio": 0.16,
    "personal_query_correction_min_graph_anchor": 0.03,
    "ae_logit_residual_scale": 0.65,
    "roadmap_logit_residual_scale": 0.55,
    "tutor_logit_residual_scale": 1.0,
    "ae_irt_logit_scale": 1.0,
    "ae_lr_mult": 5.0,
    "roadmap_theta_calibration_scale": 0.1,
    "tutor_theta_calibration_scale": 0.2,
    "model_variant": "ednet_gap_base",
}


def get_effective_config(dataset: str) -> Dict[str, Any]:
    """Return the full config that main.py will see after defaults are applied."""
    if dataset not in BEST_CFG:
        raise KeyError(f"Unknown dataset in BEST_CFG: {dataset}")
    cfg = dict(MAIN_IMPLICIT_DEFAULTS)
    if dataset not in DATASET_DEFAULTS:
        cfg.update(BEST_CFG[dataset])
        return cfg
    cfg.update(DATASET_DEFAULTS[dataset])
    cfg.update(BEST_CFG[dataset])
    return cfg


BEST_CFG: Dict[str, Dict[str, Any]] = {
    "assist_09": dict(MAIN_ASSIST09_OVERRIDES),
    "junyi": dict(MAIN_JUNYI_OVERRIDES),
    "assist_17": dict(MAIN_ASSIST17_OVERRIDES),
    "ednet_kt1_gap": dict(EDNET_GAP_OVERRIDES),
}


for _ednet_gap_sweep_name in (
    "ednet_kt1_gap_t70u5000",
    "ednet_kt1_gap_t60u5000",
    "ednet_kt1_gap_t50u5000",
    "ednet_kt1_gap_t65u3000_long",
    "ednet_kt1_gap_small_t70u2000",
    "ednet_kt1_gap_small_t60u2200",
    "ednet_kt1_gap_small_t50u2600",
    "ednet_kt1_gap_small_t65u1200_long",
    "ednet_kt1_chold_t70u2000",
    "ednet_kt1_chold_t70u2500",
    "ednet_kt1_chold_t75u1800",
    "ednet_kt1_chold_t70u1200_long",
):
    BEST_CFG[_ednet_gap_sweep_name] = dict(EDNET_GAP_OVERRIDES)


for _public_chold_name, _public_chold_base in (
    ("assist_09_chold", "assist_09"),
    ("junyi_chold", "junyi"),
    ("assist_17_chold", "assist_17"),
):
    BEST_CFG[_public_chold_name] = dict(BEST_CFG[_public_chold_base])


DEFAULT_SEEDS = [42]
