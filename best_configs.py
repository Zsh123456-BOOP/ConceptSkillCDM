#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Best configs for AE-focused training/ablation."""

from typing import Any, Dict

BEST_CFG: Dict[str, Dict[str, Any]] = {
    "assist_09": {
        # Train
        "seed": 42,
        "batch_size": 128,
        "epochs": 100,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "dropout": 0.20,
        "early_stop_patience": 3,
        "patience": 5,
        "save_interval": 10,
        "num_workers": 4,
        "no_cuda": False,

        # Model base
        "knowledge_dim": 64,
        "num_gnn_layers": 1,
        "num_relation_heads": 2,

        # Module1
        "lambda_sparse": 0.8,
        "lambda_sparse_personal": 0.0005,
        "lambda_alpha": 0.08,
        "lambda_alpha_min": 0.08,
        "alpha_min_target": 0.05,
        "use_personal_graph": True,
        "graph_entropy_min": 0.25,
        "graph_entropy_max": 0.75,
        "lambda_graph_diag": 0.10,
        "lambda_graph_uniform": 0.04,
        "graph_uniform_margin": 0.10,
        "graph_reg_warmup_epochs": 4,
        "graph_reg_cap_ratio": 6.0,
        "graph_dropout": 0.0,
        "graph_tau_init": 0.55,
        "graph_identity_residual": 0.10,
        "graph_topk": 12,
        "personal_rank": 8,
        "personal_max_alpha": 0.40,
        "personal_delta_scale": 6.0,
        "personal_warmup_epochs": 8,
        "personal_student_dim": 32,

        # Fixed prediction head
        "prediction_l2_lambda": 1e-5,
        "debug_graph_diag": True,
        "diag_batches": 1,

        # Data filtering
        "min_stu_interactions": 15,
        "min_exer_interactions": 0,
        "min_poison_count": 0,

        # Misc
        "model_variant": "gpd_base",
    },
    "junyi": {
        # Train
        "seed": 42,
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.003,
        "weight_decay": 0.001,
        "dropout": 0.40,
        "early_stop_patience": 3,
        "patience": 1,
        "save_interval": 10,
        "num_workers": 4,
        "no_cuda": False,

        # Model base
        "knowledge_dim": 128,
        "num_gnn_layers": 2,
        "num_relation_heads": 2,

        # Module1
        "lambda_sparse": 0.6,
        "lambda_sparse_personal": 0.0005,
        "lambda_alpha": 0.06,
        "lambda_alpha_min": 0.05,
        "alpha_min_target": 0.04,
        "use_personal_graph": True,
        "graph_entropy_min": 0.15,
        "graph_entropy_max": 0.70,
        "lambda_graph_diag": 0.12,
        "lambda_graph_uniform": 0.06,
        "graph_uniform_margin": 0.12,
        "graph_reg_warmup_epochs": 2,
        "graph_reg_cap_ratio": 6.0,
        "graph_dropout": 0.0,
        "graph_tau_init": 0.55,
        "graph_identity_residual": 0.10,
        "graph_topk": 24,
        "personal_rank": 8,
        "personal_max_alpha": 0.35,
        "personal_delta_scale": 4.0,
        "personal_warmup_epochs": 5,
        "personal_student_dim": 64,

        # Fixed prediction head
        "prediction_l2_lambda": 1e-5,
        "debug_graph_diag": True,
        "diag_batches": 1,

        # Data filtering
        "min_stu_interactions": 15,
        "min_exer_interactions": 0,
        "min_poison_count": 0,

        # Misc
        "model_variant": "gpd_base",
    },
}

DEFAULT_SEEDS = [42]
