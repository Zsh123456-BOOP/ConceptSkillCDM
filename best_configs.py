#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
best_configs.py
 BEST  run_all_datasets.py / run_ablation.py / run_module3_grid.py 
"""

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
        "exercise_dim": 64,
        "skill_dim": 128,
        "num_gnn_layers": 1,
        "num_relation_heads": 2,

        # Module1
        "lambda_sparse": 1.0,
        "lambda_sparse_personal": 0.005,
        "lambda_alpha": 0.01,
        "use_personal_graph": True,

        # Module3 prototype (Direction-2: default OFF)
        "num_prototypes": 0,
        "proto_tau": 1.0,
        "proto_lambda": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "enable_soft_prototype": False,
        "disable_soft_prototype": False,
        "use_soft_prototype_main_path": False,

        # Module3 residual / fusion
        "exercise_l2_lambda": 5e-5,
        "fusion_gate_max": 0.4,
        "fusion_gate_bias_init": -2.5,
        "residual_clip_t": 2.0,
        "disable_q_aligned_residual": False,

        # Data filtering
        "min_stu_interactions": 15,
        "min_exer_interactions": 0,
        "min_poison_count": 0,

        # Legacy ablation compatibility
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,

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
        "exercise_dim": 128,
        "skill_dim": 64,
        "num_gnn_layers": 2,
        "num_relation_heads": 2,

        # Module1
        "lambda_sparse": 1.0,
        "lambda_sparse_personal": 0.005,
        "lambda_alpha": 0.01,
        "use_personal_graph": True,

        # Module3 prototype (Direction-2: default OFF)
        "num_prototypes": 0,
        "proto_tau": 1.0,
        "proto_lambda": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "enable_soft_prototype": False,
        "disable_soft_prototype": False,
        "use_soft_prototype_main_path": False,

        # Module3 residual / fusion
        "exercise_l2_lambda": 5e-5,
        "fusion_gate_max": 0.4,
        "fusion_gate_bias_init": -2.5,
        "residual_clip_t": 2.0,
        "disable_q_aligned_residual": False,

        # Data filtering
        "min_stu_interactions": 15,
        "min_exer_interactions": 0,
        "min_poison_count": 0,

        # Legacy ablation compatibility
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,

        # Misc
        "model_variant": "gpd_base",
    },
}

DEFAULT_SEEDS = [42]


