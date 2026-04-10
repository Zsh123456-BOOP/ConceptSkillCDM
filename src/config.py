# src/config.py
"""
Dataset defaults and optional grid-search space.

Notes:
- `lambda_sparse` is still interpreted as graph sparsity regularization weight at CLI level.
- In model code it is mapped to `lambda_graph_entropy`.
"""

DATASET_DEFAULTS = {
    "assist_09": {
        "data_dir": "./data/assist_09",
        "batch_size": 512,
        "knowledge_dim": 128,
        "exercise_dim": 128,
        "skill_dim": 64,
        "num_relation_heads": 4,
        "num_gnn_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lambda_sparse": 0.3,
        "graph_entropy_min": 0.15,
        "graph_entropy_max": 0.80,
        "lambda_graph_diag": 0.10,
        "lambda_graph_uniform": 0.04,
        "graph_uniform_margin": 0.10,
        "graph_reg_warmup_epochs": 0,
        "graph_reg_cap_ratio": 6.0,
        "graph_dropout": 0.0,
        "graph_tau_init": 0.55,
        "lambda_sparse_personal": 0.002,
        "lambda_alpha": 0.05,
        "use_personal_graph": 0,
        "exercise_l2_lambda": 1e-5,
        "fusion_gate_max": 1.0,
        "fusion_gate_bias_init": -0.9,
        "residual_clip_t": 2.0,
        "residual_scale_init": 0.25,
        "disable_q_aligned_residual": False,
        "debug_module3_diag": True,
        "diag_batches": 1,
        "graph_topk": 12,
        "disable_q_conditioning": False,
        "disable_b_id_adapter": False,
        "disable_b_bias": False,
        "lambda_b_id_budget": 0.0,
        "b_id_budget_target": 0.25,
        "disable_self_loop": False,
        "personal_rank": 4,
        "gnn_residual_weight": 0.5,
    },
    "assist_17": {
        "data_dir": "./data/assist_17",
        "batch_size": 512,
        "learning_rate": 1e-4,
        "lambda_sparse": 0.6,
        "graph_entropy_min": 0.15,
        "graph_entropy_max": 0.80,
        "lambda_graph_diag": 0.10,
        "lambda_graph_uniform": 0.04,
        "graph_uniform_margin": 0.10,
        "graph_reg_warmup_epochs": 0,
        "graph_reg_cap_ratio": 6.0,
        "graph_dropout": 0.0,
        "graph_tau_init": 0.55,
        "lambda_sparse_personal": 0.002,
        "lambda_alpha": 0.05,
        "use_personal_graph": 0,
        "exercise_l2_lambda": 1e-5,
        "fusion_gate_max": 1.0,
        "fusion_gate_bias_init": -0.6,
        "residual_clip_t": 2.0,
        "residual_scale_init": 0.25,
        "disable_q_aligned_residual": False,
        "disable_b_id_adapter": False,
        "disable_b_bias": False,
        "lambda_b_id_budget": 0.0,
        "b_id_budget_target": 0.25,
        "debug_module3_diag": True,
        "diag_batches": 1,
        "graph_topk": 24,
    },
    "junyi": {
        "data_dir": "./data/junyi",
        "batch_size": 512,
        "learning_rate": 1e-4,
        "lambda_sparse": 0.01,
        "graph_entropy_min": 0.15,
        "graph_entropy_max": 0.70,
        "lambda_graph_diag": 0.12,
        "lambda_graph_uniform": 0.06,
        "graph_uniform_margin": 0.12,
        "graph_reg_warmup_epochs": 1,
        "graph_reg_cap_ratio": 6.0,
        "graph_dropout": -1.0,
        "graph_tau_init": 1.0,
        "lambda_sparse_personal": 0.0,
        "lambda_alpha": 0.0,
        "use_personal_graph": 0,
        "exercise_l2_lambda": 5e-5,
        "fusion_gate_max": 1.0,
        "fusion_gate_bias_init": -1.1,
        "residual_clip_t": 2.0,
        "residual_scale_init": 0.1,
        "disable_q_aligned_residual": False,
        "disable_b_id_adapter": False,
        "disable_b_bias": False,
        "lambda_b_id_budget": 0.0,
        "b_id_budget_target": 0.25,
        "debug_module3_diag": True,
        "diag_batches": 1,
    },
}

GRID_SEARCH_SPACE = {
    "assist_09": {
        "base": {},
        "search": {
            "learning_rate": [1e-4, 3e-4],
            "lambda_sparse": [0.05, 0.1],
            "dropout": [0.1, 0.2],
            "ablate_skill_encoder": [False],
            "ablate_concept_graph": [False],
        },
    },
    "assist_17": {
        "base": {},
        "search": {
            "learning_rate": [1e-4, 3e-4],
            "ablate_skill_encoder": [False],
        },
    },
    "junyi": {
        "base": {},
        "search": {
            "learning_rate": [3e-4, 1e-3],
            "lambda_sparse": [0.05, 0.1],
            "dropout": [0.1, 0.2],
            "ablate_skill_encoder": [False],
            "ablate_concept_graph": [False],
        },
    },
}


def apply_dataset_defaults(args, parser=None):
    """
    Apply dataset defaults only when user did not manually override.

    If parser is provided:
      - overwrite only when current value equals argparse default.
    Else:
      - overwrite only when attribute missing or None.
    """
    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name is None or dataset_name not in DATASET_DEFAULTS:
        return args

    defaults = DATASET_DEFAULTS[dataset_name]
    for key, value in defaults.items():
        if parser is not None and parser.get_default(key) is not None:
            if getattr(args, key, None) == parser.get_default(key):
                setattr(args, key, value)
        else:
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    return args
