# src/config.py
"""
Dataset defaults and optional grid-search space.

Notes:
- `lambda_sparse` is interpreted as graph sparsity regularization weight at CLI level.
- In model code it is mapped to `lambda_graph_entropy`.
"""

DATASET_DEFAULTS = {
    "assist_09": {
        "data_dir": "./data/assist_09",
        "batch_size": 512,
        "knowledge_dim": 128,
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
        "graph_identity_residual": 0.05,
        "lambda_sparse_personal": 0.0005,
        "lambda_alpha": 0.0,
        "lambda_alpha_min": 0.08,
        "alpha_min_target": 0.05,
        "use_personal_graph": True,
        "personal_rank": 8,
        "personal_max_alpha": 0.40,
        "personal_delta_scale": 6.0,
        "personal_warmup_epochs": 8,
        "personal_reg_warmup_epochs": 8,
        "personal_student_dim": 32,
        "personal_alpha_bias_scale": 0.03,
        "personal_direct_bias_scale": 0.0,
        "personal_disable_direct_bias": True,
        "personal_disable_student_global_context": True,
        "share_concept_embeddings": True,
        "prediction_l2_lambda": 1e-5,
        "debug_graph_diag": True,
        "diag_batches": 1,
        "graph_topk": 12,
        "disable_self_loop": False,
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
        "graph_identity_residual": 0.05,
        "lambda_sparse_personal": 0.002,
        "lambda_alpha": 0.05,
        "lambda_alpha_min": 0.05,
        "alpha_min_target": 0.04,
        "use_personal_graph": True,
        "personal_warmup_epochs": 6,
        "personal_reg_warmup_epochs": 6,
        "personal_alpha_bias_scale": 0.03,
        "personal_direct_bias_scale": 0.0,
        "personal_disable_direct_bias": True,
        "personal_disable_student_global_context": True,
        "share_concept_embeddings": True,
        "prediction_l2_lambda": 1e-5,
        "debug_graph_diag": True,
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
        "graph_identity_residual": 0.05,
        "lambda_sparse_personal": 0.0005,
        "lambda_alpha": 0.06,
        "lambda_alpha_min": 0.05,
        "alpha_min_target": 0.04,
        "use_personal_graph": True,
        "personal_rank": 8,
        "personal_max_alpha": 0.35,
        "personal_delta_scale": 4.0,
        "personal_warmup_epochs": 5,
        "personal_reg_warmup_epochs": 5,
        "personal_student_dim": 64,
        "personal_alpha_bias_scale": 0.015,
        "personal_direct_bias_scale": 0.0,
        "personal_disable_direct_bias": True,
        "personal_disable_student_global_context": True,
        "share_concept_embeddings": True,
        "prediction_l2_lambda": 5e-5,
        "debug_graph_diag": True,
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
            "ablate_concept_graph": [False],
        },
    },
    "assist_17": {
        "base": {},
        "search": {
            "learning_rate": [1e-4, 3e-4],
        },
    },
    "junyi": {
        "base": {},
        "search": {
            "learning_rate": [3e-4, 1e-3],
            "lambda_sparse": [0.05, 0.1],
            "dropout": [0.1, 0.2],
            "ablate_concept_graph": [False],
        },
    },
}


def collect_explicit_arg_dests(argv, parser=None):
    """
    Collect argparse dest names that were explicitly provided on CLI.

    This is needed because a user may intentionally pass a value that equals
    the argparse default, for example `--lambda_alpha 0.0`. In that case we
    must still treat the field as explicitly overridden and avoid replacing it
    with dataset defaults.
    """
    if parser is None:
        return set()

    option_to_dest = {}
    for action in getattr(parser, "_actions", []):
        for opt in getattr(action, "option_strings", []):
            option_to_dest[opt] = action.dest

    explicit = set()
    for token in argv or []:
        if not isinstance(token, str):
            continue
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        opt = token.split("=", 1)[0]
        dest = option_to_dest.get(opt)
        if dest:
            explicit.add(dest)
    return explicit


def apply_dataset_defaults(args, parser=None, explicit_dests=None):
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

    explicit_dests = set(explicit_dests or ())
    defaults = DATASET_DEFAULTS[dataset_name]
    for key, value in defaults.items():
        if key in explicit_dests:
            continue
        if parser is not None and parser.get_default(key) is not None:
            if getattr(args, key, None) == parser.get_default(key):
                setattr(args, key, value)
        else:
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    return args
