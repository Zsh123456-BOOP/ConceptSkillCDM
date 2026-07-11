"""Dataset defaults for the Graph-IRT training entry point."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Set


_BASE_DEFAULTS: Dict[str, Any] = {
    "batch_size": 512,
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "optimizer": "adam",
    "dropout": 0.10,
    "patience": 5,
    "early_stop_patience": 5,
    "min_epochs": 0,
    "early_stop_min_delta": 1e-5,
    "save_interval": 10,
    "num_workers": 4,
    "no_cuda": False,
    "seed": 42,
    "knowledge_dim": 128,
    "num_relation_heads": 4,
    "num_gnn_layers": 2,
    "graph_topk": 24,
    "disable_self_loop": False,
    "gnn_residual_weight": 0.5,
    "graph_identity_residual": 0.05,
    "graph_propagation_alpha": 0.20,
    "graph_prior_strength_init": 0.35,
    "student_concept_interaction": "none",
    "student_concept_interaction_scale": 1.0,
    "student_concept_interaction_rank": 8,
    "student_concept_interaction_init_std": 0.1,
    "graph_tau_init": 0.55,
    "graph_dropout": 0.0,
    "lambda_graph_entropy": 0.30,
    "graph_reg_warmup_epochs": 0,
    "graph_reg_cap_ratio": 6.0,
    "graph_entropy_min": 0.15,
    "graph_entropy_max": 0.80,
    "lambda_graph_diag": 0.10,
    "lambda_graph_uniform": 0.04,
    "graph_uniform_margin": 0.10,
    "prediction_l2_lambda": 1e-5,
    "graph_prior_mode": "evidence",
    "debug_graph_diag": True,
    "diag_batches": 1,
    "min_stu_interactions": 15,
    "min_exer_interactions": 0,
    "model_variant": "full",
}


def _dataset(data_dir: str, **overrides: Any) -> Dict[str, Any]:
    config = dict(_BASE_DEFAULTS)
    config["data_dir"] = data_dir
    config.update(overrides)
    return config


DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "assist_09": _dataset(
        "./data/assist_09",
        batch_size=128,
        knowledge_dim=64,
        num_relation_heads=2,
        dropout=0.20,
        learning_rate=3e-4,
        early_stop_patience=3,
        lambda_graph_entropy=0.20,
        graph_reg_warmup_epochs=4,
        graph_entropy_min=0.25,
        graph_entropy_max=0.75,
        lambda_graph_diag=0.02,
        lambda_graph_uniform=0.01,
        graph_propagation_alpha=0.02,
        graph_prior_strength_init=0.55,
        graph_topk=12,
    ),
    "assist_17": _dataset(
        "./data/assist_17",
        batch_size=512,
        knowledge_dim=128,
        num_relation_heads=4,
        dropout=0.25,
        learning_rate=1e-3,
        early_stop_patience=3,
        lambda_graph_entropy=0.60,
        graph_propagation_alpha=0.20,
        graph_prior_strength_init=0.35,
        graph_topk=24,
    ),
    "junyi": _dataset(
        "./data/junyi",
        batch_size=256,
        knowledge_dim=128,
        num_relation_heads=2,
        dropout=0.40,
        learning_rate=3e-3,
        weight_decay=1e-3,
        patience=1,
        early_stop_patience=3,
        lambda_graph_entropy=0.60,
        graph_reg_warmup_epochs=2,
        graph_entropy_max=0.70,
        lambda_graph_diag=0.12,
        lambda_graph_uniform=0.06,
        graph_uniform_margin=0.12,
        graph_propagation_alpha=0.25,
        graph_prior_strength_init=0.45,
        graph_topk=24,
    ),
}

DATASET_DEFAULTS["cdbd_lsat"] = {
    **DATASET_DEFAULTS["assist_09"],
    "data_dir": "./data/cdbd_lsat",
    "batch_size": 32,
    "knowledge_dim": 32,
    "num_relation_heads": 2,
    "num_gnn_layers": 1,
    "dropout": 0.10,
    "learning_rate": 1e-3,
    "graph_topk": 4,
}

DATASET_DEFAULTS["cdbd_a0910"] = {
    **DATASET_DEFAULTS["assist_09"],
    "data_dir": "./data/cdbd_a0910",
}

_PUBLIC_BENCHMARK_DEFAULTS = {
    **DATASET_DEFAULTS["assist_09"],
    "min_stu_interactions": 0,
    "min_exer_interactions": 0,
}

DATASET_DEFAULTS["frcsub"] = {
    **_PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/frcsub",
    "batch_size": 128,
    "knowledge_dim": 32,
    "num_relation_heads": 2,
    "num_gnn_layers": 1,
    "dropout": 0.10,
    "learning_rate": 1e-3,
    "graph_topk": 7,
}

DATASET_DEFAULTS["math2"] = {
    **_PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/math2",
    "batch_size": 256,
    "knowledge_dim": 32,
    "num_relation_heads": 2,
    "num_gnn_layers": 1,
    "dropout": 0.10,
    "learning_rate": 1e-3,
    "graph_topk": 8,
}

DATASET_DEFAULTS["assist_12"] = {
    **_PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/assist_12",
    "batch_size": 1024,
    "graph_topk": 24,
}

DATASET_DEFAULTS["assist_15"] = {
    **_PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/assist_15",
    "batch_size": 1024,
    "knowledge_dim": 64,
    "num_relation_heads": 2,
    "graph_topk": 24,
}

DATASET_DEFAULTS["nips34"] = {
    **_PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/nips34",
    "batch_size": 1024,
    "graph_topk": 24,
}

DATASET_DEFAULTS["nips34_l3"] = {
    **DATASET_DEFAULTS["nips34"],
    "data_dir": "./data/nips34_l3",
}

DATASET_DEFAULTS["ednet_kt1"] = {
    **_PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/ednet_kt1",
    "batch_size": 4096,
    "knowledge_dim": 64,
    "num_relation_heads": 2,
    "num_gnn_layers": 1,
    "dropout": 0.10,
    "learning_rate": 1e-4,
    "early_stop_patience": 8,
    "patience": 8,
    "graph_topk": 24,
    "graph_propagation_alpha": 0.06,
    "graph_prior_strength_init": 0.55,
}

DATASET_DEFAULTS["assist_12_clean15_item50"] = {
    **DATASET_DEFAULTS["assist_12"],
    "data_dir": "./data/assist_12_clean15_item50",
}

DATASET_DEFAULTS["ednet_kt1_clean15_sample5000"] = {
    **DATASET_DEFAULTS["ednet_kt1"],
    "data_dir": "./data/ednet_kt1_clean15_sample5000",
}

DATASET_DEFAULTS["ednet_kt1_gap"] = {
    **DATASET_DEFAULTS["ednet_kt1"],
    "data_dir": "./data/ednet_kt1_gap",
}

for _name in (
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
    DATASET_DEFAULTS[_name] = {
        **DATASET_DEFAULTS["ednet_kt1"],
        "data_dir": f"./data/{_name}",
    }

for _name, _base in (
    ("assist_09_chold", "assist_09"),
    ("assist_09_coldconcept", "assist_09"),
    ("junyi_chold", "junyi"),
    ("assist_17_chold", "assist_17"),
    ("assist_17_coldconcept", "assist_17"),
):
    DATASET_DEFAULTS[_name] = {
        **DATASET_DEFAULTS[_base],
        "data_dir": f"./data/{_name}",
    }


def collect_explicit_arg_dests(argv: Iterable[object], parser=None) -> Set[str]:
    """Return argparse destinations explicitly supplied on the command line."""
    if parser is None:
        return set()

    option_to_dest = {
        option: action.dest
        for action in getattr(parser, "_actions", ())
        for option in getattr(action, "option_strings", ())
    }
    explicit: Set[str] = set()
    for token in argv or ():
        if not isinstance(token, str):
            continue
        if token == "--":
            break
        if token.startswith("-"):
            destination = option_to_dest.get(token.split("=", 1)[0])
            if destination:
                explicit.add(destination)
    return explicit


def apply_dataset_defaults(args, parser=None, explicit_dests=None):
    """Apply dataset defaults without overwriting explicit CLI arguments."""
    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name not in DATASET_DEFAULTS:
        return args

    explicit = set(explicit_dests or ())
    for key, value in DATASET_DEFAULTS[dataset_name].items():
        if key in explicit:
            continue
        if parser is None:
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
            continue

        parser_default = parser.get_default(key)
        current = getattr(args, key, None)
        if current == parser_default:
            setattr(args, key, value)
    return args
