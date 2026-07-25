#!/usr/bin/env python
"""Collect validation-selection outcomes from targeted experiment checkpoints.

The script never opens a test split.  It reads each checkpoint's stored
train/validation metrics and, when present, ``selection_manifest.json``.  The
manifest keeps the immutable parent, trained candidate, and deployment-selected
metrics distinct so an epoch-zero fallback is not mistaken for an active
residual.  Legacy v3 scalar residuals, v4 route/quality parameters, and v5
branch routes/scales are summarized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
METRIC_NAMES = ("auc", "acc", "rmse", "bce_loss", "loss")
RESIDUAL_MODEL_VARIANTS = frozenset(
    {"gec_residual", "gec_relation_residual", "gec_branch_gate"}
)


def _tokens(value: str) -> List[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _candidate_dirs(run_ids: Iterable[str], explicit_dirs: Iterable[str]) -> List[Path]:
    candidates = {
        path.resolve()
        for run_id in run_ids
        for path in CHECKPOINTS.glob(f"*_{run_id}")
        if path.is_dir()
    }
    for value in explicit_dirs:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {path}")
        candidates.add(path.resolve())
    return sorted(candidates)


def _metric(row: Dict[str, object], prefix: str, metrics: Dict[str, object]) -> None:
    for name in METRIC_NAMES:
        value = metrics.get(name)
        if value is not None:
            row[f"{prefix}_{name}"] = float(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(*values: Any) -> Any:
    return next(
        (
            value
            for value in values
            if value is not None and value != ""
        ),
        None,
    )


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "accepted", "candidate"}:
        return True
    if normalized in {"false", "no", "rejected", "parent"}:
        return False
    raise ValueError(f"cannot interpret selection boolean {value!r}")


def _role_node(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    direct = manifest.get(role)
    if isinstance(direct, Mapping):
        return direct
    for container_name in ("checkpoints", "artifacts"):
        nested = _mapping(manifest.get(container_name)).get(role)
        if isinstance(nested, Mapping):
            return nested
    return {}


def _role_metrics(
    manifest: Mapping[str, Any],
    role: str,
) -> Dict[str, Any]:
    node = _role_node(manifest, role)
    for value in (
        node.get("val_metrics"),
        node.get("validation_metrics"),
        node.get("metrics"),
        manifest.get(f"{role}_val_metrics"),
        manifest.get(f"{role}_validation_metrics"),
        manifest.get(f"{role}_metrics"),
    ):
        if isinstance(value, Mapping):
            return {
                name: value[name]
                for name in METRIC_NAMES
                if value.get(name) is not None
            }
    if any(node.get(name) is not None for name in METRIC_NAMES):
        return {
            name: node[name]
            for name in METRIC_NAMES
            if node.get(name) is not None
        }
    return {}


def _resolve_artifact(run_dir: Path, value: Any) -> Optional[Path]:
    if value is None or value == "":
        return None
    path = Path(str(value))
    candidates = (
        (path,)
        if path.is_absolute()
        else (run_dir / path, ROOT / path)
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _role_checkpoint_path(
    run_dir: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Optional[Path]:
    node = _role_node(manifest, role)
    selection = _mapping(manifest.get("selection"))
    value = _first_present(
        node.get("checkpoint"),
        node.get("checkpoint_path"),
        node.get("path"),
        manifest.get(f"{role}_checkpoint"),
        manifest.get(f"{role}_checkpoint_path"),
        selection.get(f"{role}_checkpoint"),
        selection.get(f"{role}_checkpoint_path"),
    )
    resolved = _resolve_artifact(run_dir, value)
    if resolved is not None:
        return resolved
    defaults = {
        "parent": ("parent_fallback.pth", "fallback_parent.pth"),
        "candidate": ("candidate_best.pth",),
        "selected": ("selected_model.pth", "best_model.pth"),
    }
    return next(
        (
            run_dir / name
            for name in defaults.get(role, ())
            if (run_dir / name).is_file()
        ),
        None,
    )


def _load_checkpoint(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint payload must be a mapping: {path}")
    return checkpoint


def _checkpoint_val_metrics(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = checkpoint.get("val_metrics")
    if not isinstance(metrics, Mapping):
        return {}
    return {
        name: metrics[name]
        for name in METRIC_NAMES
        if metrics.get(name) is not None
    }


def _merged_role_metrics(
    manifest: Mapping[str, Any],
    role: str,
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = _checkpoint_val_metrics(checkpoint)
    metrics.update(_role_metrics(manifest, role))
    return metrics


def _state_tensor(
    checkpoint: Mapping[str, Any],
    name: str,
) -> Optional[torch.Tensor]:
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        return None
    value = state_dict.get(name)
    if value is None:
        value = state_dict.get(f"module.{name}")
    return None if value is None else torch.as_tensor(value).detach().cpu().float()


def _tensor_json(value: torch.Tensor) -> str:
    return json.dumps(value.tolist(), separators=(",", ":"))


def _residual_parameters(checkpoint: Mapping[str, Any]) -> Dict[str, object]:
    """Return compact v3/v4/v5 diagnostics from one checkpoint."""
    rho = _state_tensor(checkpoint, "evidence_residual.rho")
    route_raw = _state_tensor(checkpoint, "evidence_residual.route_raw")
    state_route_raw = _state_tensor(
        checkpoint,
        "evidence_residual.state_route_raw",
    )
    propagation_route_raw = _state_tensor(
        checkpoint,
        "evidence_residual.propagation_route_raw",
    )
    quality_weight = _state_tensor(
        checkpoint,
        "evidence_residual.quality_weight",
    )
    quality_bias = _state_tensor(checkpoint, "evidence_residual.quality_bias")
    quality_hidden_weight = _state_tensor(
        checkpoint,
        "evidence_residual.quality_hidden_weight",
    )
    quality_hidden_bias = _state_tensor(
        checkpoint,
        "evidence_residual.quality_hidden_bias",
    )
    quality_output_weight = _state_tensor(
        checkpoint,
        "evidence_residual.quality_output_weight",
    )
    quality_output_bias = _state_tensor(
        checkpoint,
        "evidence_residual.quality_output_bias",
    )
    raw_support_quality = quality_hidden_weight is not None
    result: Dict[str, object] = {}
    if (state_route_raw is None) != (propagation_route_raw is None):
        raise ValueError(
            "branch-gate checkpoint must contain both state_route_raw and "
            "propagation_route_raw"
        )
    if state_route_raw is not None and propagation_route_raw is not None:
        if state_route_raw.numel() != 1:
            raise ValueError(
                "branch-gate state_route_raw must contain exactly one value"
            )
        state_raw_value = float(state_route_raw.reshape(-1)[0].item())
        state_route_value = float(
            torch.tanh(state_route_raw.reshape(-1)[0]).item()
        )
        propagation_route = torch.tanh(propagation_route_raw)
        propagation_scale = 1.0 + propagation_route
        combined_raw = torch.cat(
            (state_route_raw.reshape(1), propagation_route_raw.reshape(-1))
        )
        combined_route = torch.tanh(combined_raw)
        combined_scale = 1.0 + combined_route
        branch_names = [
            "state",
            *[
                f"propagation_{index}"
                for index in range(propagation_route_raw.numel())
            ],
        ]
        result.update(
            {
                "kind": "branch_gate_v5",
                "state_route_raw": state_raw_value,
                "state_route": state_route_value,
                "state_branch_scale": 1.0 + state_route_value,
                "propagation_route_raw_json": _tensor_json(
                    propagation_route_raw
                ),
                "propagation_route_json": _tensor_json(propagation_route),
                "propagation_branch_scale_json": _tensor_json(
                    propagation_scale
                ),
                "branch_route_raw_json": json.dumps(
                    dict(zip(branch_names, combined_raw.tolist())),
                    separators=(",", ":"),
                ),
                "branch_route_json": json.dumps(
                    dict(zip(branch_names, combined_route.tolist())),
                    separators=(",", ":"),
                ),
                "branch_scale_json": json.dumps(
                    dict(zip(branch_names, combined_scale.tolist())),
                    separators=(",", ":"),
                ),
                "route_abs_max": float(combined_route.abs().max().item()),
                "route_l2": float(
                    combined_route.square().sum().sqrt().item()
                ),
                "branch_scale_min": float(combined_scale.min().item()),
                "branch_scale_max": float(combined_scale.max().item()),
            }
        )
    if rho is not None:
        rho_value = float(rho.reshape(-1)[0].item())
        result.update(
            {
                "kind": (
                    "v4_raw_support_quality"
                    if raw_support_quality
                    else "v3_rho"
                ),
                "rho": rho_value,
                "alpha": float(
                    0.20 * torch.tanh(torch.tensor(rho_value)).item()
                ),
            }
        )
    if route_raw is not None:
        route = torch.tanh(route_raw)
        result.update(
            {
                "kind": "v4_route_quality",
                "route_raw_json": _tensor_json(route_raw),
                "route_json": _tensor_json(route),
                "route_abs_max": float(route.abs().max().item()),
                "route_l2": float(route.square().sum().sqrt().item()),
            }
        )
    if quality_weight is not None:
        result.update(
            {
                "quality_weight_json": _tensor_json(quality_weight),
                "quality_weight_l2": float(
                    quality_weight.square().sum().sqrt().item()
                ),
            }
        )
    if quality_bias is not None:
        result.update(
            {
                "quality_bias_json": _tensor_json(quality_bias),
                "quality_bias_abs_max": float(quality_bias.abs().max().item()),
            }
        )
    for name, value in (
        ("quality_hidden_weight", quality_hidden_weight),
        ("quality_hidden_bias", quality_hidden_bias),
        ("quality_output_weight", quality_output_weight),
        ("quality_output_bias", quality_output_bias),
    ):
        if value is None:
            continue
        result[f"{name}_json"] = _tensor_json(value)
        result[f"{name}_l2"] = float(
            value.square().sum().sqrt().item()
        )
    return result


def _add_parameter_fields(
    row: Dict[str, object],
    prefix: str,
    values: Mapping[str, object],
) -> None:
    for name, value in values.items():
        row[f"{prefix}_residual_{name}"] = value


def _role_epoch(
    manifest: Mapping[str, Any],
    role: str,
    checkpoint: Mapping[str, Any],
) -> Optional[int]:
    node = _role_node(manifest, role)
    value = _first_present(
        node.get("epoch"),
        node.get("best_epoch"),
        manifest.get(f"{role}_epoch"),
        manifest.get(f"{role}_best_epoch"),
        checkpoint.get("epoch"),
    )
    return None if value is None else int(value)


def _role_sha(manifest: Mapping[str, Any], role: str) -> str:
    node = _role_node(manifest, role)
    selection = _mapping(manifest.get("selection"))
    value = _first_present(
        node.get("sha256"),
        node.get("checkpoint_sha256"),
        manifest.get(f"{role}_sha256"),
        manifest.get(f"{role}_checkpoint_sha256"),
        selection.get(f"{role}_checkpoint_sha256"),
    )
    return "" if value is None else str(value)


def _read_run(path: Path) -> Dict[str, object]:
    args_path = path / "args.json"
    checkpoint_path = path / "best_model.pth"
    validation_path = path / "validation_result.json"
    for required in (args_path, checkpoint_path, validation_path):
        if not required.is_file():
            raise FileNotFoundError(f"required run artifact is missing: {required}")

    with args_path.open(encoding="utf-8") as handle:
        args = json.load(handle)
    with validation_path.open(encoding="utf-8") as handle:
        validation = json.load(handle)
    manifest_path = path / "selection_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise TypeError(
                f"selection manifest must contain a JSON object: {manifest_path}"
            )
    else:
        manifest = {}

    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_args = checkpoint.get("args", {})
    parent_val_auc = float(
        checkpoint_args.get("warm_start_parent_val_auc", float("nan"))
    )
    role_paths = {
        role: _role_checkpoint_path(path, manifest, role)
        for role in ("parent", "candidate", "selected")
    }
    role_checkpoints = {
        role: (
            checkpoint
            if role == "selected" and role_path == checkpoint_path
            else _load_checkpoint(role_path)
        )
        for role, role_path in role_paths.items()
    }
    selected_checkpoint = role_checkpoints["selected"] or checkpoint
    selected_metrics = _merged_role_metrics(
        manifest,
        "selected",
        selected_checkpoint,
    )
    if not selected_metrics:
        selected_metrics = _checkpoint_val_metrics(checkpoint)
    parent_metrics = _merged_role_metrics(
        manifest,
        "parent",
        role_checkpoints["parent"],
    )
    if not parent_metrics and parent_val_auc == parent_val_auc:
        parent_metrics["auc"] = parent_val_auc
    candidate_metrics = _merged_role_metrics(
        manifest,
        "candidate",
        role_checkpoints["candidate"],
    )

    selection = _mapping(manifest.get("selection"))
    selected_node = _role_node(manifest, "selected")
    thresholds = _mapping(manifest.get("thresholds"))
    deltas = _mapping(manifest.get("deltas"))
    zero_identity = _mapping(manifest.get("zero_identity"))
    model_variant = str(
        args.get("model_variant", validation.get("model_variant", "full"))
    )
    best_epoch = int(
        checkpoint.get("epoch", validation.get("best_epoch", 0))
    )
    accepted_value = _first_present(
        manifest.get("accepted"),
        manifest.get("candidate_accepted"),
        selection.get("accepted"),
        selection.get("candidate_accepted"),
        selected_node.get("accepted"),
    )
    selected_source = _first_present(
        manifest.get("selected_source"),
        manifest.get("selection_source"),
        selection.get("selected_source"),
        selection.get("source"),
        selected_node.get("source"),
        selected_node.get("selected_source"),
    )
    selection_reason = _first_present(
        manifest.get("selection_reason"),
        manifest.get("reason"),
        selection.get("selection_reason"),
        selection.get("reason"),
        selected_node.get("reason"),
    )
    if not manifest and model_variant in RESIDUAL_MODEL_VARIANTS:
        legacy_accepted = best_epoch > 0
        accepted_value = legacy_accepted
        selected_source = "candidate" if legacy_accepted else "parent"
        selection_reason = (
            "legacy_best_epoch_gt_zero"
            if legacy_accepted
            else "legacy_epoch0_fallback"
        )

    if not candidate_metrics and selected_source == "candidate":
        candidate_metrics = dict(selected_metrics)
    residual_active_value = selected_node.get("residual_active")
    if (
        residual_active_value is None
        and model_variant in RESIDUAL_MODEL_VARIANTS
        and selected_source is not None
    ):
        residual_active_value = str(selected_source) == "candidate"
    zero_logit_sha = _first_present(
        zero_identity.get("logit_sha256"),
        zero_identity.get("parent_prediction_sha256"),
        zero_identity.get("zero_residual_prediction_sha256"),
    )

    row: Dict[str, object] = {
        "run_dir": str(path),
        "architecture": str(checkpoint.get("architecture", "")),
        "dataset": str(args.get("dataset_name", validation.get("dataset", ""))),
        "model_variant": model_variant,
        "train_evidence_mode": str(
            args.get(
                "train_evidence_mode",
                validation.get("train_evidence_mode", "excluded"),
            )
        ),
        "gec_mode": str(args.get("gec_mode", validation.get("gec_mode", "v1"))),
        "warm_start_checkpoint_sha256": str(
            checkpoint_args.get("warm_start_checkpoint_sha256", "")
        ),
        "warm_start_parent_val_auc": parent_val_auc,
        "selection_manifest_present": bool(manifest),
        "selection_schema_version": manifest.get("schema_version"),
        "selection_residual_mode": str(
            manifest.get(
                "residual_mode",
                args.get("gec_mode", validation.get("gec_mode", "")),
            )
        ),
        "selection_accepted": _optional_bool(accepted_value),
        "selected_source": (
            "" if selected_source is None else str(selected_source)
        ),
        "selection_reason": (
            "" if selection_reason is None else str(selection_reason)
        ),
        "parent_checkpoint_sha256": _role_sha(manifest, "parent"),
        "candidate_checkpoint_sha256": _role_sha(manifest, "candidate"),
        "selected_checkpoint_sha256": _role_sha(manifest, "selected"),
        "selected_residual_active": _optional_bool(residual_active_value),
        "zero_identity_verified": _optional_bool(
            zero_identity.get("verified")
        ),
        "zero_identity_logit_sha256": (
            "" if zero_logit_sha is None else str(zero_logit_sha)
        ),
        "seed": int(args.get("seed", validation.get("seed", 0))),
        "best_epoch": best_epoch,
    }
    for name, value in thresholds.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            row[f"selection_threshold_{name}"] = value
    for name, value in deltas.items():
        if value is not None:
            row[f"selection_delta_{name}"] = float(value)
    for role, metrics in (
        ("parent", parent_metrics),
        ("candidate", candidate_metrics),
        ("selected", selected_metrics),
    ):
        _metric(row, f"{role}_val", metrics)
        epoch = _role_epoch(
            manifest,
            role,
            role_checkpoints[role] or (
                checkpoint if role == "selected" else {}
            ),
        )
        if epoch is not None:
            row[f"{role}_epoch"] = epoch
        if role_paths[role] is not None:
            row[f"{role}_checkpoint"] = str(role_paths[role])

    for name in METRIC_NAMES:
        parent_value = parent_metrics.get(name)
        candidate_value = candidate_metrics.get(name)
        selected_value = selected_metrics.get(name)
        if parent_value is not None and candidate_value is not None:
            row[f"candidate_parent_{name}_delta"] = (
                float(candidate_value) - float(parent_value)
            )
        if parent_value is not None and selected_value is not None:
            row[f"selected_parent_{name}_delta"] = (
                float(selected_value) - float(parent_value)
            )

    _metric(row, "train", checkpoint.get("train_metrics", {}))
    _metric(row, "val", checkpoint.get("val_metrics", {}))
    if "val_auc" in row and parent_val_auc == parent_val_auc:
        row["val_auc_delta_from_parent"] = float(row["val_auc"]) - parent_val_auc

    candidate_parameters = _residual_parameters(
        role_checkpoints["candidate"]
    )
    selected_parameters = _residual_parameters(selected_checkpoint)
    _add_parameter_fields(row, "candidate", candidate_parameters)
    _add_parameter_fields(row, "selected", selected_parameters)
    legacy_parameters = candidate_parameters or selected_parameters
    for name, value in legacy_parameters.items():
        row[f"residual_{name}"] = value
    # Preserve the original v3 column names for existing analysis notebooks.
    if "rho" in legacy_parameters:
        row["residual_rho"] = legacy_parameters["rho"]
        row["residual_alpha"] = legacy_parameters["alpha"]

    test_path = path / "test_results.json"
    if test_path.is_file():
        with test_path.open(encoding="utf-8") as handle:
            test = json.load(handle)
        _metric(row, "test", test.get("metrics", {}))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_ids",
        default="",
        help="Comma-separated run-id suffixes used by run_graph_ablation.py.",
    )
    parser.add_argument(
        "--checkpoint_dirs",
        default="",
        help="Comma-separated explicit checkpoint directories.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Per-run CSV; a sibling *_summary.csv is written automatically.",
    )
    args = parser.parse_args()

    directories = _candidate_dirs(
        _tokens(args.run_ids),
        _tokens(args.checkpoint_dirs),
    )
    if not directories:
        raise FileNotFoundError("no matching checkpoint directories")

    runs = pd.DataFrame([_read_run(path) for path in directories])
    runs = runs.sort_values(
        [
            "architecture",
            "dataset",
            "model_variant",
            "gec_mode",
            "train_evidence_mode",
            "seed",
        ]
    ).reset_index(drop=True)

    output = Path(args.output_csv)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output, index=False)

    metric_columns = [
        column
        for column in runs.columns
        if column.startswith(
            (
                "train_",
                "val_",
                "test_",
                "parent_val_",
                "candidate_val_",
                "selected_val_",
                "candidate_parent_",
                "selected_parent_",
            )
        )
        and pd.api.types.is_numeric_dtype(runs[column])
    ]
    grouped = runs.groupby(
        [
            "architecture",
            "dataset",
            "model_variant",
            "gec_mode",
            "train_evidence_mode",
        ],
        dropna=False,
    )
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary.insert(0, "n_seeds", grouped.size())
    summary = summary.reset_index()

    summary_path = output.with_name(f"{output.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"runs={len(runs)} -> {output}")
    print(f"groups={len(summary)} -> {summary_path}")


if __name__ == "__main__":
    main()
