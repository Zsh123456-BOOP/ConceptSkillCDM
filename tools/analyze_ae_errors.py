"""Export per-sample AE error diagnostics from a saved checkpoint.

The script is intentionally read-only for training state. It reconstructs the
model from the checkpoint, runs the train-seen test split, and writes:
- ae_error_samples.csv: one row per evaluated test sample
- ae_error_summary.json: aggregate metrics and grouped diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import CognitiveDiagnosisDataset
from src.experiment_utils import compute_metrics
from src.model_cdm import CognitiveDiagnosisModel
from src.trainer import _hard_ablation_effective_hparams, _strip_module_prefix


AE_COMPONENT_KEYS = (
    "roadmap_difficulty_logit",
    "roadmap_reliability_logit",
    "roadmap_macro_logit",
    "roadmap_route_difficulty_delta",
    "roadmap_route_reliability_delta",
    "tutor_current_mastery_logit",
    "tutor_current_recent_logit",
    "tutor_route_mastery_logit",
    "tutor_route_recent_logit",
    "tutor_gap_penalty_logit",
    "tutor_local_navigation_logit",
    "tutor_route_mastery_delta",
    "tutor_route_recent_delta",
    "tutor_query_reliability",
    "tutor_route_reliability",
    "map_raw_logit_before_clip",
    "map_raw_logit_after_clip",
)


def _get(dct: Mapping[str, Any], key: str, default: Any) -> Any:
    return dct[key] if key in dct else default


def _resolve_data_dir(dataset_name: str, data_dir: Optional[str]) -> str:
    if data_dir:
        return data_dir
    direct = os.path.join("data", dataset_name)
    if os.path.exists(direct):
        return direct
    aliases = {
        "assist_09": "assist-09",
        "assist_17": "assist-17",
        "junyi": "junyi",
    }
    return os.path.join("data", aliases.get(dataset_name, dataset_name), "process_data")


def _tensor_to_numpy(value: Optional[torch.Tensor], take: int) -> np.ndarray:
    if value is None:
        return np.zeros(take, dtype=np.float32)
    arr = value.detach().cpu().numpy()
    if arr.ndim == 0:
        return np.repeat(np.asarray([float(arr.item())], dtype=np.float32), take)
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] == 1:
        return arr[:take, 0].astype(np.float32, copy=False)
    return arr[:take].mean(axis=1).astype(np.float32, copy=False)


def _bce_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    labels = labels.astype(np.float64)
    return np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))


def _safe_group_stats(frame: pd.DataFrame, group_cols: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if frame.empty:
        return rows
    for key, grp in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        item: Dict[str, Any] = {col: val for col, val in zip(group_cols, key)}
        item.update(
            {
                "n": int(len(grp)),
                "auc": _safe_auc(grp["label"].to_numpy(), grp["prob"].to_numpy()),
                "acc": float((grp["pred"] == grp["label"]).mean()),
                "wrong_rate": float(grp["wrong"].mean()),
                "abs_error_mean": float(grp["abs_error"].mean()),
                "bce_mean": float(grp["bce_total"].mean()),
                "ae_bce_delta_mean": float(grp["ae_bce_delta"].mean()),
                "ae_help_rate": float(grp["ae_helped"].mean()),
                "ae_direction_correct_rate": float(grp["ae_direction_correct"].mean()),
                "ae_abs_mean": float(grp["ae_logit_residual"].abs().mean()),
                "roadmap_macro_abs_mean": float(grp["roadmap_macro_logit"].abs().mean()),
                "tutor_local_abs_mean": float(grp["tutor_local_navigation_logit"].abs().mean()),
                "query_posterior_kl_mean": float(grp["query_row_posterior_kl"].mean()),
            }
        )
        rows.append(item)
    return rows


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    if len(np.unique(labels)) < 2:
        return None
    metrics = compute_metrics(labels, (probs > 0.5).astype(np.float32), probs)
    return float(metrics["auc"])


def _build_model(checkpoint: Mapping[str, Any], device: torch.device) -> CognitiveDiagnosisModel:
    loaded_args: Dict[str, Any] = dict(checkpoint.get("args", {}))
    info = checkpoint["info_dict"]
    q_matrix = info["q_matrix"]
    use_concept_graph = bool(_get(loaded_args, "use_concept_graph", True))
    ablate_module1 = bool(_get(loaded_args, "ablate_module1", False))
    eff_gnn_layers = _hard_ablation_effective_hparams(
        use_concept_graph=use_concept_graph,
        num_gnn_layers=int(_get(loaded_args, "num_gnn_layers", 0)),
    )
    model = CognitiveDiagnosisModel(
        num_students=info["num_students"],
        num_exercises=info["num_exercises"],
        num_concepts=info["num_concepts"],
        q_matrix=q_matrix,
        item_prior_matrix=info.get("item_prior_matrix"),
        sequence_prior_matrix=info.get("sequence_prior_matrix"),
        knowledge_dim=_get(loaded_args, "knowledge_dim", 32),
        num_relation_heads=_get(loaded_args, "num_relation_heads", 4),
        num_gnn_layers=eff_gnn_layers,
        dropout=_get(loaded_args, "dropout", 0.3),
        use_concept_graph=use_concept_graph,
        graph_topk=_get(loaded_args, "graph_topk", None),
        allow_self_loop=not bool(_get(loaded_args, "disable_self_loop", False)),
        graph_identity_residual=_get(loaded_args, "graph_identity_residual", 0.0),
        use_personal_graph=_get(loaded_args, "use_personal_graph", False),
        personal_rank=_get(loaded_args, "personal_rank", 4),
        ablate_module1=ablate_module1,
        lambda_sparse_personal=_get(loaded_args, "lambda_sparse_personal", 0.0),
        lambda_alpha=_get(loaded_args, "lambda_alpha", 0.0),
        lambda_graph_entropy=_get(loaded_args, "lambda_sparse", 0.01),
        graph_entropy_min=_get(loaded_args, "graph_entropy_min", 0.15),
        graph_entropy_max=_get(loaded_args, "graph_entropy_max", 0.85),
        lambda_graph_diag=_get(loaded_args, "lambda_graph_diag", 0.10),
        lambda_graph_uniform=_get(loaded_args, "lambda_graph_uniform", 0.04),
        graph_uniform_margin=_get(loaded_args, "graph_uniform_margin", 0.10),
        graph_reg_warmup_epochs=_get(loaded_args, "graph_reg_warmup_epochs", 1),
        graph_reg_cap_ratio=_get(loaded_args, "graph_reg_cap_ratio", 6.0),
        graph_dropout=None
        if float(_get(loaded_args, "graph_dropout", -1.0)) < 0.0
        else float(_get(loaded_args, "graph_dropout", -1.0)),
        graph_tau_init=_get(loaded_args, "graph_tau_init", 1.0),
        graph_propagation_alpha=_get(loaded_args, "graph_propagation_alpha", 0.20),
        graph_query_readout_scale=_get(loaded_args, "graph_query_readout_scale", 0.35),
        graph_query_readout_2hop_scale=_get(loaded_args, "graph_query_readout_2hop_scale", 0.15),
        prediction_l2_lambda=_get(loaded_args, "prediction_l2_lambda", 5e-5),
        gnn_residual_weight=_get(loaded_args, "gnn_residual_weight", 0.5),
        personal_max_alpha=_get(loaded_args, "personal_max_alpha", 0.35),
        personal_delta_scale=_get(loaded_args, "personal_delta_scale", 1.0),
        personal_warmup_epochs=_get(loaded_args, "personal_warmup_epochs", 0),
        personal_reg_warmup_epochs=_get(loaded_args, "personal_reg_warmup_epochs", None),
        personal_student_dim=_get(loaded_args, "personal_student_dim", _get(loaded_args, "knowledge_dim", 32)),
        lambda_alpha_min=_get(loaded_args, "lambda_alpha_min", 0.0),
        alpha_min_target=_get(loaded_args, "alpha_min_target", 0.0),
        personal_alpha_temperature=_get(loaded_args, "personal_alpha_temperature", 2.0),
        personal_alpha_budget=_get(loaded_args, "personal_alpha_budget", 0.10),
        personal_alpha_base_init=_get(loaded_args, "personal_alpha_base_init", 0.08),
        personal_alpha_bias_scale=_get(loaded_args, "personal_alpha_bias_scale", 1.0),
        personal_freeze_alpha_gate=_get(loaded_args, "personal_freeze_alpha_gate", False),
        personal_disable_student_global_context=_get(loaded_args, "personal_disable_student_global_context", False),
        personal_local_hops=_get(loaded_args, "personal_local_hops", 1),
        personal_include_neighbor_rows=_get(loaded_args, "personal_include_neighbor_rows", False),
        personal_query_row_budget=_get(loaded_args, "personal_query_row_budget", 1.0),
        personal_neighbor_row_budget=_get(loaded_args, "personal_neighbor_row_budget", 0.30),
        personal_query_support_hops=_get(loaded_args, "personal_query_support_hops", 0),
        personal_support_only=_get(loaded_args, "personal_support_only", True),
        personal_query_correction_scale=_get(loaded_args, "personal_query_correction_scale", 0.15),
        personal_query_correction_max_ratio=_get(loaded_args, "personal_query_correction_max_ratio", 0.20),
        personal_query_correction_min_graph_anchor=_get(
            loaded_args, "personal_query_correction_min_graph_anchor", 0.01
        ),
        personal_query_message_gain=_get(loaded_args, "personal_query_message_gain", 1.0),
        lambda_personal_kl=_get(loaded_args, "lambda_personal_kl", 0.0),
        lambda_personal_query_residual=_get(loaded_args, "lambda_personal_query_residual", 0.0),
        personal_query_residual_margin=_get(loaded_args, "personal_query_residual_margin", 0.0),
        personal_support_include_query_self=_get(loaded_args, "personal_support_include_query_self", True),
        personal_support_include_graph=_get(loaded_args, "personal_support_include_graph", True),
        personal_support_include_neighbors=_get(loaded_args, "personal_support_include_neighbors", False),
        personal_item_support_mass=_get(loaded_args, "personal_item_support_mass", 0.0),
        personal_mastery_prior_scale=_get(loaded_args, "personal_mastery_prior_scale", 0.0),
        personal_recent_mastery_prior_scale=_get(loaded_args, "personal_recent_mastery_prior_scale", 0.0),
        personal_value_use_global_basis=_get(loaded_args, "personal_value_use_global_basis", True),
        personal_message_alignment_gate=_get(loaded_args, "personal_message_alignment_gate", True),
        personal_projection_hidden_factor=_get(loaded_args, "personal_projection_hidden_factor", 2),
        graph_headwise_query_gate=_get(loaded_args, "graph_headwise_query_gate", True),
        graph_edge_bias_rank=_get(loaded_args, "graph_edge_bias_rank", 8),
        graph_query_adapter_enable=_get(loaded_args, "graph_query_adapter_enable", True),
        graph_prior_logit_scale=_get(loaded_args, "graph_prior_logit_scale", 0.0),
        ae_query_residual_scale=_get(loaded_args, "ae_query_residual_scale", 0.0),
        ae_logit_residual_scale=_get(loaded_args, "ae_logit_residual_scale", 0.0),
        ae_logit_residual_clip=_get(loaded_args, "ae_logit_residual_clip", 1.0),
        ae_irt_logit_scale=_get(loaded_args, "ae_irt_logit_scale", 1.0),
        ae_interaction_logit_scale=_get(loaded_args, "ae_interaction_logit_scale", 0.0),
        ae_logit_dim=_get(loaded_args, "ae_logit_dim", 32),
        ae_posterior_prior_scale=_get(loaded_args, "ae_posterior_prior_scale", 0.0),
        ae_posterior_theta_scale=_get(loaded_args, "ae_posterior_theta_scale", 0.0),
        relation_theta_scale=_get(loaded_args, "relation_theta_scale", 0.0),
        relation_theta_delta_clip=_get(loaded_args, "relation_theta_delta_clip", 2.0),
        share_concept_embeddings=_get(loaded_args, "share_concept_embeddings", False),
    ).to(device)
    incompatible = model.load_state_dict(_strip_module_prefix(checkpoint["model_state_dict"]), strict=False)
    if getattr(incompatible, "unexpected_keys", None):
        print(f"Unexpected keys: {list(incompatible.unexpected_keys)[:20]}")
    if getattr(incompatible, "missing_keys", None):
        print(f"Missing keys: {list(incompatible.missing_keys)[:20]}")
    model.set_epoch(int(checkpoint.get("epoch", _get(loaded_args, "epochs", 1))))
    model.eval()
    return model


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.save_dir) / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    loaded_args: Dict[str, Any] = dict(checkpoint.get("args", {}))
    info = checkpoint["info_dict"]
    data_dir = _resolve_data_dir(_get(loaded_args, "dataset_name", args.dataset_name), args.data_dir)
    split_name = str(args.split).strip().lower()
    if split_name not in {"valid", "test"}:
        raise ValueError(f"--split must be valid or test, got {args.split!r}")
    split_file = "valid.csv" if split_name == "valid" else "test.csv"
    raw_eval_df = pd.read_csv(os.path.join(data_dir, split_file))
    filtered_eval_df = raw_eval_df[
        raw_eval_df["stu_id"].isin(set(info["stu_id_map"].keys()))
        & raw_eval_df["exer_id"].isin(set(info["exer_id_map"].keys()))
    ].reset_index(drop=True)
    dataset = CognitiveDiagnosisDataset(
        filtered_eval_df,
        info["stu_id_map"],
        info["exer_id_map"],
        info["cpt_id_map"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = _build_model(checkpoint, device)

    q_matrix = info["q_matrix"].detach().cpu()
    rows: List[pd.DataFrame] = []
    offset = 0
    with torch.no_grad():
        for batch_idx, (student_ids, exercise_ids, labels) in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            take = int(labels.numel())
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            labels = labels.to(device).float().reshape(-1)
            logits, details = model(student_ids, exercise_ids, return_details=True, return_logits=True)
            logits = logits.reshape(-1)
            probs = torch.sigmoid(logits)
            frame = filtered_eval_df.iloc[offset : offset + take][["stu_id", "exer_id", "label"]].copy()
            frame["mapped_student_id"] = student_ids.detach().cpu().numpy()
            frame["mapped_exercise_id"] = exercise_ids.detach().cpu().numpy()
            frame["prob"] = probs.detach().cpu().numpy()
            frame["pred"] = (frame["prob"] > 0.5).astype(np.float32)
            frame["abs_error"] = np.abs(frame["prob"].to_numpy() - frame["label"].to_numpy())
            frame["wrong"] = (frame["pred"].to_numpy() != frame["label"].to_numpy()).astype(np.float32)
            q_counts = q_matrix[exercise_ids.detach().cpu()].sum(dim=1).numpy()
            frame["q_count"] = q_counts.astype(np.int32)
            frame["is_multi_concept"] = (frame["q_count"] >= 2).astype(np.int32)
            for key in ("irt_logit", "irt_logit_for_total", "relation_theta_logit", "ae_logit_residual", "logits"):
                frame[key] = _tensor_to_numpy(details.get(key), take)
            for key in (
                "query_row_posterior_delta_abs",
                "query_row_posterior_kl",
                "query_row_personal_message_delta",
                "query_row_global_readout_delta",
                "personal_query_trust_scale_mean",
                "query_row_message_alignment",
            ):
                frame[key] = _tensor_to_numpy(details.get(key), take)
            for key in AE_COMPONENT_KEYS:
                frame[key] = _tensor_to_numpy(details.get(key), take)
            no_ae_logit = frame["irt_logit_for_total"].to_numpy() + frame["relation_theta_logit"].to_numpy()
            label_np = frame["label"].to_numpy(dtype=np.float64)
            frame["bce_total"] = _bce_from_logits(frame["logits"].to_numpy(), label_np)
            frame["bce_no_ae"] = _bce_from_logits(no_ae_logit, label_np)
            frame["ae_bce_delta"] = frame["bce_total"] - frame["bce_no_ae"]
            frame["ae_helped"] = (frame["ae_bce_delta"] < 0).astype(np.float32)
            frame["ae_direction_correct"] = (
                ((frame["label"] > 0.5) & (frame["ae_logit_residual"] > 0.0))
                | ((frame["label"] <= 0.5) & (frame["ae_logit_residual"] < 0.0))
            ).astype(np.float32)
            rows.append(frame)
            offset += take

    samples = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    labels = samples["label"].to_numpy(dtype=np.float32)
    probs = samples["prob"].to_numpy(dtype=np.float32)
    preds = samples["pred"].to_numpy(dtype=np.float32)
    metrics = compute_metrics(labels, preds, probs) if len(samples) else {"auc": None, "acc": None, "rmse": None}
    component_means = {
        key: {
            "mean": float(samples[key].mean()),
            "abs_mean": float(samples[key].abs().mean()),
            "wrong_abs_mean": float(samples.loc[samples["wrong"] > 0, key].abs().mean()),
            "correct_abs_mean": float(samples.loc[samples["wrong"] <= 0, key].abs().mean()),
        }
        for key in AE_COMPONENT_KEYS
        if key in samples
    }
    summary: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "dataset_name": _get(loaded_args, "dataset_name", args.dataset_name),
        "split": split_name,
        "num_samples": int(len(samples)),
        "metrics": metrics,
        "coverage": {
            "raw_rows": int(len(raw_eval_df)),
            "seen_rows": int(len(filtered_eval_df)),
            "seen_ratio": float(len(filtered_eval_df) / max(1, len(raw_eval_df))),
        },
        "overall": {
            "wrong_rate": float(samples["wrong"].mean()),
            "abs_error_mean": float(samples["abs_error"].mean()),
            "bce_total_mean": float(samples["bce_total"].mean()),
            "ae_bce_delta_mean": float(samples["ae_bce_delta"].mean()),
            "ae_help_rate": float(samples["ae_helped"].mean()),
            "ae_direction_correct_rate": float(samples["ae_direction_correct"].mean()),
            "ae_abs_mean": float(samples["ae_logit_residual"].abs().mean()),
            "roadmap_macro_abs_mean": float(samples["roadmap_macro_logit"].abs().mean()),
            "tutor_local_abs_mean": float(samples["tutor_local_navigation_logit"].abs().mean()),
        },
        "by_q_count": _safe_group_stats(samples, ["q_count"]),
        "by_multi_concept": _safe_group_stats(samples, ["is_multi_concept"]),
        "by_label": _safe_group_stats(samples, ["label"]),
        "component_means": component_means,
        "top_wrong": samples.sort_values("abs_error", ascending=False)
        .head(args.top_k)
        .to_dict(orient="records"),
    }
    output_dir = Path(args.output_dir or (Path(args.save_dir) / "error_analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_dir / "ae_error_samples.csv", index=False)
    with open(output_dir / "ae_error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AE test prediction errors.")
    parser.add_argument("--save_dir", required=True, help="Directory containing best_model.pth")
    parser.add_argument("--dataset_name", default="assist_09")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    print("Wrote AE error diagnostics.")


if __name__ == "__main__":
    main()
