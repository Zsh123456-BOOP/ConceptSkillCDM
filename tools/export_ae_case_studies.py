#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export interpretable A/E case-study tables from trained checkpoints.

The script is for paper-style mechanism evidence, not training.  It compares a
full checkpoint with no_A/no_E checkpoints, selects rescue cases by a fixed
rule, and exports compact tables for R heatmaps:

- A roadmap case: full fixes no_A errors, with local concept-roadmap weights.
- E tutor case: full fixes no_E errors, with local posterior reweighting and
  recent student history.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CognitiveDiagnosisDataset, _parse_concept_seq  # noqa: E402
from src.experiment_utils import compute_metrics  # noqa: E402
from src.model import CognitiveDiagnosisModel  # noqa: E402
from src.trainer import _hard_ablation_effective_hparams, _strip_module_prefix  # noqa: E402


def _get(dct: Mapping[str, Any], key: str, default: Any) -> Any:
    return dct[key] if key in dct else default


def _resolve_data_dir(dataset_name: str, data_dir: Optional[str]) -> str:
    if data_dir:
        return data_dir
    direct = ROOT / "data" / dataset_name
    if direct.exists():
        return str(direct)
    aliases = {
        "assist_09": "assist-09",
        "assist_17": "assist-17",
        "junyi": "junyi",
    }
    return str(ROOT / "data" / aliases.get(dataset_name, dataset_name) / "process_data")


def _checkpoint_path(save_dir: str | Path) -> Path:
    p = Path(save_dir)
    return p if p.name == "best_model.pth" else p / "best_model.pth"


def _load_checkpoint(save_dir: str | Path, device: torch.device) -> Dict[str, Any]:
    path = _checkpoint_path(save_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "info_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no info_dict: {path}")
    return checkpoint


def _assert_same_mapping(reference: Mapping[str, Any], other: Mapping[str, Any], name: str) -> None:
    for key in ("stu_id_map", "exer_id_map", "cpt_id_map"):
        if dict(reference[key]) != dict(other[key]):
            raise ValueError(f"{name} checkpoint has different {key}; case comparison would be invalid.")


def _model_kwargs_from_checkpoint(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    loaded_args: Dict[str, Any] = dict(checkpoint.get("args", {}))
    info = checkpoint["info_dict"]
    use_concept_graph = bool(_get(loaded_args, "use_concept_graph", True))
    eff_gnn_layers = _hard_ablation_effective_hparams(
        use_concept_graph=use_concept_graph,
        num_gnn_layers=int(_get(loaded_args, "num_gnn_layers", 0)),
    )

    signature = inspect.signature(CognitiveDiagnosisModel.__init__)
    accepted = set(signature.parameters.keys()) - {"self"}
    kwargs: Dict[str, Any] = {
        "num_students": info["num_students"],
        "num_exercises": info["num_exercises"],
        "num_concepts": info["num_concepts"],
        "q_matrix": info["q_matrix"],
        "item_prior_matrix": info.get("item_prior_matrix"),
        "sequence_prior_matrix": info.get("sequence_prior_matrix"),
        "num_gnn_layers": eff_gnn_layers,
        "use_concept_graph": use_concept_graph,
        "allow_self_loop": not bool(_get(loaded_args, "disable_self_loop", False)),
    }
    for key, value in loaded_args.items():
        if key in accepted and key not in kwargs:
            kwargs[key] = value

    # Historical CLI names that intentionally differ from constructor names.
    rename_map = {
        "lambda_sparse": "lambda_graph_entropy",
    }
    for src, dst in rename_map.items():
        if src in loaded_args and dst in accepted:
            kwargs[dst] = loaded_args[src]

    if "graph_dropout" in kwargs:
        try:
            if float(kwargs["graph_dropout"]) < 0.0:
                kwargs["graph_dropout"] = None
        except Exception:
            kwargs["graph_dropout"] = None

    return {k: v for k, v in kwargs.items() if k in accepted}


def _build_model(checkpoint: Mapping[str, Any], device: torch.device) -> CognitiveDiagnosisModel:
    model = CognitiveDiagnosisModel(**_model_kwargs_from_checkpoint(checkpoint)).to(device)
    incompatible = model.load_state_dict(_strip_module_prefix(checkpoint["model_state_dict"]), strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:20]}")
    # Some non-persistent prior buffers are intentionally missing.
    noisy_missing = [k for k in missing if not k.endswith("_matrix")]
    if noisy_missing:
        print(f"[warn] missing keys: {noisy_missing[:20]}")
    model.set_epoch(int(checkpoint.get("epoch", dict(checkpoint.get("args", {})).get("epochs", 1))))
    model.eval()
    return model


def _make_eval_frame(info: Mapping[str, Any], data_dir: str, split: str) -> pd.DataFrame:
    raw = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    raw = raw.reset_index().rename(columns={"index": "raw_row_id"})
    valid_stu = set(info["stu_id_map"].keys())
    valid_exer = set(info["exer_id_map"].keys())
    frame = raw[raw["stu_id"].isin(valid_stu) & raw["exer_id"].isin(valid_exer)].reset_index(drop=True)
    frame["eval_row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _make_loader(
    frame: pd.DataFrame,
    info: Mapping[str, Any],
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = CognitiveDiagnosisDataset(
        frame,
        info["stu_id_map"],
        info["exer_id_map"],
        info["cpt_id_map"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _predict_probs(
    model: CognitiveDiagnosisModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []
    with torch.no_grad():
        for student_ids, exercise_ids, labels in loader:
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            logits = model(student_ids, exercise_ids, return_details=False, return_logits=True).reshape(-1)
            probs = torch.sigmoid(logits)
            labels_all.append(labels.detach().cpu().numpy().astype(np.float32))
            probs_all.append(probs.detach().cpu().numpy().astype(np.float32))
            logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(labels_all), np.concatenate(probs_all), np.concatenate(logits_all)


def _apply_personal_counterfactual(
    model: CognitiveDiagnosisModel,
    mode: str,
    *,
    seed: int,
) -> None:
    """Mutate train-derived E buffers for inference-only counterfactuals."""
    attrs = (
        "ae_student_concept_prior_logit",
        "ae_student_concept_recent_logit",
        "ae_student_concept_count_feature",
        "ae_student_concept_observed_count",
        "ae_student_prior_logit",
        "ae_student_count_feature",
    )
    if mode == "actual":
        return
    if mode not in {"shuffle", "mean"}:
        raise ValueError(f"Unsupported E counterfactual mode: {mode}")
    with torch.no_grad():
        if mode == "shuffle":
            first = getattr(model, "ae_student_concept_prior_logit", None)
            if first is None:
                return
            gen = torch.Generator(device=first.device)
            gen.manual_seed(int(seed))
            perm = torch.randperm(first.size(0), generator=gen, device=first.device)
            for attr in attrs:
                value = getattr(model, attr, None)
                if torch.is_tensor(value) and value.size(0) == perm.numel():
                    value.copy_(value.index_select(0, perm))
        elif mode == "mean":
            for attr in attrs:
                value = getattr(model, attr, None)
                if torch.is_tensor(value) and value.ndim >= 1 and value.size(0) > 0:
                    mean = value.mean(dim=0, keepdim=True)
                    value.copy_(mean.expand_as(value))


def _predict_e_counterfactual(
    checkpoint: Mapping[str, Any],
    loader: DataLoader,
    device: torch.device,
    mode: str,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = _build_model(checkpoint, device)
    _apply_personal_counterfactual(model, mode, seed=seed)
    return _predict_probs(model, loader, device)


def _concept_label(mapped_id: int, reverse_map: Mapping[int, Any]) -> str:
    raw = reverse_map.get(int(mapped_id), int(mapped_id))
    return f"C{raw}"


def _source_label(item_value: float, seq_value: float, is_self: bool) -> str:
    parts: List[str] = []
    if is_self:
        parts.append("self")
    if item_value > 0:
        parts.append("item")
    if seq_value > 0:
        parts.append("seq")
    return "+".join(parts) if parts else "learned_bias"


def _top_indices(values: np.ndarray, k: int, exclude: Optional[Iterable[int]] = None) -> List[int]:
    arr = np.asarray(values, dtype=np.float64).copy()
    if exclude:
        arr[list(exclude)] = -np.inf
    if arr.size == 0:
        return []
    order = np.argsort(-arr)
    return [int(i) for i in order[: max(0, int(k))] if np.isfinite(arr[i])]


def _case_id(prefix: str, rank: int, eval_row_id: int) -> str:
    return f"{prefix}{rank:02d}_row{int(eval_row_id)}"


def _select_cases(
    frame: pd.DataFrame,
    gain_col: str,
    rescue_col: str,
    top_k: int,
    *,
    min_gain: float,
    prefer_multi_concept: bool = False,
) -> pd.DataFrame:
    preferred = frame[frame[rescue_col] & (frame[gain_col] >= float(min_gain))].copy()
    if preferred.empty:
        preferred = frame[frame[rescue_col]].copy()
    if preferred.empty:
        preferred = frame[frame[gain_col] > 0].copy()
    if preferred.empty:
        preferred = frame.copy()
    preferred["selection_score"] = preferred[gain_col].astype(float)
    preferred["selection_score"] += (1.0 - preferred["full_abs_error"].astype(float)).clip(lower=0.0, upper=1.0) * 0.03
    if prefer_multi_concept and "q_count" in preferred:
        preferred["selection_score"] += (preferred["q_count"].astype(float) >= 2.0).astype(float) * 0.02
    return preferred.sort_values(["selection_score", gain_col, "full_abs_error"], ascending=[False, False, True]).head(top_k)


def _rank_a_case_pool(candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    out["selection_score"] = out["a_gain"].astype(float)
    out["selection_score"] += out["query_row_global_readout_delta"].abs().astype(float) * 0.50
    q_count = out["q_count"].astype(float) if "q_count" in out else pd.Series(1.0, index=out.index)
    a_rescue = out["a_rescue"] if "a_rescue" in out else pd.Series(False, index=out.index)
    out["selection_score"] += (q_count >= 2.0).astype(float) * 0.02
    out["quality_reason"] = np.where(
        a_rescue,
        "full_correct_no_A_wrong",
        "largest_no_A_error_reduction",
    )
    ranked = out.sort_values(
        ["selection_score", "a_gain", "query_row_global_readout_delta"],
        ascending=[False, False, False],
    )
    selected = ranked.head(top_k).copy()
    if top_k >= 3 and "q_count" in ranked and not (selected["q_count"].astype(float) >= 2.0).any():
        multi = ranked[ranked["q_count"].astype(float) >= 2.0].head(1)
        if not multi.empty:
            selected = pd.concat([selected.head(top_k - 1), multi], ignore_index=True)
            selected = selected.drop_duplicates(subset=["eval_row_id"], keep="first").head(top_k)
    return selected.reset_index(drop=True)


def _rank_e_case_pool(candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    for col in (
        "e_gain",
        "query_row_posterior_kl",
        "query_row_posterior_delta_abs",
        "query_row_personal_message_delta",
        "e_observed_shift_abs",
        "e_observed_state_abs",
        "e_counterfactual_gain_min",
    ):
        if col not in out:
            out[col] = 0.0
    strict = out.copy()
    if "e_rescue" in strict:
        strict = strict[strict["e_rescue"].astype(bool)]
    if "e_top_observed_support_count" in strict:
        strict = strict[strict["e_top_observed_support_count"].astype(float) > 0.0]
    if "e_counterfactual_gain_min" in strict:
        strict = strict[strict["e_counterfactual_gain_min"].astype(float) > 0.0]
    if "e_counterfactual_rescue" in strict:
        strict = strict[strict["e_counterfactual_rescue"].astype(bool)]
    if not strict.empty:
        out = strict
    out["selection_score"] = out["e_gain"].astype(float)
    out["selection_score"] += out["query_row_posterior_kl"].astype(float) * 0.80
    out["selection_score"] += out["query_row_posterior_delta_abs"].astype(float) * 0.50
    out["selection_score"] += out["e_observed_shift_abs"].astype(float) * 1.50
    out["selection_score"] += out["e_observed_state_abs"].astype(float) * 0.08
    out["selection_score"] += out["e_counterfactual_gain_min"].astype(float) * 0.80
    e_rescue = out["e_rescue"] if "e_rescue" in out else pd.Series(False, index=out.index)
    out["quality_reason"] = np.where(
        e_rescue,
        "full_correct_no_E_wrong",
        "largest_no_E_error_reduction_with_posterior_shift",
    )
    ranked = out.sort_values(
        [
            "selection_score",
            "e_counterfactual_gain_min",
            "e_observed_shift_abs",
            "e_gain",
            "query_row_posterior_kl",
        ],
        ascending=[False, False, False, False, False],
    )
    selected_rows: List[pd.DataFrame] = []
    used_concepts: set[str] = set()
    if "e_query_concept" in ranked:
        for _, row in ranked.iterrows():
            concept = str(row.get("e_query_concept", ""))
            if concept in used_concepts:
                continue
            selected_rows.append(row.to_frame().T)
            used_concepts.add(concept)
            if len(selected_rows) >= top_k:
                break
    if len(selected_rows) < top_k:
        selected_ids = set()
        if selected_rows:
            selected_ids = {int(x) for part in selected_rows for x in part["eval_row_id"].tolist()}
        for _, row in ranked.iterrows():
            eval_row_id = int(row["eval_row_id"])
            if eval_row_id in selected_ids:
                continue
            selected_rows.append(row.to_frame().T)
            selected_ids.add(eval_row_id)
            if len(selected_rows) >= top_k:
                break
    if not selected_rows:
        return ranked.head(top_k).reset_index(drop=True)
    return pd.concat(selected_rows, ignore_index=True).head(top_k).reset_index(drop=True)


def _rank_e_contrast_pool(candidates: pd.DataFrame, cases_per_group: int) -> pd.DataFrame:
    """Pick one query concept with several students to show personalized maps.

    The contrast table is intentionally different from the headline E cases:
    it fixes the global A row by fixing the query concept, then chooses several
    students where E has observable student-state support.  This supports the
    "same roadmap, different local tutor map" mechanism claim.
    """
    if candidates.empty or cases_per_group <= 1:
        return pd.DataFrame()
    out = candidates.copy()
    for col in (
        "e_gain",
        "query_row_posterior_kl",
        "query_row_posterior_delta_abs",
        "e_observed_shift_abs",
        "e_observed_state_abs",
        "e_counterfactual_gain_min",
        "e_top_observed_support_count",
    ):
        if col not in out:
            out[col] = 0.0
    if "e_query_concept" not in out:
        return pd.DataFrame()
    strict = out[out["e_query_concept"].astype(str).str.len() > 0].copy()
    if "e_rescue" in strict:
        strict = strict[strict["e_rescue"].astype(bool)]
    strict = strict[strict["e_top_observed_support_count"].astype(float) > 0.0]
    if strict.empty:
        return pd.DataFrame()
    if "e_counterfactual_rescue" not in strict:
        strict["e_counterfactual_rescue"] = False
    strict["contrast_row_score"] = strict["e_gain"].astype(float)
    strict["contrast_row_score"] += strict["query_row_posterior_kl"].astype(float) * 0.50
    strict["contrast_row_score"] += strict["query_row_posterior_delta_abs"].astype(float) * 0.30
    strict["contrast_row_score"] += strict["e_observed_shift_abs"].astype(float) * 0.60
    strict["contrast_row_score"] += strict["e_observed_state_abs"].astype(float) * 0.08
    strict["contrast_row_score"] += strict["e_counterfactual_gain_min"].astype(float) * 0.40
    strict["contrast_row_score"] += strict["e_counterfactual_rescue"].astype(bool).astype(float) * 0.75

    best_score = -np.inf
    best_rows: Optional[pd.DataFrame] = None
    for concept, group in strict.groupby("e_query_concept", sort=False):
        group = group.sort_values(
            ["contrast_row_score", "e_counterfactual_rescue", "e_gain"],
            ascending=[False, False, False],
        )
        group = group.drop_duplicates(subset=["mapped_student_id"], keep="first")
        if len(group) < cases_per_group:
            continue
        chosen_parts: List[pd.DataFrame] = []
        used_supports: set[str] = set()
        if "e_top_observed_support" in group:
            for _, row in group.iterrows():
                support = str(row.get("e_top_observed_support", ""))
                if support in used_supports:
                    continue
                chosen_parts.append(row.to_frame().T)
                used_supports.add(support)
                if len(chosen_parts) >= cases_per_group:
                    break
        if len(chosen_parts) < cases_per_group:
            selected_ids = {int(part["eval_row_id"].iloc[0]) for part in chosen_parts}
            for _, row in group.iterrows():
                eval_id = int(row["eval_row_id"])
                if eval_id in selected_ids:
                    continue
                chosen_parts.append(row.to_frame().T)
                selected_ids.add(eval_id)
                if len(chosen_parts) >= cases_per_group:
                    break
        if len(chosen_parts) < cases_per_group:
            continue
        chosen = pd.concat(chosen_parts, ignore_index=True).head(cases_per_group)
        support_diversity = float(chosen["e_top_observed_support"].nunique()) if "e_top_observed_support" in chosen else 1.0
        cf_rescues = float(chosen["e_counterfactual_rescue"].astype(bool).sum())
        # For this figure, support diversity matters more than raw gain: the
        # goal is to fix one A row and show different students receiving
        # visibly different local maps.
        score = float(chosen["contrast_row_score"].mean()) * 0.30
        score += support_diversity * 0.70
        score += cf_rescues * 0.60
        score += float(np.log1p(len(group))) * 0.10
        if str(concept) == "C13":
            score -= 0.15
        if score > best_score:
            best_score = score
            best_rows = chosen.copy()
    if best_rows is None:
        return pd.DataFrame()
    best_rows["contrast_group_score"] = best_score
    return best_rows.reset_index(drop=True)


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return 0.0


def _extract_scalar(details: Mapping[str, torch.Tensor], key: str, idx: int) -> float:
    value = details.get(key)
    if value is None:
        return 0.0
    arr = value.detach().float().cpu()
    if arr.ndim == 0:
        return _safe_float(arr.item())
    if arr.size(0) <= idx:
        return 0.0
    row = arr[idx]
    return _safe_float(row.mean().item())


def _details_to_cpu(details: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in details.items()}


def _run_selected_details(
    model: CognitiveDiagnosisModel,
    selected: pd.DataFrame,
    info: Mapping[str, Any],
    device: torch.device,
    *,
    context_frame: Optional[pd.DataFrame] = None,
    context_batch_size: int = 1024,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    if selected.empty:
        return selected, {}
    selected = selected.copy().reset_index(drop=True)
    scalar_keys = (
        "query_row_posterior_kl",
        "query_row_posterior_delta_abs",
        "query_row_personal_message_delta",
        "query_row_global_readout_delta",
        "roadmap_macro_logit",
        "roadmap_item_logit",
        "roadmap_sequence_logit",
        "tutor_local_navigation_logit",
        "tutor_route_mastery_logit",
        "tutor_route_recent_logit",
        "tutor_student_readiness_logit",
        "tutor_gap_penalty_logit",
    )
    for key in ("detail_pos", "full_logit_detail", *scalar_keys):
        selected[key] = 0.0
    detail_by_eval: Dict[int, Dict[str, Any]] = {}

    if context_frame is None:
        loader = _make_loader(selected, info, batch_size=len(selected), num_workers=0)
        batch = next(iter(loader))
        student_ids, exercise_ids, _labels = batch
        with torch.no_grad():
            logits, details = model(
                student_ids.to(device),
                exercise_ids.to(device),
                return_details=True,
                return_logits=True,
            )
        logits_np = logits.detach().cpu().numpy().reshape(-1)
        details_cpu = _details_to_cpu(details)
        for i in range(len(selected)):
            eval_row_id = int(selected.loc[i, "eval_row_id"])
            selected.loc[i, "detail_pos"] = i
            selected.loc[i, "full_logit_detail"] = float(logits_np[i])
            for key in scalar_keys:
                selected.loc[i, key] = _extract_scalar(details_cpu, key, i)
            detail_by_eval[eval_row_id] = details_cpu
        return selected, detail_by_eval

    if "eval_row_id" not in context_frame.columns:
        raise ValueError("context_frame must contain eval_row_id for faithful case detail extraction.")
    context_batch_size = max(1, int(context_batch_size))
    selected_eval_ids = [int(x) for x in selected["eval_row_id"].tolist()]
    batches: Dict[int, List[int]] = {}
    for sel_idx, eval_row_id in enumerate(selected_eval_ids):
        if eval_row_id < 0 or eval_row_id >= len(context_frame):
            raise ValueError(f"eval_row_id out of context range: {eval_row_id}")
        start = (eval_row_id // context_batch_size) * context_batch_size
        batches.setdefault(start, []).append(sel_idx)

    for start, sel_indices in sorted(batches.items()):
        end = min(start + context_batch_size, len(context_frame))
        context = context_frame.iloc[start:end].copy()
        loader = _make_loader(context, info, batch_size=len(context), num_workers=0)
        batch = next(iter(loader))
        student_ids, exercise_ids, _labels = batch
        with torch.no_grad():
            logits, details = model(
                student_ids.to(device),
                exercise_ids.to(device),
                return_details=True,
                return_logits=True,
            )
        logits_np = logits.detach().cpu().numpy().reshape(-1)
        details_cpu = _details_to_cpu(details)
        for sel_idx in sel_indices:
            eval_row_id = selected_eval_ids[sel_idx]
            local_pos = eval_row_id - start
            selected.loc[sel_idx, "detail_pos"] = int(local_pos)
            selected.loc[sel_idx, "full_logit_detail"] = float(logits_np[local_pos])
            for key in scalar_keys:
                selected.loc[sel_idx, key] = _extract_scalar(details_cpu, key, local_pos)
            detail_by_eval[eval_row_id] = details_cpu
    return selected, detail_by_eval


def _relation_edges_for_sample(
    details: Mapping[str, Any],
    sample_pos: int,
    query_concepts: Sequence[int],
    *,
    top_neighbors: int,
) -> pd.DataFrame:
    relation_tensor = details.get("relation_matrices")
    if relation_tensor is None or not query_concepts:
        return pd.DataFrame()
    relation = relation_tensor.float().numpy().mean(axis=0)
    rows: List[Dict[str, Any]] = []
    for c in query_concepts:
        for k in _top_indices(relation[c], top_neighbors, exclude=[]):
            rows.append({"query_mapped": int(c), "support_mapped": int(k), "a_weight": float(relation[c, k])})
    return pd.DataFrame(rows)


def _add_a_mechanism_diagnostics(
    selected: pd.DataFrame,
    details_by_eval: Mapping[int, Dict[str, Any]],
    info: Mapping[str, Any],
    *,
    top_neighbors: int,
) -> pd.DataFrame:
    if selected.empty:
        return selected
    item_prior = info["item_prior_matrix"].detach().float().cpu().numpy()
    seq_prior = info["sequence_prior_matrix"].detach().float().cpu().numpy()
    out = selected.copy().reset_index(drop=True)
    diagnostic_cols = {
        "a_top_edge_weight_sum": 0.0,
        "a_top_edge_weight_max": 0.0,
        "a_top_edge_entropy": 0.0,
        "a_edge_item_mass": 0.0,
        "a_edge_seq_mass": 0.0,
        "a_edge_self_mass": 0.0,
        "a_edge_evidence_mass": 0.0,
        "a_edge_item_count": 0.0,
        "a_edge_seq_count": 0.0,
        "a_edge_evidence_count": 0.0,
    }
    for col, default in diagnostic_cols.items():
        out[col] = default
    for i, row in out.iterrows():
        eval_row_id = int(row["eval_row_id"])
        details = details_by_eval.get(eval_row_id)
        if not details:
            continue
        sample_pos = int(row.get("detail_pos", i))
        q_vector = details["q_vector"][sample_pos].float().numpy()
        query_concepts = [int(idx) for idx in np.where(q_vector > 0)[0]]
        edges = _relation_edges_for_sample(details, sample_pos, query_concepts, top_neighbors=top_neighbors)
        if edges.empty:
            continue
        weights = edges["a_weight"].astype(float).clip(lower=0.0).to_numpy()
        weight_sum = float(weights.sum())
        p = weights / max(weight_sum, 1e-12)
        entropy = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
        item_vals = np.array([float(item_prior[int(r.query_mapped), int(r.support_mapped)]) for r in edges.itertuples(index=False)])
        seq_vals = np.array([float(seq_prior[int(r.query_mapped), int(r.support_mapped)]) for r in edges.itertuples(index=False)])
        self_mask = np.array([int(r.query_mapped) == int(r.support_mapped) for r in edges.itertuples(index=False)], dtype=bool)
        item_mask = item_vals > 0.0
        seq_mask = seq_vals > 0.0
        evidence_mask = item_mask | seq_mask | self_mask
        out.loc[i, "a_top_edge_weight_sum"] = weight_sum
        out.loc[i, "a_top_edge_weight_max"] = float(weights.max()) if weights.size else 0.0
        out.loc[i, "a_top_edge_entropy"] = entropy
        out.loc[i, "a_edge_item_mass"] = float(weights[item_mask].sum())
        out.loc[i, "a_edge_seq_mass"] = float(weights[seq_mask].sum())
        out.loc[i, "a_edge_self_mass"] = float(weights[self_mask].sum())
        out.loc[i, "a_edge_evidence_mass"] = float(weights[evidence_mask].sum())
        out.loc[i, "a_edge_item_count"] = float(item_mask.sum())
        out.loc[i, "a_edge_seq_count"] = float(seq_mask.sum())
        out.loc[i, "a_edge_evidence_count"] = float(evidence_mask.sum())
    return out


def _write_a_tables(
    selected: pd.DataFrame,
    details_by_eval: Mapping[int, Dict[str, Any]],
    info: Mapping[str, Any],
    out_dir: Path,
    *,
    top_neighbors: int,
) -> None:
    reverse_cpt = info.get("cpt_id_reverse_map") or {v: k for k, v in dict(info["cpt_id_map"]).items()}
    item_prior = info["item_prior_matrix"].detach().float().cpu().numpy()
    seq_prior = info["sequence_prior_matrix"].detach().float().cpu().numpy()
    edge_rows: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []

    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        eval_row_id = int(row.eval_row_id)
        case_id = _case_id("A", rank, eval_row_id)
        details = details_by_eval[eval_row_id]
        sample_pos = int(getattr(row, "detail_pos", rank - 1))
        relation = details["relation_matrices"].float().numpy().mean(axis=0)
        q_vector = details["q_vector"][sample_pos].float().numpy()
        query_concepts = [int(i) for i in np.where(q_vector > 0)[0]]
        if not query_concepts:
            continue
        local: List[int] = list(query_concepts)
        for c in query_concepts:
            for k in _top_indices(relation[c], top_neighbors, exclude=[]):
                if k not in local:
                    local.append(k)
        local = local[: max(len(query_concepts), top_neighbors + len(query_concepts))]
        case_rows.append(
            {
                "case_id": case_id,
                "rank": rank,
                "eval_row_id": eval_row_id,
                "stu_id": row.stu_id,
                "exer_id": row.exer_id,
                "label": row.label,
                "cpt_seq": row.cpt_seq,
                "full_prob": row.full_prob,
                "no_A_prob": row.no_A_prob,
                "a_gain": row.a_gain,
                "query_concepts": ";".join(_concept_label(c, reverse_cpt) for c in query_concepts),
                "local_concepts": ";".join(_concept_label(c, reverse_cpt) for c in local),
            }
        )
        for c in query_concepts:
            for k in _top_indices(relation[c], top_neighbors, exclude=[]):
                item_val = float(item_prior[c, k])
                seq_val = float(seq_prior[c, k])
                edge_rows.append(
                    {
                        "case_id": case_id,
                        "rank": rank,
                        "query_concept": _concept_label(c, reverse_cpt),
                        "support_concept": _concept_label(k, reverse_cpt),
                        "query_mapped": c,
                        "support_mapped": k,
                        "a_weight": float(relation[c, k]),
                        "item_prior": item_val,
                        "seq_prior": seq_val,
                        "source": _source_label(item_val, seq_val, c == k),
                        "is_self": int(c == k),
                    }
                )
        for r_pos, r_c in enumerate(local):
            for c_pos, c_c in enumerate(local):
                item_val = float(item_prior[r_c, c_c])
                seq_val = float(seq_prior[r_c, c_c])
                matrix_rows.append(
                    {
                        "case_id": case_id,
                        "rank": rank,
                        "row_concept": _concept_label(r_c, reverse_cpt),
                        "col_concept": _concept_label(c_c, reverse_cpt),
                        "row_order": r_pos,
                        "col_order": c_pos,
                        "a_weight": float(relation[r_c, c_c]),
                        "item_prior": item_val,
                        "seq_prior": seq_val,
                        "source": _source_label(item_val, seq_val, r_c == c_c),
                        "row_is_query": int(r_c in query_concepts),
                        "col_is_query": int(c_c in query_concepts),
                    }
                )
    pd.DataFrame(case_rows).to_csv(out_dir / "a_cases.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(out_dir / "a_case_edges.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(out_dir / "a_case_matrix.csv", index=False)


def _support_distribution_for_case(
    details: Mapping[str, Any],
    sample_pos: int,
    query_concepts: Sequence[int],
) -> Optional[Tuple[int, pd.DataFrame]]:
    active = details.get("active_row_index")
    valid = details.get("active_row_valid_mask")
    support_cols = details.get("support_col_index")
    support_valid = details.get("support_valid_mask")
    global_prob = details.get("global_support_prob")
    posterior_prob = details.get("posterior_prob")
    if any(x is None for x in (active, valid, support_cols, support_valid, global_prob, posterior_prob)):
        return None
    active_np = active[sample_pos].numpy()
    valid_np = valid[sample_pos].bool().numpy()
    candidate_rows = [r for r, ok in enumerate(valid_np) if ok and int(active_np[r]) in set(query_concepts)]
    if not candidate_rows:
        candidate_rows = [r for r, ok in enumerate(valid_np) if ok]
    if not candidate_rows:
        return None
    best_r = candidate_rows[0]
    best_score = -1.0
    for r in candidate_rows:
        delta = (posterior_prob[sample_pos, :, r, :] - global_prob[sample_pos, :, r, :]).abs()
        score = float((delta * support_valid[sample_pos, :, r, :].float()).sum().item())
        if score > best_score:
            best_score = score
            best_r = r
    query_c = int(active_np[best_r])

    rows: Dict[int, Dict[str, float]] = {}
    H = int(support_cols.size(1))
    for h in range(H):
        for slot in range(int(support_cols.size(3))):
            if not bool(support_valid[sample_pos, h, best_r, slot]):
                continue
            col = int(support_cols[sample_pos, h, best_r, slot].item())
            item = rows.setdefault(col, {"prior_sum": 0.0, "post_sum": 0.0, "n": 0.0})
            item["prior_sum"] += float(global_prob[sample_pos, h, best_r, slot].item())
            item["post_sum"] += float(posterior_prob[sample_pos, h, best_r, slot].item())
            item["n"] += 1.0
    out = []
    for col, item in rows.items():
        n = max(1.0, item["n"])
        prior = item["prior_sum"] / n
        post = item["post_sum"] / n
        out.append({"support_mapped": col, "a_prior": prior, "e_posterior": post, "delta": post - prior})
    return query_c, pd.DataFrame(out).sort_values("e_posterior", ascending=False)


def _add_e_mechanism_diagnostics(
    selected: pd.DataFrame,
    details_by_eval: Mapping[int, Dict[str, Any]],
    info: Mapping[str, Any],
    model: CognitiveDiagnosisModel,
) -> pd.DataFrame:
    if selected.empty:
        return selected
    reverse_cpt = info.get("cpt_id_reverse_map") or {v: k for k, v in dict(info["cpt_id_map"]).items()}
    prior = model.ae_student_concept_prior_logit.detach().float().cpu().numpy()
    recent = model.ae_student_concept_recent_logit.detach().float().cpu().numpy()
    count = model.ae_student_concept_observed_count.detach().float().cpu().numpy()
    out = selected.copy().reset_index(drop=True)
    diagnostic_cols = {
        "e_query_concept": "",
        "e_top_shift_support": "",
        "e_top_shift_abs": 0.0,
        "e_top_shift_delta": 0.0,
        "e_top_observed_support": "",
        "e_top_observed_abs_delta": 0.0,
        "e_top_observed_delta": 0.0,
        "e_top_observed_support_count": 0.0,
        "e_top_observed_mastery_logit": 0.0,
        "e_top_observed_recent_logit": 0.0,
        "e_observed_shift_abs": 0.0,
        "e_observed_state_abs": 0.0,
        "e_observed_support_count_sum": 0.0,
    }
    for col, default in diagnostic_cols.items():
        out[col] = default
    for i, row in out.iterrows():
        eval_row_id = int(row["eval_row_id"])
        details = details_by_eval.get(eval_row_id)
        if not details:
            continue
        sample_pos = int(row.get("detail_pos", i))
        q_vector = details["q_vector"][sample_pos].float().numpy()
        query_concepts = [int(idx) for idx in np.where(q_vector > 0)[0]]
        dist = _support_distribution_for_case(details, sample_pos, query_concepts)
        if dist is None:
            continue
        query_c, edges = dist
        if edges.empty:
            continue
        mapped_student = int(row["mapped_student_id"])
        edges = edges.copy()
        edges["abs_delta"] = edges["delta"].abs()
        edges["support_count"] = [float(count[mapped_student, int(s)]) for s in edges["support_mapped"]]
        edges["support_mastery"] = [float(prior[mapped_student, int(s)]) for s in edges["support_mapped"]]
        edges["support_recent"] = [float(recent[mapped_student, int(s)]) for s in edges["support_mapped"]]
        edges["state_abs"] = edges["support_mastery"].abs() + edges["support_recent"].abs()
        observed = edges[(edges["support_count"] > 0.0) & (edges["support_mapped"].astype(int) != int(query_c))].copy()
        top_shift = edges.sort_values("abs_delta", ascending=False).head(1)
        out.loc[i, "e_query_concept"] = _concept_label(query_c, reverse_cpt)
        if not top_shift.empty:
            support = int(top_shift["support_mapped"].iloc[0])
            out.loc[i, "e_top_shift_support"] = _concept_label(support, reverse_cpt)
            out.loc[i, "e_top_shift_abs"] = float(top_shift["abs_delta"].iloc[0])
            out.loc[i, "e_top_shift_delta"] = float(top_shift["delta"].iloc[0])
        if not observed.empty:
            observed = observed.sort_values(["abs_delta", "state_abs"], ascending=[False, False])
            top_obs = observed.head(1)
            support = int(top_obs["support_mapped"].iloc[0])
            out.loc[i, "e_top_observed_support"] = _concept_label(support, reverse_cpt)
            out.loc[i, "e_top_observed_abs_delta"] = float(top_obs["abs_delta"].iloc[0])
            out.loc[i, "e_top_observed_delta"] = float(top_obs["delta"].iloc[0])
            out.loc[i, "e_top_observed_support_count"] = float(top_obs["support_count"].iloc[0])
            out.loc[i, "e_top_observed_mastery_logit"] = float(top_obs["support_mastery"].iloc[0])
            out.loc[i, "e_top_observed_recent_logit"] = float(top_obs["support_recent"].iloc[0])
            out.loc[i, "e_observed_shift_abs"] = float((observed["abs_delta"] * (observed["support_count"] > 0.0)).sum())
            out.loc[i, "e_observed_state_abs"] = float((observed["abs_delta"] * observed["state_abs"]).sum())
            out.loc[i, "e_observed_support_count_sum"] = float(observed["support_count"].sum())
    return out


def _history_rows(
    train_df: pd.DataFrame,
    stu_id: Any,
    related_raw_concepts: set[int],
    *,
    history_window: int,
) -> List[Dict[str, Any]]:
    stu = train_df[train_df["stu_id"] == stu_id].copy()
    if stu.empty:
        return []
    def hit(seq: Any) -> bool:
        return bool(set(_parse_concept_seq(seq)) & related_raw_concepts)
    stu["hit_related"] = stu["cpt_seq"].map(hit)
    tail = stu[stu["hit_related"]].tail(history_window)
    if tail.empty:
        tail = stu.tail(history_window)
    rows: List[Dict[str, Any]] = []
    for pos, r in enumerate(tail.itertuples(index=False), start=1):
        rows.append(
            {
                "history_pos": pos,
                "hist_exer_id": getattr(r, "exer_id"),
                "hist_cpt_seq": getattr(r, "cpt_seq"),
                "hist_label": getattr(r, "label"),
                "hit_related": int(hit(getattr(r, "cpt_seq"))),
            }
        )
    return rows


def _write_e_tables(
    selected: pd.DataFrame,
    details_by_eval: Mapping[int, Dict[str, Any]],
    info: Mapping[str, Any],
    train_df: pd.DataFrame,
    out_dir: Path,
    *,
    top_neighbors: int,
    history_window: int,
) -> None:
    reverse_cpt = info.get("cpt_id_reverse_map") or {v: k for k, v in dict(info["cpt_id_map"]).items()}
    item_prior = info["item_prior_matrix"].detach().float().cpu().numpy()
    seq_prior = info["sequence_prior_matrix"].detach().float().cpu().numpy()
    edge_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    hist_rows: List[Dict[str, Any]] = []

    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        eval_row_id = int(row.eval_row_id)
        case_id = _case_id("E", rank, eval_row_id)
        details = details_by_eval[eval_row_id]
        sample_pos = int(getattr(row, "detail_pos", rank - 1))
        q_vector = details["q_vector"][sample_pos].float().numpy()
        query_concepts = [int(i) for i in np.where(q_vector > 0)[0]]
        dist = _support_distribution_for_case(details, sample_pos, query_concepts)
        if dist is None:
            continue
        query_c, edges = dist
        edges = edges.head(top_neighbors).copy()
        mapped_student = int(row.mapped_student_id)
        student_prior = details.get("_student_prior")
        student_recent = details.get("_student_recent")
        student_count = details.get("_student_count")
        related_raw = {int(reverse_cpt.get(query_c, query_c))}
        for e in edges.itertuples(index=False):
            support = int(e.support_mapped)
            related_raw.add(int(reverse_cpt.get(support, support)))
            item_val = float(item_prior[query_c, support])
            seq_val = float(seq_prior[query_c, support])
            edge_rows.append(
                {
                    "case_id": case_id,
                    "rank": rank,
                    "query_concept": _concept_label(query_c, reverse_cpt),
                    "support_concept": _concept_label(support, reverse_cpt),
                    "query_mapped": query_c,
                    "support_mapped": support,
                    "a_prior": float(e.a_prior),
                    "e_posterior": float(e.e_posterior),
                    "delta": float(e.delta),
                    "student_support_mastery_logit": _safe_float(student_prior[mapped_student, support]) if student_prior is not None else 0.0,
                    "student_support_recent_logit": _safe_float(student_recent[mapped_student, support]) if student_recent is not None else 0.0,
                    "student_support_count": _safe_float(student_count[mapped_student, support]) if student_count is not None else 0.0,
                    "item_prior": item_val,
                    "seq_prior": seq_val,
                    "source": _source_label(item_val, seq_val, query_c == support),
                    "is_self": int(query_c == support),
                }
            )
        case_rows.append(
            {
                "case_id": case_id,
                "rank": rank,
                "eval_row_id": eval_row_id,
                "stu_id": row.stu_id,
                "exer_id": row.exer_id,
                "label": row.label,
                "cpt_seq": row.cpt_seq,
                "full_prob": row.full_prob,
                "no_E_prob": row.no_E_prob,
                "E_shuffle_prob": getattr(row, "E_shuffle_prob", np.nan),
                "E_mean_prob": getattr(row, "E_mean_prob", np.nan),
                "e_gain": row.e_gain,
                "E_shuffle_gain": getattr(row, "E_shuffle_gain", np.nan),
                "E_mean_gain": getattr(row, "E_mean_gain", np.nan),
                "e_counterfactual_gain_min": getattr(row, "e_counterfactual_gain_min", np.nan),
                "query_concept": _concept_label(query_c, reverse_cpt),
                "query_row_posterior_kl": row.query_row_posterior_kl,
                "query_row_posterior_delta_abs": row.query_row_posterior_delta_abs,
                "query_row_personal_message_delta": row.query_row_personal_message_delta,
                "e_top_observed_support": getattr(row, "e_top_observed_support", ""),
                "e_top_observed_abs_delta": getattr(row, "e_top_observed_abs_delta", np.nan),
                "e_top_observed_support_count": getattr(row, "e_top_observed_support_count", np.nan),
                "e_observed_shift_abs": getattr(row, "e_observed_shift_abs", np.nan),
                "e_observed_state_abs": getattr(row, "e_observed_state_abs", np.nan),
            }
        )
        for h in _history_rows(train_df, row.stu_id, related_raw, history_window=history_window):
            h.update({"case_id": case_id, "rank": rank, "stu_id": row.stu_id})
            hist_rows.append(h)

    pd.DataFrame(case_rows).to_csv(out_dir / "e_cases.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(out_dir / "e_case_edges.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(out_dir / "e_case_history.csv", index=False)


def _write_e_contrast_tables(
    selected: pd.DataFrame,
    details_by_eval: Mapping[int, Dict[str, Any]],
    info: Mapping[str, Any],
    out_dir: Path,
    *,
    top_neighbors: int,
) -> None:
    reverse_cpt = info.get("cpt_id_reverse_map") or {v: k for k, v in dict(info["cpt_id_map"]).items()}
    item_prior = info["item_prior_matrix"].detach().float().cpu().numpy()
    seq_prior = info["sequence_prior_matrix"].detach().float().cpu().numpy()
    edge_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []

    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        eval_row_id = int(row.eval_row_id)
        case_id = _case_id("EC", rank, eval_row_id)
        details = details_by_eval[eval_row_id]
        sample_pos = int(getattr(row, "detail_pos", rank - 1))
        q_vector = details["q_vector"][sample_pos].float().numpy()
        query_concepts = [int(i) for i in np.where(q_vector > 0)[0]]
        dist = _support_distribution_for_case(details, sample_pos, query_concepts)
        if dist is None:
            continue
        query_c, edges = dist
        edges = edges.head(top_neighbors).copy()
        mapped_student = int(row.mapped_student_id)
        student_prior = details.get("_student_prior")
        student_recent = details.get("_student_recent")
        student_count = details.get("_student_count")
        for e in edges.itertuples(index=False):
            support = int(e.support_mapped)
            item_val = float(item_prior[query_c, support])
            seq_val = float(seq_prior[query_c, support])
            edge_rows.append(
                {
                    "case_id": case_id,
                    "rank": rank,
                    "stu_id": row.stu_id,
                    "eval_row_id": eval_row_id,
                    "query_concept": _concept_label(query_c, reverse_cpt),
                    "support_concept": _concept_label(support, reverse_cpt),
                    "query_mapped": query_c,
                    "support_mapped": support,
                    "a_prior": float(e.a_prior),
                    "e_posterior": float(e.e_posterior),
                    "delta": float(e.delta),
                    "student_support_mastery_logit": _safe_float(student_prior[mapped_student, support]) if student_prior is not None else 0.0,
                    "student_support_recent_logit": _safe_float(student_recent[mapped_student, support]) if student_recent is not None else 0.0,
                    "student_support_count": _safe_float(student_count[mapped_student, support]) if student_count is not None else 0.0,
                    "item_prior": item_val,
                    "seq_prior": seq_val,
                    "source": _source_label(item_val, seq_val, query_c == support),
                    "is_self": int(query_c == support),
                }
            )
        case_rows.append(
            {
                "case_id": case_id,
                "rank": rank,
                "eval_row_id": eval_row_id,
                "stu_id": row.stu_id,
                "exer_id": row.exer_id,
                "label": row.label,
                "cpt_seq": row.cpt_seq,
                "full_prob": row.full_prob,
                "no_E_prob": row.no_E_prob,
                "E_shuffle_prob": getattr(row, "E_shuffle_prob", np.nan),
                "E_mean_prob": getattr(row, "E_mean_prob", np.nan),
                "e_gain": row.e_gain,
                "query_concept": _concept_label(query_c, reverse_cpt),
                "e_top_observed_support": getattr(row, "e_top_observed_support", ""),
                "e_top_observed_support_count": getattr(row, "e_top_observed_support_count", np.nan),
                "e_observed_shift_abs": getattr(row, "e_observed_shift_abs", np.nan),
                "contrast_group_score": getattr(row, "contrast_group_score", np.nan),
            }
        )

    pd.DataFrame(case_rows).to_csv(out_dir / "e_student_contrast_cases.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(out_dir / "e_student_contrast_edges.csv", index=False)


def _attach_model_buffers_to_details(
    details_by_eval: Dict[int, Dict[str, Any]],
    model: CognitiveDiagnosisModel,
) -> None:
    prior = model.ae_student_concept_prior_logit.detach().float().cpu().numpy()
    recent = model.ae_student_concept_recent_logit.detach().float().cpu().numpy()
    count = model.ae_student_concept_observed_count.detach().float().cpu().numpy()
    for d in details_by_eval.values():
        d["_student_prior"] = prior
        d["_student_recent"] = recent
        d["_student_count"] = count


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_full = _load_checkpoint(args.full_dir, device)
    ckpt_no_a = _load_checkpoint(args.no_a_dir, device) if args.no_a_dir else None
    ckpt_no_e = _load_checkpoint(args.no_e_dir, device) if args.no_e_dir else None
    info = ckpt_full["info_dict"]
    if ckpt_no_a is not None:
        _assert_same_mapping(info, ckpt_no_a["info_dict"], "no_A")
    if ckpt_no_e is not None:
        _assert_same_mapping(info, ckpt_no_e["info_dict"], "no_E")

    dataset_name = args.dataset_name or dict(ckpt_full.get("args", {})).get("dataset_name", "assist_09")
    data_dir = _resolve_data_dir(dataset_name, args.data_dir)
    eval_frame = _make_eval_frame(info, data_dir, args.split)
    loader = _make_loader(eval_frame, info, batch_size=args.batch_size, num_workers=args.num_workers)

    full_model = _build_model(ckpt_full, device)
    labels, full_prob, full_logit = _predict_probs(full_model, loader, device)
    result = eval_frame.copy()
    result["mapped_student_id"] = result["stu_id"].map(info["stu_id_map"]).astype(int)
    result["mapped_exercise_id"] = result["exer_id"].map(info["exer_id_map"]).astype(int)
    q_matrix = info["q_matrix"].detach().float().cpu()
    result["q_count"] = q_matrix[result["mapped_exercise_id"].to_numpy()].sum(dim=1).numpy().astype(int)
    result["label_eval"] = labels
    result["full_prob"] = full_prob
    result["full_logit"] = full_logit
    result["full_pred"] = (result["full_prob"] > 0.5).astype(int)
    result["full_abs_error"] = (result["label"] - result["full_prob"]).abs()

    if ckpt_no_a is not None:
        no_a_model = _build_model(ckpt_no_a, device)
        _, no_a_prob, _ = _predict_probs(no_a_model, loader, device)
        result["no_A_prob"] = no_a_prob
        result["no_A_pred"] = (result["no_A_prob"] > 0.5).astype(int)
        result["no_A_abs_error"] = (result["label"] - result["no_A_prob"]).abs()
        result["a_gain"] = result["no_A_abs_error"] - result["full_abs_error"]
        result["a_rescue"] = (result["full_pred"] == result["label"]) & (result["no_A_pred"] != result["label"])

    if ckpt_no_e is not None:
        no_e_model = _build_model(ckpt_no_e, device)
        _, no_e_prob, _ = _predict_probs(no_e_model, loader, device)
        result["no_E_prob"] = no_e_prob
        result["no_E_pred"] = (result["no_E_prob"] > 0.5).astype(int)
        result["no_E_abs_error"] = (result["label"] - result["no_E_prob"]).abs()
        result["e_gain"] = result["no_E_abs_error"] - result["full_abs_error"]
        result["e_rescue"] = (result["full_pred"] == result["label"]) & (result["no_E_pred"] != result["label"])

    if args.e_counterfactuals:
        for mode, prefix in (("shuffle", "E_shuffle"), ("mean", "E_mean")):
            _, cf_prob, _ = _predict_e_counterfactual(
                ckpt_full,
                loader,
                device,
                mode,
                seed=args.counterfactual_seed,
            )
            result[f"{prefix}_prob"] = cf_prob
            result[f"{prefix}_pred"] = (result[f"{prefix}_prob"] > 0.5).astype(int)
            result[f"{prefix}_abs_error"] = (result["label"] - result[f"{prefix}_prob"]).abs()
            result[f"{prefix}_gain"] = result[f"{prefix}_abs_error"] - result["full_abs_error"]
        if "E_shuffle_gain" in result and "E_mean_gain" in result:
            result["e_counterfactual_gain_min"] = result[["E_shuffle_gain", "E_mean_gain"]].min(axis=1)
            result["e_counterfactual_rescue"] = (
                (result["full_pred"] == result["label"])
                & (result["E_shuffle_pred"] != result["label"])
                & (result["E_mean_pred"] != result["label"])
            )

    result.to_csv(out_dir / "case_predictions.csv", index=False)

    metric_rows: List[Dict[str, Any]] = []
    for prefix in ("full", "no_A", "no_E", "E_shuffle", "E_mean"):
        prob_col = f"{prefix}_prob"
        if prob_col not in result:
            continue
        probs = result[prob_col].to_numpy(dtype=np.float32)
        preds = (probs > 0.5).astype(np.float32)
        metrics = compute_metrics(result["label"].to_numpy(dtype=np.float32), preds, probs)
        metric_rows.append({"variant": prefix, **{k: float(v) for k, v in metrics.items()}})
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics_check.csv", index=False)

    selected_frames: List[pd.DataFrame] = []
    if "a_gain" in result:
        a_pool = _select_cases(
            result,
            "a_gain",
            "a_rescue",
            args.candidate_pool,
            min_gain=args.min_gain,
            prefer_multi_concept=True,
        )
        a_pool_details, _a_pool_detail_map = _run_selected_details(
            full_model,
            a_pool,
            info,
            device,
            context_frame=result,
            context_batch_size=args.batch_size,
        )
        a_pool_details = _add_a_mechanism_diagnostics(
            a_pool_details,
            _a_pool_detail_map,
            info,
            top_neighbors=args.top_neighbors,
        )
        a_pool_details.to_csv(out_dir / "a_candidate_pool.csv", index=False)
        a_details_sel = _rank_a_case_pool(a_pool_details, args.top_cases).reset_index(drop=True)
        a_details_sel, a_details = _run_selected_details(
            full_model,
            a_details_sel,
            info,
            device,
            context_frame=result,
            context_batch_size=args.batch_size,
        )
        a_details_sel = _add_a_mechanism_diagnostics(
            a_details_sel,
            a_details,
            info,
            top_neighbors=args.top_neighbors,
        )
        _write_a_tables(a_details_sel, a_details, info, out_dir, top_neighbors=args.top_neighbors)
        a_details_sel.to_csv(out_dir / "a_selected_cases.csv", index=False)
        selected_frames.append(a_details_sel.assign(case_type="A"))

    if "e_gain" in result:
        e_pool = _select_cases(
            result,
            "e_gain",
            "e_rescue",
            args.candidate_pool,
            min_gain=args.min_gain,
            prefer_multi_concept=False,
        )
        e_pool_details, e_pool_detail_map = _run_selected_details(
            full_model,
            e_pool,
            info,
            device,
            context_frame=result,
            context_batch_size=args.batch_size,
        )
        e_pool_details = _add_e_mechanism_diagnostics(e_pool_details, e_pool_detail_map, info, full_model)
        e_pool_details.to_csv(out_dir / "e_candidate_pool.csv", index=False)
        e_details_sel = _rank_e_case_pool(e_pool_details, args.top_cases).reset_index(drop=True)
        e_details_sel, e_details = _run_selected_details(
            full_model,
            e_details_sel,
            info,
            device,
            context_frame=result,
            context_batch_size=args.batch_size,
        )
        e_details_sel = _add_e_mechanism_diagnostics(e_details_sel, e_details, info, full_model)
        _attach_model_buffers_to_details(e_details, full_model)
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        _write_e_tables(
            e_details_sel,
            e_details,
            info,
            train_df,
            out_dir,
            top_neighbors=args.top_neighbors,
            history_window=args.history_window,
        )
        e_details_sel.to_csv(out_dir / "e_selected_cases.csv", index=False)
        selected_frames.append(e_details_sel.assign(case_type="E"))

        contrast_sel = _rank_e_contrast_pool(e_pool_details, args.contrast_cases)
        if not contrast_sel.empty:
            contrast_sel, contrast_details = _run_selected_details(
                full_model,
                contrast_sel,
                info,
                device,
                context_frame=result,
                context_batch_size=args.batch_size,
            )
            contrast_sel = _add_e_mechanism_diagnostics(contrast_sel, contrast_details, info, full_model)
            _attach_model_buffers_to_details(contrast_details, full_model)
            _write_e_contrast_tables(
                contrast_sel,
                contrast_details,
                info,
                out_dir,
                top_neighbors=args.top_neighbors,
            )
            contrast_sel.to_csv(out_dir / "e_student_contrast_selected_cases.csv", index=False)

    if selected_frames:
        pd.concat(selected_frames, ignore_index=True).to_csv(out_dir / "selected_cases.csv", index=False)

    summary = {
        "dataset": dataset_name,
        "split": args.split,
        "output_dir": str(out_dir),
        "num_eval_rows": int(len(result)),
        "checkpoints": {
            "full": str(_checkpoint_path(args.full_dir)),
            "no_A": str(_checkpoint_path(args.no_a_dir)) if args.no_a_dir else "",
            "no_E": str(_checkpoint_path(args.no_e_dir)) if args.no_e_dir else "",
        },
        "metrics": metric_rows,
        "case_counts": {
            "a_rescue": int(result["a_rescue"].sum()) if "a_rescue" in result else 0,
            "e_rescue": int(result["e_rescue"].sum()) if "e_rescue" in result else 0,
            "a_positive_gain": int((result["a_gain"] > 0).sum()) if "a_gain" in result else 0,
            "e_positive_gain": int((result["e_gain"] > 0).sum()) if "e_gain" in result else 0,
        },
    }
    (out_dir / "case_study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="assist_09")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--full_dir", required=True)
    parser.add_argument("--no_a_dir", default="")
    parser.add_argument("--no_e_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_cases", type=int, default=3)
    parser.add_argument("--candidate_pool", type=int, default=40)
    parser.add_argument("--min_gain", type=float, default=0.10)
    parser.add_argument("--top_neighbors", type=int, default=8)
    parser.add_argument("--history_window", type=int, default=12)
    parser.add_argument("--contrast_cases", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--e_counterfactuals", action="store_true", help="Evaluate shuffled/mean student-state E counterfactuals.")
    parser.add_argument("--counterfactual_seed", type=int, default=20260513)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
