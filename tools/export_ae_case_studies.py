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
    return out.sort_values(
        ["selection_score", "a_gain", "query_row_global_readout_delta"],
        ascending=[False, False, False],
    ).head(top_k)


def _rank_e_case_pool(candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    out["selection_score"] = out["e_gain"].astype(float)
    out["selection_score"] += out["query_row_posterior_kl"].astype(float) * 0.60
    out["selection_score"] += out["query_row_posterior_delta_abs"].astype(float) * 0.30
    out["selection_score"] += out["query_row_personal_message_delta"].astype(float) * 0.20
    e_rescue = out["e_rescue"] if "e_rescue" in out else pd.Series(False, index=out.index)
    out["quality_reason"] = np.where(
        e_rescue,
        "full_correct_no_E_wrong",
        "largest_no_E_error_reduction_with_posterior_shift",
    )
    return out.sort_values(
        ["selection_score", "e_gain", "query_row_posterior_kl", "query_row_posterior_delta_abs"],
        ascending=[False, False, False, False],
    ).head(top_k)


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


def _run_selected_details(
    model: CognitiveDiagnosisModel,
    selected: pd.DataFrame,
    info: Mapping[str, Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    if selected.empty:
        return selected, {}
    loader = _make_loader(selected.reset_index(drop=True), info, batch_size=len(selected), num_workers=0)
    batch = next(iter(loader))
    student_ids, exercise_ids, labels = batch
    with torch.no_grad():
        logits, details = model(
            student_ids.to(device),
            exercise_ids.to(device),
            return_details=True,
            return_logits=True,
        )
    selected = selected.copy().reset_index(drop=True)
    selected["detail_pos"] = np.arange(len(selected), dtype=np.int64)
    selected["full_logit_detail"] = logits.detach().cpu().numpy().reshape(-1)
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
    for key in scalar_keys:
        selected[key] = [_extract_scalar(details, key, i) for i in range(len(selected))]
    detail_by_eval = {
        int(selected.loc[i, "eval_row_id"]): {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in details.items()}
        for i in range(len(selected))
    }
    return selected, detail_by_eval


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
                "e_gain": row.e_gain,
                "query_concept": _concept_label(query_c, reverse_cpt),
                "query_row_posterior_kl": row.query_row_posterior_kl,
                "query_row_posterior_delta_abs": row.query_row_posterior_delta_abs,
                "query_row_personal_message_delta": row.query_row_personal_message_delta,
            }
        )
        for h in _history_rows(train_df, row.stu_id, related_raw, history_window=history_window):
            h.update({"case_id": case_id, "rank": rank, "stu_id": row.stu_id})
            hist_rows.append(h)

    pd.DataFrame(case_rows).to_csv(out_dir / "e_cases.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(out_dir / "e_case_edges.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(out_dir / "e_case_history.csv", index=False)


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

    result.to_csv(out_dir / "case_predictions.csv", index=False)

    metric_rows: List[Dict[str, Any]] = []
    for prefix in ("full", "no_A", "no_E"):
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
        a_pool_details, a_pool_detail_map = _run_selected_details(full_model, a_pool, info, device)
        a_pool_details.to_csv(out_dir / "a_candidate_pool.csv", index=False)
        a_details_sel = _rank_a_case_pool(a_pool_details, args.top_cases).reset_index(drop=True)
        a_details_sel["detail_pos"] = np.arange(len(a_details_sel), dtype=np.int64)
        a_details_sel, a_details = _run_selected_details(full_model, a_details_sel, info, device)
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
        e_pool_details, _ = _run_selected_details(full_model, e_pool, info, device)
        e_pool_details.to_csv(out_dir / "e_candidate_pool.csv", index=False)
        e_details_sel = _rank_e_case_pool(e_pool_details, args.top_cases).reset_index(drop=True)
        e_details_sel["detail_pos"] = np.arange(len(e_details_sel), dtype=np.int64)
        e_details_sel, e_details = _run_selected_details(full_model, e_details_sel, info, device)
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
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
