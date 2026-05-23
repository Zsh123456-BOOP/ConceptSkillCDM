#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main-problem diagnostics for the CRG/LCRF paper story.

This script is intentionally inference-only. It never trains, never changes
model structure, and writes a missing report when a requested checkpoint/control
cannot be evaluated from existing artifacts.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CognitiveDiagnosisDataset, _parse_concept_seq, make_degree_random_prior  # noqa: E402
from src.experiment_utils import compute_metrics  # noqa: E402
from tools.analyze_a_support_evidence import _row_normalize, _self_only  # noqa: E402
from tools.analyze_ae_errors import _resolve_data_dir  # noqa: E402
from tools.analyze_crg_support_corruption_controls import (  # noqa: E402
    _clone_masks,
    _restore_masks,
    _set_masks,
)
from tools.export_ae_case_studies import _build_model, _load_checkpoint  # noqa: E402


DATASETS = ("assist_09", "junyi", "assist_17")
BASE_VARIANTS = ("full", "no_CRG", "no_LCRF")
CONTROL_VARIANTS = ("self_only", "degree_random_support")
PRED_VARIANTS = (*BASE_VARIANTS, *CONTROL_VARIANTS, "global_only")


@dataclass
class DatasetAssets:
    dataset: str
    data_dir: Path
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    info: Mapping[str, Any]
    checkpoint_paths: Dict[str, Path]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_main_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing main ablation table: {path}")
    table = pd.read_csv(path)
    required = {"dataset", "variant", "checkpoint_path"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return table


def _variant_checkpoint_paths(main_table: pd.DataFrame, dataset: str) -> Dict[str, Path]:
    rows = main_table[main_table["dataset"].astype(str).eq(dataset)]
    paths: Dict[str, Path] = {}
    for variant in BASE_VARIANTS:
        hit = rows[rows["variant"].astype(str).eq(variant)]
        if not hit.empty:
            raw = str(hit.iloc[0]["checkpoint_path"])
            paths[variant] = Path(raw)
    return paths


def _load_assets(dataset: str, main_table: pd.DataFrame, device: torch.device, missing: List[str]) -> Optional[DatasetAssets]:
    ckpt_paths = _variant_checkpoint_paths(main_table, dataset)
    if "full" not in ckpt_paths:
        missing.append(f"{dataset}: missing full checkpoint in main table")
        return None
    if not ckpt_paths["full"].exists():
        missing.append(f"{dataset}: full checkpoint path does not exist: {ckpt_paths['full']}")
        return None
    full_ckpt = _load_checkpoint(ckpt_paths["full"], device)
    info = full_ckpt["info_dict"]
    data_dir = Path(_resolve_data_dir(dataset, None))
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        missing.append(f"{dataset}: missing split files under {data_dir}")
        return None
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    valid_stu = set(info["stu_id_map"].keys())
    valid_exer = set(info["exer_id_map"].keys())
    train_df = train_df[train_df["stu_id"].isin(valid_stu)].reset_index(drop=True)
    test_df = test_df[test_df["stu_id"].isin(valid_stu) & test_df["exer_id"].isin(valid_exer)].reset_index(drop=True)
    return DatasetAssets(dataset, data_dir, train_df, test_df, info, ckpt_paths)


def _ordered_groups(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    ordered = df.copy()
    ordered["_source_order"] = np.arange(len(ordered))
    order_cols = [c for c in ("timestamp", "time", "order_id", "original_row_id") if c in ordered.columns]
    ordered = ordered.sort_values(["stu_id", *order_cols, "_source_order"], kind="mergesort")
    for _, group in ordered.groupby("stu_id", sort=False):
        yield group


def _student_history_stats(train_df: pd.DataFrame, cpt_id_map: Mapping[int, int]) -> Dict[str, Dict[Any, Any]]:
    concept_sets: Dict[Any, set] = {}
    concept_counts: Dict[Any, Dict[int, int]] = {}
    correct_sum: Dict[Any, Dict[int, float]] = {}
    recent_label: Dict[Any, Dict[int, float]] = {}
    total_counts: Dict[Any, int] = {}
    for stu_df in _ordered_groups(train_df):
        sid = stu_df["stu_id"].iloc[0]
        concept_sets.setdefault(sid, set())
        concept_counts.setdefault(sid, {})
        correct_sum.setdefault(sid, {})
        recent_label.setdefault(sid, {})
        total_counts[sid] = int(len(stu_df))
        for row in stu_df.itertuples(index=False):
            label = float(getattr(row, "label"))
            for raw_c in _parse_concept_seq(getattr(row, "cpt_seq")):
                if raw_c not in cpt_id_map:
                    continue
                c = int(cpt_id_map[raw_c])
                concept_sets[sid].add(c)
                concept_counts[sid][c] = concept_counts[sid].get(c, 0) + 1
                correct_sum[sid][c] = correct_sum[sid].get(c, 0.0) + label
                recent_label[sid][c] = label
    return {
        "sets": concept_sets,
        "counts": concept_counts,
        "correct_sum": correct_sum,
        "recent_label": recent_label,
        "total_counts": total_counts,
    }


def _event_concepts(row: Any, cpt_id_map: Mapping[int, int]) -> List[int]:
    return sorted({int(cpt_id_map[c]) for c in _parse_concept_seq(getattr(row, "cpt_seq")) if c in cpt_id_map})


def _prior_variants(info: Mapping[str, Any], seed: int = 20260523) -> Dict[str, torch.Tensor]:
    item = info["item_prior_matrix"].detach().cpu().float()
    seq = info["sequence_prior_matrix"].detach().cpu().float()
    degree_random, _ = make_degree_random_prior(item, seq, seed=seed)
    c = int(item.size(0))
    gen = torch.Generator().manual_seed(seed + c * 997)
    random_scores = torch.rand(c, c, generator=gen, dtype=torch.float32)
    random_scores = random_scores * (1.0 - torch.eye(c, dtype=torch.float32))
    random_prior = random_scores / random_scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return {
        "self-only": _self_only(c),
        "random": random_prior,
        "degree-random": degree_random,
        "item-only": _row_normalize(item),
        "seq-only": _row_normalize(seq),
        "fused CRG": _row_normalize(item + seq),
    }


def _route_scores_from_history(prior: np.ndarray, history: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    c = prior.shape[0]
    if not history:
        return np.zeros(c, dtype=np.float64), np.zeros(c, dtype=np.float64)
    h = np.array(sorted(set(int(x) for x in history if 0 <= int(x) < c)), dtype=np.int64)
    if h.size == 0:
        return np.zeros(c, dtype=np.float64), np.zeros(c, dtype=np.float64)
    edge = np.clip(prior[h, :].astype(np.float64), 0.0, 1.0)
    noisy_or = 1.0 - np.prod(1.0 - edge, axis=0)
    max_score = edge.max(axis=0)
    return noisy_or, max_score


def _rank_of_target(scores: np.ndarray, target: int) -> int:
    if scores.size == 0 or not np.isfinite(scores).any() or float(np.max(scores)) <= 0.0:
        return int(scores.size + 1)
    order = np.argsort(-scores, kind="mergesort")
    found = np.where(order == int(target))[0]
    return int(found[0]) + 1 if len(found) else int(scores.size + 1)


def _metric_from_rank(rank: int, k: int) -> float:
    return 1.0 if rank <= k else 0.0


def run_exp1(assets: DatasetAssets, out_dir: Path, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    history = _student_history_stats(assets.train_df, assets.info["cpt_id_map"])
    variants = {k: v.numpy() for k, v in _prior_variants(assets.info, seed).items()}
    fused = variants["fused CRG"]
    rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    for event_id, row in enumerate(assets.test_df.itertuples(index=False)):
        sid = getattr(row, "stu_id")
        h = sorted(history["sets"].get(sid, set()))
        concepts = _event_concepts(row, assets.info["cpt_id_map"])
        if not concepts:
            continue
        fused_noisy, fused_max = _route_scores_from_history(fused, h)
        for c in concepts:
            direct_seen = c in set(h)
            bridge = (not direct_seen) and (float(fused_noisy[c]) > 0.0)
            group_flags = {
                "all": True,
                "direct_seen": direct_seen,
                "direct_unseen": not direct_seen,
                "direct_unseen_bridgeable": bridge,
            }
            for method, prior in variants.items():
                noisy, max_score = _route_scores_from_history(prior, h)
                rank = _rank_of_target(noisy, c)
                base = {
                    "dataset": assets.dataset,
                    "event_id": event_id,
                    "stu_id": sid,
                    "query_concept": c,
                    "history_size": len(h),
                    "method": method,
                    "noisy_or_score": float(noisy[c]) if c < noisy.size else 0.0,
                    "max_score": float(max_score[c]) if c < max_score.size else 0.0,
                    "rank": rank,
                    "hit@5": _metric_from_rank(rank, 5),
                    "hit@10": _metric_from_rank(rank, 10),
                    "ndcg@10": 0.0 if rank > 10 else float(1.0 / math.log2(rank + 1.0)),
                    "mrr": 0.0 if rank > noisy.size else float(1.0 / rank),
                    "direct_seen": direct_seen,
                    "direct_unseen": not direct_seen,
                    "direct_unseen_bridgeable": bridge,
                }
                for group, keep in group_flags.items():
                    if keep:
                        item = dict(base)
                        item["group"] = group
                        rows.append(item)
            if bridge and len(case_rows) < 8:
                top_hist = sorted(h, key=lambda x: fused[x, c] if x < fused.shape[0] else 0.0, reverse=True)[:5]
                for source in top_hist:
                    case_rows.append(
                        {
                            "dataset": assets.dataset,
                            "event_id": event_id,
                            "student_id_anonymized": f"S{len(case_rows) + 1}",
                            "source_concept": source,
                            "query_concept": c,
                            "route_prob": float(fused[source, c]),
                            "history_size": len(h),
                        }
                    )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["dataset", "group", "method"], as_index=False)
        .agg(
            n_eval=("event_id", "count"),
            hit5=("hit@5", "mean"),
            hit10=("hit@10", "mean"),
            ndcg10=("ndcg@10", "mean"),
            mrr=("mrr", "mean"),
            noisy_or_mean=("noisy_or_score", "mean"),
            max_score_mean=("max_score", "mean"),
        )
    )
    detail.to_csv(out_dir / "main_problem_exp1_history_to_query_route_retrieval.csv", index=False)
    summary.to_csv(out_dir / "main_problem_exp1_history_to_query_route_summary.csv", index=False)
    pd.DataFrame(case_rows).to_csv(out_dir / "main_problem_exp1_route_case.csv", index=False)
    return detail, summary


def _make_eval_frame(assets: DatasetAssets) -> pd.DataFrame:
    frame = assets.test_df.reset_index(drop=True).copy()
    frame["event_id"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _predict_model(model: torch.nn.Module, frame: pd.DataFrame, info: Mapping[str, Any], device: torch.device, batch_size: int, num_workers: int) -> pd.DataFrame:
    dataset = CognitiveDiagnosisDataset(frame, info["stu_id_map"], info["exer_id_map"], info["cpt_id_map"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    mapped_students: List[np.ndarray] = []
    mapped_exercises: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for student_ids, exercise_ids, y in loader:
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            logits = model(student_ids, exercise_ids, return_details=False, return_logits=True).reshape(-1)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            mapped_students.append(student_ids.detach().cpu().numpy())
            mapped_exercises.append(exercise_ids.detach().cpu().numpy())
    out = frame[["event_id", "stu_id", "exer_id", "cpt_seq", "label"]].copy()
    out["label_eval"] = np.concatenate(labels).astype(np.float32)
    out["mapped_student_id"] = np.concatenate(mapped_students).astype(np.int64)
    out["mapped_exercise_id"] = np.concatenate(mapped_exercises).astype(np.int64)
    out["prob"] = np.concatenate(probs).astype(np.float32)
    return out


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    labels = labels.astype(np.float32)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    return float(compute_metrics(labels, (probs >= 0.5).astype(np.float32), probs)["auc"])


def _bce(labels: np.ndarray, probs: np.ndarray) -> float:
    y = labels.astype(np.float64)
    p = probs.astype(np.float64).clip(1e-8, 1.0 - 1e-8)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))) if len(y) else float("nan")


def _rmse(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((labels.astype(np.float64) - probs.astype(np.float64)) ** 2))) if len(labels) else float("nan")


def _acc(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs >= 0.5).astype(np.float32) == labels.astype(np.float32))) if len(labels) else float("nan")


def _metric_row(dataset: str, subgroup: str, variant: str, labels: np.ndarray, probs: np.ndarray, full_ref: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    auc = _safe_auc(labels, probs)
    bce = _bce(labels, probs)
    row = {
        "dataset": dataset,
        "subgroup": subgroup,
        "variant": variant,
        "n_eval": int(len(labels)),
        "auc": auc,
        "bce": bce,
        "rmse": _rmse(labels, probs),
        "acc": _acc(labels, probs),
    }
    if full_ref is not None:
        row["auc_gap_full_minus_variant"] = None if auc is None or full_ref.get("auc") is None else float(full_ref["auc"] - auc)
        row["bce_gap_variant_minus_full"] = float(bce - full_ref["bce"])
    return row


def _annotate_coverage(frame: pd.DataFrame, assets: DatasetAssets) -> pd.DataFrame:
    history = _student_history_stats(assets.train_df, assets.info["cpt_id_map"])
    fused = _row_normalize(
        assets.info["item_prior_matrix"].detach().cpu()
        + assets.info["sequence_prior_matrix"].detach().cpu()
    ).numpy()
    direct_seen: List[bool] = []
    bridgeable: List[bool] = []
    weak_direct: List[bool] = []
    route_mass: List[float] = []
    hist_count: List[int] = []
    query_concepts_list: List[str] = []
    for row in frame.itertuples(index=False):
        sid = getattr(row, "stu_id")
        hset = set(history["sets"].get(sid, set()))
        hcount = history["counts"].get(sid, {})
        concepts = _event_concepts(row, assets.info["cpt_id_map"])
        query_concepts_list.append(",".join(str(c) for c in concepts))
        hist_count.append(int(history["total_counts"].get(sid, 0)))
        seen_vals = [c in hset for c in concepts]
        count_vals = [int(hcount.get(c, 0)) for c in concepts]
        scores = []
        for c in concepts:
            noisy, _ = _route_scores_from_history(fused, hset)
            scores.append(float(noisy[c]) if c < noisy.size else 0.0)
        direct_seen.append(bool(any(seen_vals)))
        bridgeable.append(bool(any((not seen) and s > 0.0 for seen, s in zip(seen_vals, scores))))
        weak_direct.append(bool(min(count_vals, default=0) <= 1))
        route_mass.append(float(max(scores)) if scores else 0.0)
    out = frame.copy()
    out["query_concepts_mapped"] = query_concepts_list
    out["student_train_history_len"] = hist_count
    out["direct_seen"] = direct_seen
    out["direct_unseen"] = ~out["direct_seen"]
    out["direct_unseen_bridgeable"] = out["direct_unseen"] & pd.Series(bridgeable, index=out.index)
    out["direct_unseen_unbridgeable"] = out["direct_unseen"] & ~out["direct_unseen_bridgeable"]
    out["weak_direct_evidence"] = weak_direct
    out["route_mass_to_query"] = route_mass
    if len(out) and np.any(np.asarray(route_mass) > 0):
        high = float(np.quantile(route_mass, 0.70))
        low = float(np.quantile(route_mass, 0.30))
    else:
        high, low = float("inf"), 0.0
    out["high_route_mass"] = out["route_mass_to_query"] >= high
    out["low_route_mass"] = out["route_mass_to_query"] <= low
    return out


def _load_model_for_variant(assets: DatasetAssets, variant: str, device: torch.device, missing: List[str]) -> Optional[torch.nn.Module]:
    path = assets.checkpoint_paths.get(variant)
    if path is None:
        missing.append(f"{assets.dataset}: missing {variant} checkpoint in main table")
        return None
    if not path.exists():
        missing.append(f"{assets.dataset}: {variant} checkpoint path does not exist: {path}")
        return None
    ckpt = _load_checkpoint(path, device)
    return _build_model(ckpt, device)


def _apply_control_variant(model: torch.nn.Module, info: Mapping[str, Any], variant: str, seed: int) -> Optional[Mapping[str, torch.Tensor]]:
    original = _clone_masks(model)
    rel = model.structure_module.relation_learning
    if variant == "self_only":
        z = torch.zeros_like(rel.item_support_mask.detach().cpu()).bool()
        _set_masks(model, z, z)
    elif variant == "degree_random_support":
        item = info["item_prior_matrix"].detach().cpu().float()
        seq = info["sequence_prior_matrix"].detach().cpu().float()
        random_prior, _ = make_degree_random_prior(item, seq, seed=seed)
        mask = random_prior > 0
        _set_masks(model, mask, torch.zeros_like(mask))
    else:
        _restore_masks(model, original)
        return None
    return original


def _group_masks(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    groups = {
        "all": np.ones(len(frame), dtype=bool),
        "direct_seen": frame["direct_seen"].to_numpy(dtype=bool),
        "direct_unseen": frame["direct_unseen"].to_numpy(dtype=bool),
        "direct_unseen_bridgeable": frame["direct_unseen_bridgeable"].to_numpy(dtype=bool),
        "direct_unseen_unbridgeable": frame["direct_unseen_unbridgeable"].to_numpy(dtype=bool),
        "weak_direct_evidence": frame["weak_direct_evidence"].to_numpy(dtype=bool),
        "high_route_mass": frame["high_route_mass"].to_numpy(dtype=bool),
        "low_route_mass": frame["low_route_mass"].to_numpy(dtype=bool),
    }
    return groups


def run_exp2(
    assets: DatasetAssets,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    missing: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = _annotate_coverage(_make_eval_frame(assets), assets)
    predictions = frame[["event_id", "stu_id", "exer_id", "label", "query_concepts_mapped", "direct_seen", "direct_unseen", "direct_unseen_bridgeable", "direct_unseen_unbridgeable", "weak_direct_evidence", "high_route_mass", "low_route_mass", "route_mass_to_query"]].copy()
    variant_probs: Dict[str, np.ndarray] = {}
    for variant in BASE_VARIANTS:
        model = _load_model_for_variant(assets, variant, device, missing)
        if model is None:
            continue
        pred = _predict_model(model, frame, assets.info, device, batch_size, num_workers)
        predictions[f"prob_{variant}"] = pred["prob"].to_numpy()
        predictions["label_eval"] = pred["label_eval"].to_numpy()
        variant_probs[variant] = pred["prob"].to_numpy()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    full_model = _load_model_for_variant(assets, "full", device, missing)
    if full_model is not None:
        for variant in CONTROL_VARIANTS:
            original = _apply_control_variant(full_model, assets.info, variant, seed)
            pred = _predict_model(full_model, frame, assets.info, device, batch_size, num_workers)
            predictions[f"prob_{variant}"] = pred["prob"].to_numpy()
            variant_probs[variant] = pred["prob"].to_numpy()
            if original is not None:
                _restore_masks(full_model, original)
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    missing.append(f"{assets.dataset}: global_only control skipped; no existing checkpoint or exact inference hook for a pure global-only prediction path")

    labels = predictions["label_eval"].to_numpy(dtype=np.float32) if "label_eval" in predictions else predictions["label"].to_numpy(dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for subgroup, mask in _group_masks(frame).items():
        if int(mask.sum()) == 0:
            continue
        full_ref = None
        if "full" in variant_probs:
            full_ref = {
                "auc": _safe_auc(labels[mask], variant_probs["full"][mask]),
                "bce": _bce(labels[mask], variant_probs["full"][mask]),
            }
        for variant, probs in variant_probs.items():
            rows.append(_metric_row(assets.dataset, subgroup, variant, labels[mask], probs[mask], full_ref))
    pred_out = out_dir / "main_problem_exp2_coverage_conditioned_predictions.csv"
    met_out = out_dir / "main_problem_exp2_coverage_conditioned_metrics.csv"
    predictions.to_csv(pred_out, index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(met_out, index=False)
    return predictions, metrics


def _select_exp3_samples(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    direct = frame[frame["direct_seen"]].copy().reset_index(drop=True)
    if max_samples > 0 and len(direct) > max_samples:
        direct = direct.sample(n=max_samples, random_state=seed).sort_values("event_id").reset_index(drop=True)
    return direct


def _single_predict(model: torch.nn.Module, row: pd.Series, info: Mapping[str, Any], device: torch.device) -> float:
    dataset = CognitiveDiagnosisDataset(pd.DataFrame([row]), info["stu_id_map"], info["exer_id_map"], info["cpt_id_map"])
    sid, eid, _ = dataset[0]
    with torch.no_grad():
        logits = model(sid.view(1).to(device), eid.view(1).to(device), return_details=False, return_logits=True).reshape(-1)
        return float(torch.sigmoid(logits)[0].detach().cpu().item())


def _mask_student_concepts(model: torch.nn.Module, mapped_student: int, concepts: Sequence[int]) -> Dict[str, torch.Tensor]:
    buffers = {}
    names = [
        "ae_student_concept_prior_logit",
        "ae_student_concept_recent_logit",
        "ae_student_concept_count_feature",
        "ae_student_concept_observed_count",
    ]
    for name in names:
        tensor = getattr(model, name, None)
        if tensor is None:
            continue
        buffers[name] = tensor[int(mapped_student), list(concepts)].detach().clone()
        tensor[int(mapped_student), list(concepts)] = 0.0
    return buffers


def _restore_student_concepts(model: torch.nn.Module, mapped_student: int, concepts: Sequence[int], buffers: Mapping[str, torch.Tensor]) -> None:
    for name, saved in buffers.items():
        tensor = getattr(model, name, None)
        if tensor is not None:
            tensor[int(mapped_student), list(concepts)] = saved.to(device=tensor.device, dtype=tensor.dtype)


def _random_mask_concepts(row: pd.Series, assets: DatasetAssets, history: Mapping[str, Dict[Any, Any]], query_concepts: Sequence[int], seed: int) -> List[int]:
    sid = row["stu_id"]
    h = sorted(set(history["sets"].get(sid, set())) - set(query_concepts))
    if not h:
        return []
    rng = np.random.default_rng(seed + int(row["event_id"]) * 17)
    take = min(len(h), max(1, len(query_concepts)))
    return sorted(rng.choice(np.asarray(h, dtype=np.int64), size=take, replace=False).astype(int).tolist())


def run_exp3(
    assets: DatasetAssets,
    out_dir: Path,
    device: torch.device,
    seed: int,
    max_samples: int,
    missing: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = _annotate_coverage(_make_eval_frame(assets), assets)
    direct = _select_exp3_samples(frame, max_samples, seed)
    history = _student_history_stats(assets.train_df, assets.info["cpt_id_map"])
    variants = [v for v in ("full", "no_CRG", "no_LCRF") if v in assets.checkpoint_paths]
    variants.extend(["self_only"])
    sample_rows: List[Dict[str, Any]] = []
    for variant in variants:
        base_variant = "full" if variant == "self_only" else variant
        model = _load_model_for_variant(assets, base_variant, device, missing)
        if model is None:
            continue
        original_masks = None
        if variant == "self_only":
            original_masks = _apply_control_variant(model, assets.info, "self_only", seed)
        for row in direct.itertuples(index=False):
            row_s = pd.Series(row._asdict())
            concepts = _event_concepts(row, assets.info["cpt_id_map"])
            mapped_student = int(assets.info["stu_id_map"][getattr(row, "stu_id")])
            label = float(getattr(row, "label"))
            random_concepts = _random_mask_concepts(row_s, assets, history, concepts, seed)
            for mask_type, mask_concepts in (
                ("original", []),
                ("target_concept_mask", concepts),
                ("random_history_mask", random_concepts),
            ):
                saved = {}
                if mask_concepts:
                    saved = _mask_student_concepts(model, mapped_student, mask_concepts)
                prob = _single_predict(model, row_s, assets.info, device)
                if saved:
                    _restore_student_concepts(model, mapped_student, mask_concepts, saved)
                p = min(max(prob, 1e-8), 1.0 - 1e-8)
                bce = -(label * math.log(p) + (1.0 - label) * math.log(1.0 - p))
                sample_rows.append(
                    {
                        "dataset": assets.dataset,
                        "event_id": int(getattr(row, "event_id")),
                        "variant": variant,
                        "mask_type": mask_type,
                        "label": label,
                        "pred_prob": prob,
                        "bce": bce,
                        "query_concepts_mapped": ",".join(str(c) for c in concepts),
                        "masked_concepts": ",".join(str(c) for c in mask_concepts),
                        "counterfactual_scope": "student_concept_state_buffers",
                    }
                )
        if original_masks is not None:
            _restore_masks(model, original_masks)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    missing.append(
        f"{assets.dataset}: exp3 masks aggregated student-concept state buffers; exact raw-history recomputation hook was not available"
    )
    sample = pd.DataFrame(sample_rows)
    rows: List[Dict[str, Any]] = []
    if not sample.empty:
        for (variant, mask_type), group in sample.groupby(["variant", "mask_type"]):
            labels = group["label"].to_numpy(dtype=np.float32)
            probs = group["pred_prob"].to_numpy(dtype=np.float32)
            rows.append(_metric_row(assets.dataset, "direct_seen_sample", str(variant), labels, probs, None) | {"mask_type": str(mask_type)})
        pivot = sample.pivot_table(index=["event_id", "variant"], columns="mask_type", values="bce", aggfunc="first").reset_index()
        if "original" in pivot and "target_concept_mask" in pivot:
            pivot["target_bce_increase"] = pivot["target_concept_mask"] - pivot["original"]
        if "original" in pivot and "random_history_mask" in pivot:
            pivot["random_bce_increase"] = pivot["random_history_mask"] - pivot["original"]
        inc = pivot.groupby("variant", as_index=False).agg(
            paired_n=("event_id", "count"),
            target_bce_increase=("target_bce_increase", "mean"),
            random_bce_increase=("random_bce_increase", "mean"),
        )
        for item in inc.to_dict("records"):
            rows.append({"dataset": assets.dataset, "subgroup": "paired_bce_increase", **item})
    sample.to_csv(out_dir / "main_problem_exp3_direct_evidence_removal_sample.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "main_problem_exp3_direct_evidence_removal_summary.csv", index=False)
    return sample, summary


def _plot_figures(out_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_root / "plot_missing_report.md").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    fig_dir = _ensure_dir(out_root / "paper_figures")

    def _read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for path in paths:
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    exp1 = _read_csvs(out_root.glob("*/main_problem_exp1_history_to_query_route_summary.csv"))
    if exp1.empty:
        return
    focus = exp1[exp1["group"].eq("direct_unseen_bridgeable") & exp1["method"].isin(["self-only", "random", "degree-random", "seq-only", "fused CRG"])].copy()
    if not focus.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        labels = sorted(focus["dataset"].unique())
        methods = ["self-only", "random", "degree-random", "seq-only", "fused CRG"]
        x = np.arange(len(labels))
        width = 0.15
        for i, method in enumerate(methods):
            vals = []
            for ds in labels:
                hit = focus[(focus["dataset"].eq(ds)) & (focus["method"].eq(method))]
                vals.append(float(hit["hit10"].iloc[0]) if not hit.empty else np.nan)
            ax.bar(x + (i - 2) * width, vals, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Hit@10")
        ax.set_title("History-to-query route retrieval")
        ax.legend(fontsize=7, ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_main_problem_exp1_route_retrieval.pdf")
        fig.savefig(fig_dir / "fig_main_problem_exp1_route_retrieval.png", dpi=220)
        plt.close(fig)

    exp2 = _read_csvs(out_root.glob("*/main_problem_exp2_coverage_conditioned_metrics.csv"))
    if not exp2.empty:
        focus2 = exp2[exp2["subgroup"].isin(["direct_seen", "direct_unseen_bridgeable", "high_route_mass"]) & exp2["variant"].isin(["no_CRG", "no_LCRF", "self_only", "degree_random_support"])].copy()
        if not focus2.empty:
            fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), sharex=False)
            for ax, metric, ylabel in (
                (axes[0], "auc_gap_full_minus_variant", "Full - variant AUC"),
                (axes[1], "bce_gap_variant_minus_full", "Variant - full BCE"),
            ):
                plot_df = focus2.dropna(subset=[metric]).copy()
                labels = [f"{r.dataset}\n{r.subgroup}\n{r.variant}" for r in plot_df.itertuples(index=False)]
                ax.bar(np.arange(len(plot_df)), plot_df[metric].to_numpy(dtype=float))
                ax.set_xticks(np.arange(len(plot_df)))
                ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=6)
                ax.set_ylabel(ylabel)
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_main_problem_exp2_coverage_conditioned_prediction.pdf")
            fig.savefig(fig_dir / "fig_main_problem_exp2_coverage_conditioned_prediction.png", dpi=220)
            plt.close(fig)

    exp3 = _read_csvs(out_root.glob("*/main_problem_exp3_direct_evidence_removal_summary.csv"))
    if not exp3.empty:
        focus3 = exp3[exp3.get("subgroup", pd.Series([], dtype=str)).eq("paired_bce_increase")].copy()
        if not focus3.empty:
            fig, ax = plt.subplots(figsize=(7.4, 3.2))
            labels = [f"{r.dataset}\n{r.variant}" for r in focus3.itertuples(index=False)]
            x = np.arange(len(focus3))
            ax.bar(x - 0.17, focus3["target_bce_increase"].to_numpy(dtype=float), width=0.34, label="target mask")
            ax.bar(x + 0.17, focus3["random_bce_increase"].to_numpy(dtype=float), width=0.34, label="random mask")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
            ax.set_ylabel("Paired BCE increase")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_main_problem_exp3_direct_evidence_removal.pdf")
            fig.savefig(fig_dir / "fig_main_problem_exp3_direct_evidence_removal.png", dpi=220)
            plt.close(fig)


def _success_exp1(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset, part in summary[summary["group"].eq("direct_unseen_bridgeable")].groupby("dataset"):
        rand = part[part["method"].eq("random")]
        seq = part[part["method"].eq("seq-only")]
        fused = part[part["method"].eq("fused CRG")]
        rand_v = float(rand["hit10"].iloc[0]) if not rand.empty else 0.0
        seq_v = float(seq["hit10"].iloc[0]) if not seq.empty else 0.0
        fused_v = float(fused["hit10"].iloc[0]) if not fused.empty else 0.0
        best_v = max(seq_v, fused_v)
        rows.append(
            {
                "dataset": dataset,
                "exp": "exp1_history_to_query_retrieval",
                "random_hit10": rand_v,
                "seq_hit10": seq_v,
                "fused_hit10": fused_v,
                "success": bool(best_v >= 2.0 * rand_v or best_v - rand_v >= 0.05),
            }
        )
    return pd.DataFrame(rows)


def write_review_packet(out_root: Path, missing: Sequence[str]) -> None:
    def _read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for path in paths:
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    exp1 = _read_csvs(out_root.glob("*/main_problem_exp1_history_to_query_route_summary.csv"))
    exp2 = _read_csvs(out_root.glob("*/main_problem_exp2_coverage_conditioned_metrics.csv"))
    exp3 = _read_csvs(out_root.glob("*/main_problem_exp3_direct_evidence_removal_summary.csv"))
    lines = ["# Main Problem Experiment Review Packet", ""]
    lines.append("This packet uses existing checkpoints only. No retraining or model-structure changes were performed.")
    lines.append("")
    if not exp1.empty:
        success = _success_exp1(exp1)
        lines.extend(["## Experiment 1: History-to-Query Concept Route Retrieval", ""])
        lines.append(success.to_markdown(index=False))
        lines.append("")
    if not exp2.empty:
        focus = exp2[
            exp2["subgroup"].isin(["direct_seen", "direct_unseen_bridgeable", "high_route_mass", "weak_direct_evidence"])
            & exp2["variant"].isin(["full", "no_CRG", "no_LCRF", "self_only", "degree_random_support"])
        ].copy()
        lines.extend(["## Experiment 2: Coverage-conditioned Prediction", ""])
        cols = [c for c in ("dataset", "subgroup", "variant", "n_eval", "auc", "bce", "auc_gap_full_minus_variant", "bce_gap_variant_minus_full") if c in focus.columns]
        lines.append(focus[cols].to_markdown(index=False) if not focus.empty else "No usable prediction metrics were produced.")
        lines.append("")
    if not exp3.empty:
        lines.extend(["## Experiment 3: Direct Concept Evidence Removal Counterfactual", ""])
        lines.append(exp3.to_markdown(index=False) if not exp3.empty else "No usable mask-counterfactual metrics were produced.")
        lines.append("")
    lines.extend(["## Recommendation", ""])
    recommendation = (
        "Use Experiment 1 as the safest main-problem evidence if prediction-level subgroup gaps are weak. "
        "Promote Experiment 2 to the main text only when direct-unseen-bridgeable or high-route-mass gaps meet the pre-defined thresholds. "
        "Use Experiment 3 as a mechanism counterfactual only as a buffer-level state-removal analysis; do not claim exact raw-history recomputation."
    )
    lines.append(recommendation)
    lines.append("")
    lines.extend(["## Missing or downgraded controls", ""])
    if missing:
        for item in sorted(set(missing)):
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    text = "\n".join(lines) + "\n"
    docs_path = ROOT / "docs" / "main_problem_experiment_review_packet.md"
    docs_path.write_text(text, encoding="utf-8")
    (out_root / "main_problem_experiment_review_packet.md").write_text(text, encoding="utf-8")
    if missing:
        (out_root / "missing_report.md").write_text("\n".join(f"- {m}" for m in sorted(set(missing))) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--main-table", default=str(ROOT / "results" / "crg_lcrf_core3_final_20260520" / "main_table" / "table_main_ablation_core3.csv"))
    parser.add_argument("--output-root", default=str(ROOT / "results" / "main_problem_experiments_20260523"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--exp3-max-samples", type=int, default=3000)
    parser.add_argument("--skip-exp3", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Only redraw figures and review packet from existing CSV outputs.")
    args = parser.parse_args()

    out_root = _ensure_dir(Path(args.output_root))
    if args.plot_only:
        missing = [
            "plot-only rerun: reused existing CSV outputs without re-running inference",
            "global_only control was not evaluated unless an existing exact checkpoint was present",
            "exp3 is a sampled buffer-level student-concept state mask diagnostic, not exact raw-history recomputation",
        ]
        _plot_figures(out_root)
        write_review_packet(out_root, missing)
        return
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    main_table = _read_main_table(Path(args.main_table))
    missing: List[str] = []
    manifest_rows: List[Dict[str, Any]] = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        ds_out = _ensure_dir(out_root / dataset)
        assets = _load_assets(dataset, main_table, device, missing)
        if assets is None:
            continue
        exp1_detail, exp1_summary = run_exp1(assets, ds_out, args.seed)
        manifest_rows.append({"dataset": dataset, "experiment": "exp1", "rows": int(len(exp1_detail)), "output": str(ds_out)})
        exp2_pred, exp2_metrics = run_exp2(assets, ds_out, device, args.batch_size, args.num_workers, args.seed, missing)
        manifest_rows.append({"dataset": dataset, "experiment": "exp2", "rows": int(len(exp2_metrics)), "output": str(ds_out)})
        if not args.skip_exp3:
            exp3_sample, exp3_summary = run_exp3(assets, ds_out, device, args.seed, args.exp3_max_samples, missing)
            manifest_rows.append({"dataset": dataset, "experiment": "exp3", "rows": int(len(exp3_sample)), "output": str(ds_out)})
    pd.DataFrame(manifest_rows).to_csv(out_root / "run_manifest.csv", index=False)
    _plot_figures(out_root)
    write_review_packet(out_root, missing)


if __name__ == "__main__":
    main()
