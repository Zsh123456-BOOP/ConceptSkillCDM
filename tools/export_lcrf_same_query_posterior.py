#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export same-query LCRF posterior mechanism cases.

The script is inference-only.  It scans a full checkpoint on a test split, finds
query concepts whose CRG support is identical for many learners, and exports
learner-conditioned posterior distributions over the same support.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CognitiveDiagnosisDataset  # noqa: E402
from tools.export_ae_case_studies import (  # noqa: E402
    _build_model,
    _load_checkpoint,
    _make_eval_frame,
    _resolve_data_dir,
)


EPS = 1e-12


@dataclass
class PosteriorRecord:
    eval_row_id: int
    learner_id: Any
    mapped_student_id: int
    exer_id: Any
    query_item_id: Any
    query_concept_id: int
    support_signature: Tuple[int, ...]
    support_concepts: Tuple[int, ...]
    true_label: float
    pred_full: float
    pred_global: float
    gate_alpha: float
    query_mastery: float
    query_recent_mastery: float
    posterior: np.ndarray
    global_prob: np.ndarray


@dataclass
class CandidateBucket:
    records: List[PosteriorRecord] = field(default_factory=list)
    n_total: int = 0
    learner_ids: set = field(default_factory=set)
    label_counts: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})

    def add(self, record: PosteriorRecord, max_records: int) -> None:
        self.n_total += 1
        self.learner_ids.add(record.learner_id)
        self.label_counts[int(record.true_label >= 0.5)] = self.label_counts.get(int(record.true_label >= 0.5), 0) + 1
        if len(self.records) < max_records:
            self.records.append(record)


@contextmanager
def _disabled_lcrf(model: torch.nn.Module):
    old_model = bool(getattr(model, "use_personal_graph", False))
    old_struct = bool(getattr(model.structure_module, "use_personal_graph", False))
    model.use_personal_graph = False
    model.structure_module.use_personal_graph = False
    try:
        yield
    finally:
        model.use_personal_graph = old_model
        model.structure_module.use_personal_graph = old_struct


def _make_loader(frame: pd.DataFrame, info: Mapping[str, Any], batch_size: int, num_workers: int) -> DataLoader:
    dataset = CognitiveDiagnosisDataset(frame, info["stu_id_map"], info["exer_id_map"], info["cpt_id_map"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _select_query_distribution(
    details: Mapping[str, Any],
    sample_pos: int,
    q_vector: np.ndarray,
) -> Optional[Tuple[int, Tuple[int, ...], Tuple[int, ...], np.ndarray, np.ndarray, float]]:
    active = details.get("active_row_index")
    valid = details.get("active_row_valid_mask")
    support_cols = details.get("support_col_index")
    support_valid = details.get("support_valid_mask")
    global_prob = details.get("global_support_prob")
    posterior_prob = details.get("posterior_prob")
    gate_alpha = details.get("alpha")
    if any(x is None for x in (active, valid, support_cols, support_valid, global_prob, posterior_prob)):
        return None
    active_s = active[sample_pos]
    valid_s = valid[sample_pos].bool()
    query_set = set(int(x) for x in np.where(q_vector > 0)[0].tolist())
    rows = [r for r in range(int(active_s.numel())) if bool(valid_s[r]) and int(active_s[r].item()) in query_set]
    if not rows:
        rows = [r for r in range(int(active_s.numel())) if bool(valid_s[r])]
    if not rows:
        return None
    best_row = rows[0]
    best_score = -1.0
    for r in rows:
        valid_hr = support_valid[sample_pos, :, r, :].float()
        delta = (posterior_prob[sample_pos, :, r, :] - global_prob[sample_pos, :, r, :]).abs() * valid_hr
        score = float(delta.sum().item())
        if score > best_score:
            best_score = score
            best_row = r
    query_c = int(active_s[best_row].item())

    by_col: Dict[int, Dict[str, float]] = {}
    h_count = int(support_cols.size(1))
    k_count = int(support_cols.size(3))
    for h in range(h_count):
        for k in range(k_count):
            if not bool(support_valid[sample_pos, h, best_row, k]):
                continue
            col = int(support_cols[sample_pos, h, best_row, k].item())
            item = by_col.setdefault(col, {"global": 0.0, "post": 0.0, "n": 0.0})
            item["global"] += float(global_prob[sample_pos, h, best_row, k].item())
            item["post"] += float(posterior_prob[sample_pos, h, best_row, k].item())
            item["n"] += 1.0
    if not by_col:
        return None
    support = tuple(sorted(by_col.keys()))
    global_vec = []
    posterior_vec = []
    for col in support:
        n = max(1.0, by_col[col]["n"])
        global_vec.append(by_col[col]["global"] / n)
        posterior_vec.append(by_col[col]["post"] / n)
    global_arr = np.asarray(global_vec, dtype=np.float64)
    post_arr = np.asarray(posterior_vec, dtype=np.float64)
    global_arr = global_arr / max(global_arr.sum(), EPS)
    post_arr = post_arr / max(post_arr.sum(), EPS)
    if gate_alpha is None:
        alpha = 0.0
    else:
        alpha_tensor = gate_alpha[sample_pos]
        alpha = float(alpha_tensor.float().mean().item())
    return query_c, support, support, global_arr, post_arr, alpha


def _select_query_slot_distribution(
    details: Mapping[str, Any],
    sample_pos: int,
    q_vector: np.ndarray,
) -> Optional[Tuple[int, Tuple[int, ...], Tuple[int, ...], np.ndarray, np.ndarray, float]]:
    active = details.get("active_row_index")
    valid = details.get("active_row_valid_mask")
    support_cols = details.get("support_col_index")
    support_valid = details.get("support_valid_mask")
    global_prob = details.get("global_support_prob")
    posterior_prob = details.get("posterior_prob")
    gate_alpha = details.get("alpha")
    if any(x is None for x in (active, valid, support_cols, support_valid, global_prob, posterior_prob)):
        return None
    active_s = active[sample_pos]
    valid_s = valid[sample_pos].bool()
    query_set = set(int(x) for x in np.where(q_vector > 0)[0].tolist())
    rows = [r for r in range(int(active_s.numel())) if bool(valid_s[r]) and int(active_s[r].item()) in query_set]
    if not rows:
        rows = [r for r in range(int(active_s.numel())) if bool(valid_s[r])]
    if not rows:
        return None
    best_row = rows[0]
    best_score = -1.0
    for r in rows:
        valid_hr = support_valid[sample_pos, :, r, :].float()
        delta = (posterior_prob[sample_pos, :, r, :] - global_prob[sample_pos, :, r, :]).abs() * valid_hr
        score = float(delta.sum().item())
        if score > best_score:
            best_score = score
            best_row = r
    query_c = int(active_s[best_row].item())
    mask_t = support_valid[sample_pos, :, best_row, :].reshape(-1).bool()
    if not bool(mask_t.any()):
        return None
    cols_t = support_cols[sample_pos, :, best_row, :].reshape(-1)[mask_t]
    global_t = global_prob[sample_pos, :, best_row, :].reshape(-1)[mask_t]
    post_t = posterior_prob[sample_pos, :, best_row, :].reshape(-1)[mask_t]
    support = tuple(int(i) for i in np.arange(int(cols_t.numel())).tolist())
    support_concepts = tuple(int(x) for x in cols_t.cpu().numpy().astype(np.int64).tolist())
    global_arr = global_t.cpu().numpy().astype(np.float64)
    post_arr = post_t.cpu().numpy().astype(np.float64)
    global_arr = global_arr / max(global_arr.sum(), EPS)
    post_arr = post_arr / max(post_arr.sum(), EPS)
    if gate_alpha is None:
        alpha = 0.0
    else:
        alpha = float(gate_alpha[sample_pos].float().mean().item())
    return query_c, support, support_concepts, global_arr, post_arr, alpha


def _select_item_distribution(
    details: Mapping[str, Any],
    sample_pos: int,
    q_vector: np.ndarray,
) -> Optional[Tuple[int, Tuple[int, ...], Tuple[int, ...], np.ndarray, np.ndarray, float]]:
    active = details.get("active_row_index")
    valid = details.get("active_row_valid_mask")
    support_cols = details.get("support_col_index")
    support_valid = details.get("support_valid_mask")
    global_prob = details.get("global_support_prob")
    posterior_prob = details.get("posterior_prob")
    gate_alpha = details.get("alpha")
    if any(x is None for x in (active, valid, support_cols, support_valid, global_prob, posterior_prob)):
        return None
    active_s = active[sample_pos]
    valid_s = valid[sample_pos].bool()
    query_set = set(int(x) for x in np.where(q_vector > 0)[0].tolist())
    rows = [r for r in range(int(active_s.numel())) if bool(valid_s[r]) and int(active_s[r].item()) in query_set]
    if not rows:
        rows = [r for r in range(int(active_s.numel())) if bool(valid_s[r])]
    if not rows:
        return None

    row_index = torch.as_tensor(rows, dtype=torch.long)
    cols_t = support_cols[sample_pos, :, row_index, :].reshape(-1)
    mask_t = support_valid[sample_pos, :, row_index, :].reshape(-1).bool()
    if not bool(mask_t.any()):
        return None
    cols_np = cols_t[mask_t].cpu().numpy().astype(np.int64)
    global_np = global_prob[sample_pos, :, row_index, :].reshape(-1)[mask_t].cpu().numpy().astype(np.float64)
    post_np = posterior_prob[sample_pos, :, row_index, :].reshape(-1)[mask_t].cpu().numpy().astype(np.float64)
    support_np = np.unique(cols_np)
    max_col = int(support_np.max()) + 1
    counts = np.bincount(cols_np, minlength=max_col).astype(np.float64)
    global_sum = np.bincount(cols_np, weights=global_np, minlength=max_col)
    post_sum = np.bincount(cols_np, weights=post_np, minlength=max_col)
    support = tuple(int(x) for x in support_np.tolist())
    denom = np.maximum(counts[support_np], 1.0)
    global_arr = global_sum[support_np] / denom
    post_arr = post_sum[support_np] / denom
    global_arr = global_arr / max(global_arr.sum(), EPS)
    post_arr = post_arr / max(post_arr.sum(), EPS)
    if gate_alpha is None:
        alpha = 0.0
    else:
        alpha = float(gate_alpha[sample_pos].float().mean().item())
    query_c = int(active_s[rows[0]].item())
    return query_c, support, support, global_arr, post_arr, alpha


def _mean_pairwise_l1(vectors: Sequence[np.ndarray], max_pairs: int = 2500) -> float:
    if len(vectors) < 2:
        return 0.0
    pairs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pairs.append((i, j))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(20260519)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(x)] for x in idx]
    return float(np.mean([np.abs(vectors[i] - vectors[j]).sum() for i, j in pairs]))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), EPS)
    q = q / max(q.sum(), EPS)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log((p + EPS) / (m + EPS))) + 0.5 * np.sum(q * np.log((q + EPS) / (m + EPS))))


def _mean_pairwise_js(vectors: Sequence[np.ndarray], max_pairs: int = 2500) -> float:
    if len(vectors) < 2:
        return 0.0
    pairs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pairs.append((i, j))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(20260519)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(x)] for x in idx]
    return float(np.mean([_js(vectors[i], vectors[j]) for i, j in pairs]))


def _select_learners(records: Sequence[PosteriorRecord], n: int) -> List[PosteriorRecord]:
    if len(records) <= n:
        return list(records)
    arr = np.stack([r.posterior for r in records], axis=0)
    mean = arr.mean(axis=0)
    score = np.abs(arr - mean).sum(axis=1)
    pos = [i for i, r in enumerate(records) if r.true_label >= 0.5]
    neg = [i for i, r in enumerate(records) if r.true_label < 0.5]
    chosen: List[int] = []
    for pool in (pos, neg):
        if pool:
            chosen.append(max(pool, key=lambda idx: score[idx]))
    for idx in np.argsort(-score):
        idx = int(idx)
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= n:
            break
    return [records[i] for i in chosen[:n]]


def run(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = _load_checkpoint(args.full_save_dir, device)
    info = checkpoint["info_dict"]
    dataset_name = args.dataset_name or dict(checkpoint.get("args", {})).get("dataset_name", "assist_09")
    data_dir = _resolve_data_dir(dataset_name, args.data_dir)
    eval_frame = _make_eval_frame(info, data_dir, args.split)
    if args.max_rows and int(args.max_rows) > 0:
        eval_frame = eval_frame.iloc[: int(args.max_rows)].copy()
    loader = _make_loader(eval_frame, info, args.batch_size, args.num_workers)
    model = _build_model(checkpoint, device)
    model.eval()

    prior = getattr(model, "ae_student_concept_prior_logit", None)
    recent = getattr(model, "ae_student_concept_recent_logit", None)
    count = getattr(model, "ae_student_concept_observed_count", None)
    if not torch.is_tensor(prior) or not torch.is_tensor(recent) or not torch.is_tensor(count):
        raise ValueError("Model does not expose learner-concept mastery buffers required for LCRF cases.")
    prior_np = _to_numpy(prior.float())
    recent_np = _to_numpy(recent.float())
    count_np = _to_numpy(count.float())
    candidate_modes = set(str(args.candidate_modes).split(","))

    buckets: DefaultDict[Tuple[str, str, int, Tuple[int, ...]], CandidateBucket] = defaultdict(CandidateBucket)
    offset = 0
    with torch.no_grad():
        for batch_idx, (student_ids, exercise_ids, labels) in enumerate(loader):
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            labels = labels.to(device)
            logits_full, details = model(student_ids, exercise_ids, return_details=True, return_logits=True)
            probs_full = torch.sigmoid(logits_full.reshape(-1))
            with _disabled_lcrf(model):
                logits_global = model(student_ids, exercise_ids, return_details=False, return_logits=True)
            probs_global = torch.sigmoid(logits_global.reshape(-1))
            details_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in details.items()}
            q_np = _to_numpy(details_cpu["q_vector"].float())
            stu_np = _to_numpy(student_ids)
            label_np = _to_numpy(labels.float()).reshape(-1)
            full_np = _to_numpy(probs_full.float())
            global_np = _to_numpy(probs_global.float())
            batch_frame = eval_frame.iloc[offset : offset + len(label_np)].reset_index(drop=True)
            for i in range(len(label_np)):
                learner_raw = batch_frame.loc[i, "stu_id"]
                exer_raw = batch_frame.loc[i, "exer_id"]
                mapped_student = int(stu_np[i])
                candidate_specs = []
                if "concept" in candidate_modes:
                    concept_dist = _select_query_distribution(details_cpu, i, q_np[i])
                    if concept_dist is not None:
                        candidate_specs.append(("same_query_concept", str(int(concept_dist[0])), concept_dist))
                if "concept_slot" in candidate_modes:
                    concept_slot_dist = _select_query_slot_distribution(details_cpu, i, q_np[i])
                    if concept_slot_dist is not None:
                        candidate_specs.append(("same_query_concept_slot", str(int(concept_slot_dist[0])), concept_slot_dist))
                if "item" in candidate_modes:
                    item_dist = _select_item_distribution(details_cpu, i, q_np[i])
                    if item_dist is not None:
                        candidate_specs.append(("same_query_item", str(exer_raw), item_dist))
                for candidate_type, query_key, dist in candidate_specs:
                    query_c, support, support_concepts, global_vec, posterior_vec, alpha = dist
                    if len(support) < int(args.min_support_size):
                        continue
                    rec = PosteriorRecord(
                        eval_row_id=int(batch_frame.loc[i, "eval_row_id"]),
                        learner_id=learner_raw,
                        mapped_student_id=mapped_student,
                        exer_id=exer_raw,
                        query_item_id=exer_raw,
                        query_concept_id=int(query_c),
                        support_signature=tuple(int(x) for x in support),
                        support_concepts=tuple(int(x) for x in support_concepts),
                        true_label=float(label_np[i]),
                        pred_full=float(full_np[i]),
                        pred_global=float(global_np[i]),
                        gate_alpha=float(alpha),
                        query_mastery=float(prior_np[mapped_student, query_c]),
                        query_recent_mastery=float(recent_np[mapped_student, query_c]),
                        posterior=posterior_vec.astype(np.float64),
                        global_prob=global_vec.astype(np.float64),
                    )
                    buckets[(candidate_type, query_key, int(query_c), tuple(int(x) for x in support))].add(rec, int(args.max_records_per_bucket))
            offset += len(label_np)
            if args.progress_every > 0 and (batch_idx + 1) % int(args.progress_every) == 0:
                print(f"[{dataset_name}] scanned rows={offset}, buckets={len(buckets)}")

    cheap_candidates: List[Dict[str, Any]] = []
    for (candidate_type, query_key, query_c, support), bucket in buckets.items():
        if len(bucket.learner_ids) < int(args.min_learners) or len(bucket.records) < 2:
            continue
        arr = np.stack([r.posterior for r in bucket.records], axis=0)
        has_both = int(bucket.label_counts.get(0, 0) > 0 and bucket.label_counts.get(1, 0) > 0)
        cheap_variability = float(np.abs(arr - arr.mean(axis=0, keepdims=True)).sum(axis=1).mean())
        cheap_candidates.append(
            {
                "candidate_type": candidate_type,
                "query_key": query_key,
                "query_c": int(query_c),
                "support": tuple(int(x) for x in support),
                "bucket": bucket,
                "has_both": has_both,
                "cheap_variability": cheap_variability,
                "cheap_score": float((2.0 if has_both else 0.0) + cheap_variability + math.log1p(len(bucket.learner_ids)) / 10.0),
            }
        )

    cheap_candidates = sorted(
        cheap_candidates,
        key=lambda x: (int(x["has_both"]), float(x["cheap_score"])),
        reverse=True,
    )[: int(args.max_summary_buckets)]

    summary_rows: List[Dict[str, Any]] = []
    for item in cheap_candidates:
        candidate_type = str(item["candidate_type"])
        query_key = str(item["query_key"])
        query_c = int(item["query_c"])
        support = tuple(int(x) for x in item["support"])
        bucket = item["bucket"]
        vectors = [r.posterior for r in bucket.records]
        l1 = _mean_pairwise_l1(vectors)
        js = _mean_pairwise_js(vectors)
        has_both = int(item["has_both"])
        summary_rows.append(
            {
                "dataset": dataset_name,
                "candidate_type": candidate_type,
                "query_key": query_key,
                "query_item_id": query_key if candidate_type == "same_query_item" else "",
                "query_concept_id": int(query_c),
                "support_signature": ";".join(str(x) for x in support),
                "support_concepts_signature": ";".join(str(x) for x in bucket.records[0].support_concepts),
                "support_size": int(len(support)),
                "n_total": int(bucket.n_total),
                "n_records_sampled": int(len(bucket.records)),
                "n_learners": int(len(bucket.learner_ids)),
                "n_positive": int(bucket.label_counts.get(1, 0)),
                "n_negative": int(bucket.label_counts.get(0, 0)),
                "has_both_labels": bool(has_both),
                "support_identical_check_passed": True,
                "mean_pairwise_l1": l1,
                "mean_pairwise_js": js,
                "cheap_variability": float(item["cheap_variability"]),
                "posterior_variability_check_passed": bool(l1 >= float(args.min_l1) or js >= float(args.min_js)),
                "score": float((2.0 if has_both else 0.0) + l1 + 4.0 * js + math.log1p(len(bucket.learner_ids)) / 10.0),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["posterior_variability_check_passed", "has_both_labels", "score"],
        ascending=[False, False, False],
    )
    summary_csv = out_dir / f"{dataset_name}_same_query_candidate_summary.csv"
    summary.to_csv(summary_csv, index=False)

    top_rows: List[Dict[str, Any]] = []
    if not summary.empty:
        best = summary.iloc[0]
        candidate_type = str(best["candidate_type"])
        query_key = str(best["query_key"])
        query_c = int(best["query_concept_id"])
        support = tuple(int(x) for x in str(best["support_signature"]).split(";") if str(x) != "")
        bucket = buckets[(candidate_type, query_key, query_c, support)]
        selected = _select_learners(bucket.records, int(args.case_learners))
        case_prefix = "I" if candidate_type == "same_query_item" else "Q"
        safe_query_key = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in query_key)
        case_id = f"{dataset_name}_{case_prefix}{safe_query_key}_S{len(support)}"
        for learner_rank, rec in enumerate(selected, start=1):
            for support_pos, (support_slot, support_c) in enumerate(zip(rec.support_signature, rec.support_concepts), start=1):
                idx = support_pos - 1
                top_rows.append(
                    {
                        "dataset": dataset_name,
                        "case_id": case_id,
                        "learner_rank": int(learner_rank),
                        "support_col_index": int(support_pos),
                        "support_slot_id": int(support_slot),
                        "support_valid_mask": True,
                        "global_support_prob": float(rec.global_prob[idx]),
                        "posterior_prob": float(rec.posterior[idx]),
                        "posterior_minus_global": float(rec.posterior[idx] - rec.global_prob[idx]),
                        "gate_alpha": float(rec.gate_alpha),
                        "learner_id": rec.learner_id,
                        "mapped_student_id": int(rec.mapped_student_id),
                        "eval_row_id": int(rec.eval_row_id),
                        "candidate_type": candidate_type,
                        "query_item_id": rec.query_item_id,
                        "query_concept_id": int(rec.query_concept_id),
                        "support_concept_id": int(support_c),
                        "query_mastery": float(rec.query_mastery),
                        "query_recent_mastery": float(rec.query_recent_mastery),
                        "support_mastery": float(prior_np[rec.mapped_student_id, support_c]),
                        "support_recent_mastery": float(recent_np[rec.mapped_student_id, support_c]),
                        "support_count": float(count_np[rec.mapped_student_id, support_c]),
                        "pred_global": float(rec.pred_global),
                        "pred_full": float(rec.pred_full),
                        "true_label": float(rec.true_label),
                        "support_identical_check_passed": True,
                        "mean_pairwise_l1": float(best["mean_pairwise_l1"]),
                        "mean_pairwise_js": float(best["mean_pairwise_js"]),
                    }
                )
    topk = pd.DataFrame(top_rows)
    topk_csv = out_dir / f"{dataset_name}_same_query_learner_posterior_topk.csv"
    topk.to_csv(topk_csv, index=False)
    print(json.dumps({"summary": str(summary_csv), "topk": str(topk_csv), "summary_rows": len(summary), "topk_rows": len(topk)}, ensure_ascii=False, indent=2))
    return summary, topk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--full_save_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--min_learners", type=int, default=20)
    parser.add_argument("--min_support_size", type=int, default=4)
    parser.add_argument("--max_records_per_bucket", type=int, default=320)
    parser.add_argument("--max_summary_buckets", type=int, default=240)
    parser.add_argument("--case_learners", type=int, default=8)
    parser.add_argument("--min_l1", type=float, default=0.10)
    parser.add_argument("--min_js", type=float, default=0.03)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--candidate_modes", default="concept,item", help="Comma-separated modes: concept,item,concept_slot.")
    parser.add_argument("--progress_every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
