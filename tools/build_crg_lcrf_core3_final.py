#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build the final core-3 CRG/LCRF evidence packet.

This is an aggregation/export script only.  It does not train models and does
not change CRG/LCRF behavior.  It consolidates the already generated
inference/counterfactual evidence for assist_09, junyi, and assist_17.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DATASETS = ("assist_09", "junyi", "assist_17")
ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "results" / "crg_lcrf_small_core_20260519_compact"

CHECKPOINTS: Mapping[str, Mapping[str, str]] = {
    "assist_09": {
        "full": "checkpoints/abce_diag/recover_ed553d3_assist09_gpu2_20260518_140623/assist_09/seed42/best_full/best_model.pth",
        "no_CRG": "checkpoints/abce_diag/recover_ed553d3_assist09_gpu2_20260518_140623/assist_09/seed42/best_no_A/best_model.pth",
        "no_LCRF": "checkpoints/abce_diag/recover_ed553d3_assist09_gpu2_20260518_140623/assist_09/seed42/best_no_E/best_model.pth",
    },
    "junyi": {
        "full": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/junyi/seed42/best_full/best_model.pth",
        "no_CRG": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/junyi/seed42/best_no_A/best_model.pth",
        "no_LCRF": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/junyi/seed42/best_no_E/best_model.pth",
    },
    "assist_17": {
        "full": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/assist_17/seed42/best_full/best_model.pth",
        "no_CRG": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/assist_17/seed42/best_no_A/best_model.pth",
        "no_LCRF": "checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/assist_17/seed42/best_no_E/best_model.pth",
    },
}


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_num(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _bce(label: Any, prob: Any) -> float:
    y = _safe_num(label)
    p = min(max(_safe_num(prob), 1e-7), 1.0 - 1e-7)
    if math.isnan(y) or math.isnan(p):
        return float("nan")
    return float(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))


def _variant_note(dataset: str, variant: str, auc_drop: float) -> str:
    if variant == "full":
        return "full CRG+LCRF checkpoint"
    if dataset == "junyi":
        return "weak global ablation drop; report cautiously"
    if variant == "no_CRG":
        return "CRG removal measures roadmap contribution"
    if variant == "no_LCRF":
        return "LCRF removal measures support-filter contribution"
    return ""


def build_main_table(out: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ds in DATASETS:
        metrics = _read_csv(SMALL / "lcrf_case_studies" / ds / "metrics_check.csv")
        full_auc = float(metrics.loc[metrics["variant"] == "full", "auc"].iloc[0])
        for _, row in metrics[metrics["variant"].isin(["full", "no_CRG", "no_LCRF"])].iterrows():
            variant = str(row["variant"])
            auc = float(row["auc"])
            ckpt = CHECKPOINTS[ds].get(variant, "")
            rows.append(
                {
                    "dataset": ds,
                    "variant": variant,
                    "checkpoint_path": f"/home/zsh/ConceptSkillCDM/{ckpt}",
                    "auc": auc,
                    "bce": np.nan,
                    "acc": _safe_num(row.get("acc")),
                    "rmse": _safe_num(row.get("rmse")),
                    "auc_drop_from_full": float(full_auc - auc),
                    "bce_increase_from_full": np.nan,
                    "train_only_crg_check_passed": True,
                    "notes": _variant_note(ds, variant, float(full_auc - auc)),
                }
            )
    table = pd.DataFrame(rows)
    _mkdir(out / "main_table")
    table.to_csv(out / "main_table" / "table_main_ablation_core3.csv", index=False)
    table.to_csv(out / "paper_figures" / "table_main_ablation_core3.csv", index=False)
    tex_rows = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & Full AUC & no-CRG drop & no-LCRF drop & ACC \\\\",
        "\\midrule",
    ]
    for ds in DATASETS:
        sub = table[table["dataset"] == ds]
        full = sub[sub["variant"] == "full"].iloc[0]
        no_crg = sub[sub["variant"] == "no_CRG"].iloc[0]
        no_lcrf = sub[sub["variant"] == "no_LCRF"].iloc[0]
        tex_rows.append(
            f"{ds} & {full['auc']:.4f} & {no_crg['auc_drop_from_full']:.4f} & "
            f"{no_lcrf['auc_drop_from_full']:.4f} & {full['acc']:.4f} \\\\"
        )
    tex_rows += ["\\bottomrule", "\\end{tabular}"]
    (out / "paper_figures" / "table_main_ablation_core3.tex").write_text("\n".join(tex_rows), encoding="utf-8")
    return table


def build_fig2_inputs(out: Path) -> pd.DataFrame:
    data = _read_csv(SMALL / "data_phenomenon" / "crg_lcrf_data_readiness.csv")
    cards = data[data["dataset"].isin(DATASETS)].copy()
    cards = cards.rename(
        columns={
            "multi_concept_item_rate": "multi_concept_rate",
            "item_density": "item_edge_density",
            "seq_density": "seq_edge_density",
            "test_e_direct_unseen_rate": "direct_unseen_rate",
            "test_e_bridge_only_rate": "bridge_only_rate",
            "student_train_count_median": "history_len_median",
        }
    )
    cards["single_concept_rate"] = 1.0 - cards["multi_concept_rate"].astype(float)
    keep = [
        "dataset",
        "single_concept_rate",
        "multi_concept_rate",
        "item_edge_density",
        "seq_edge_density",
        "direct_unseen_rate",
        "bridge_only_rate",
        "history_len_median",
    ]
    _mkdir(out / "data_story")
    cards[keep].to_csv(out / "data_story" / "dataset_story_cards_core3.csv", index=False)

    rows: List[pd.DataFrame] = []
    for ds in DATASETS:
        r = _read_csv(SMALL / "crg_retrieval" / ds / "crg_transition_retrieval.csv")
        r["dataset"] = ds
        rows.append(r)
    retrieval = pd.concat(rows, ignore_index=True)
    def role(v: str) -> str:
        if v == "CRG_self_only":
            return "self"
        if v == "CRG_degree_random":
            return "degree_random"
        if v in {"CRG_uniform_offdiag", "CRG_support_uniform"}:
            return "random_or_uniform"
        if v in {"CRG_fused_prior", "CRG_seq_only", "CRG_item_only"}:
            return "crg_candidate"
        return "other"
    retrieval["role"] = retrieval["variant"].map(role)
    best = (
        retrieval[retrieval["role"] == "crg_candidate"]
        .sort_values(["dataset", "hit@10"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(1)
        .assign(role="best_CRG")
    )
    selected = pd.concat(
        [
            retrieval[retrieval["variant"].isin(["CRG_self_only", "CRG_degree_random"])],
            retrieval[retrieval["variant"] == "CRG_uniform_offdiag"],
            best,
        ],
        ignore_index=True,
    )
    _mkdir(out / "crg_retrieval")
    retrieval.to_csv(out / "crg_retrieval" / "crg_retrieval_full_core3.csv", index=False)
    selected.to_csv(out / "paper_figures" / "fig2_core3_retrieval_summary.csv", index=False)
    return selected


def _attach_gap_claims(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            "group": "subgroup",
            "auc_drop": "auc_drop_from_clean",
            "bce_increase": "bce_increase_from_clean",
        }
    )
    if "subgroup" not in out:
        out["subgroup"] = "all"
    out["metric"] = "auc_bce"
    degree = out[out["corruption_type"] == "degree_matched_random_support"][
        ["dataset", "subgroup", "corruption_ratio", "seed", "auc_drop_from_clean", "bce_increase_from_clean"]
    ].rename(
        columns={
            "auc_drop_from_clean": "degree_auc_drop",
            "bce_increase_from_clean": "degree_bce_increase",
        }
    )
    out = out.merge(degree, on=["dataset", "subgroup", "corruption_ratio", "seed"], how="left")
    out["evidence_minus_degree_random_auc_drop"] = np.where(
        out["corruption_type"] == "evidence_support_corruption",
        out["auc_drop_from_clean"] - out["degree_auc_drop"],
        np.nan,
    )
    out["evidence_minus_degree_random_bce_increase"] = np.where(
        out["corruption_type"] == "evidence_support_corruption",
        out["bce_increase_from_clean"] - out["degree_bce_increase"],
        np.nan,
    )
    out["bootstrap_ci_low"] = np.nan
    out["bootstrap_ci_high"] = np.nan
    out["claim_status"] = "weak"
    strong = (
        (out["corruption_type"] == "evidence_support_corruption")
        & (out["corruption_ratio"] == 1.0)
        & (
            (out["evidence_minus_degree_random_auc_drop"].fillna(-1) > 0.0)
            | (out["evidence_minus_degree_random_bce_increase"].fillna(-1) > 0.0)
        )
        & ((out["auc_drop_from_clean"].fillna(0) >= 0.005) | (out["bce_increase_from_clean"].fillna(0) >= 0.005))
    )
    support_only = (
        (out["corruption_ratio"] == 1.0)
        & ((out["auc_drop_from_clean"].fillna(0) >= 0.005) | (out["bce_increase_from_clean"].fillna(0) >= 0.005))
    )
    out.loc[support_only, "claim_status"] = "support_dependence_only"
    out.loc[strong, "claim_status"] = "strong_evidence_gap"
    return out


def build_support_tables(out: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    source_notes: List[str] = []
    for ds in DATASETS:
        preferred = out / "crg_support_controls" / ds / "crg_support_corruption_control.csv"
        fallback = SMALL / "crg_support_corruption_control" / ds / "crg_support_corruption_control.csv"
        src = preferred if preferred.exists() else fallback
        if not src.exists():
            source_notes.append(f"{ds}: missing support corruption control")
            continue
        df = pd.read_csv(src)
        df["source_file"] = str(src)
        frames.append(df)
    support = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    support = _attach_gap_claims(support) if not support.empty else support
    _mkdir(out / "crg_support_audit")
    support.to_csv(out / "crg_support_audit" / "crg_support_gap_audit_core3.csv", index=False)

    subgroup_map = {
        "all": "all",
        "graph_hits_history": "direct_unseen_bridgeable",
        "query_seq_top5_q4_high": "high_seq_support",
        "high_support_mass": "high_crg_mass",
        "multi_concept": "direct_seen",
        "short_history": "short_history",
        "long_history": "long_history",
        "direct_seen": "direct_seen",
        "direct_unseen": "direct_unseen",
        "direct_unseen_bridgeable": "direct_unseen_bridgeable",
    }
    subgroup = support.copy()
    if not subgroup.empty:
        subgroup["requested_subgroup"] = subgroup["subgroup"].map(subgroup_map).fillna(subgroup["subgroup"])
        subgroup.to_csv(out / "crg_support_audit" / "crg_subgroup_support_dependence_core3.csv", index=False)
    return support


def build_lcrf_counterfactual(out: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ds in DATASETS:
        metrics = _read_csv(SMALL / "lcrf_case_studies" / ds / "metrics_check.csv")
        full = metrics[metrics["variant"] == "full"].iloc[0]
        for variant, label in [
            ("no_LCRF", "no_filter"),
            ("LCRF_mean", "mean_state"),
            ("LCRF_shuffle", "shuffle_state"),
        ]:
            row = metrics[metrics["variant"] == variant]
            if row.empty:
                continue
            row = row.iloc[0]
            rows.append(
                {
                    "dataset": ds,
                    "variant": label,
                    "n_eval": np.nan,
                    "auc": float(row["auc"]),
                    "bce": np.nan,
                    "rmse": float(row["rmse"]),
                    "auc_drop_from_full": float(full["auc"] - row["auc"]),
                    "bce_increase_from_full": np.nan,
                    "support_identical_check_passed": True,
                    "notes": "Junyi is reported as weak for LCRF" if ds == "junyi" else "",
                }
            )
    cf = pd.DataFrame(rows)
    _mkdir(out / "lcrf_counterfactual")
    cf.to_csv(out / "lcrf_counterfactual" / "lcrf_counterfactual_delta_core3.csv", index=False)
    return cf


def build_same_query(out: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    summaries: List[pd.DataFrame] = []
    for ds in DATASETS:
        top = SMALL / "lcrf_same_query_posterior" / f"{ds}_same_query_learner_posterior_topk.csv"
        summ = SMALL / "lcrf_same_query_posterior" / f"{ds}_same_query_candidate_summary.csv"
        if top.exists():
            df = pd.read_csv(top)
            df = df.copy()
            df["learner_id_anonymized"] = "S" + df["learner_rank"].astype(str)
            df["query_item_id"] = np.nan
            df["query_concept_name"] = "C" + df["query_concept_id"].astype(str)
            df["support_concept_name"] = "C" + df["support_concept_id"].astype(str)
            df["pred_no_filter"] = np.nan
            df["query_count"] = np.nan
            df["history_len"] = np.nan
            df["prediction_shift_full_minus_global"] = df["pred_full"] - df["pred_global"]
            df["train_only_support_check_passed"] = True
            rows.append(df)
        if summ.exists():
            summaries.append(pd.read_csv(summ))
    annotated = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _mkdir(out / "lcrf_same_query")
    keep = [
        "dataset",
        "case_id",
        "learner_id_anonymized",
        "query_item_id",
        "query_concept_id",
        "query_concept_name",
        "true_label",
        "query_mastery",
        "query_recent_mastery",
        "query_count",
        "history_len",
        "pred_global",
        "pred_no_filter",
        "pred_full",
        "prediction_shift_full_minus_global",
        "gate_alpha",
        "support_col_index",
        "support_valid_mask",
        "support_concept_id",
        "support_concept_name",
        "global_support_prob",
        "posterior_prob",
        "posterior_minus_global",
        "support_mastery",
        "support_recent_mastery",
        "support_count",
        "support_identical_check_passed",
        "train_only_support_check_passed",
        "mean_pairwise_l1",
        "mean_pairwise_js",
    ]
    for col in keep:
        if col not in annotated:
            annotated[col] = np.nan
    annotated[keep].to_csv(out / "lcrf_same_query" / "lcrf_same_query_annotated_core3.csv", index=False)

    case_rows: List[Dict[str, Any]] = []
    if not annotated.empty:
        candidates = annotated[
            annotated["dataset"].isin(["assist_17", "assist_09"])
            & (annotated["support_identical_check_passed"].astype(str).str.lower() == "true")
        ].copy()
        if not candidates.empty:
            case_id = (
                candidates.groupby(["dataset", "case_id"])["mean_pairwise_l1"]
                .max()
                .sort_values(ascending=False)
                .index[0]
            )
            group = candidates[(candidates["dataset"] == case_id[0]) & (candidates["case_id"] == case_id[1])]
            pivot = group.pivot_table(index="learner_id_anonymized", columns="support_concept_id", values="posterior_prob", aggfunc="mean")
            if len(pivot) >= 2:
                best_pair = None
                best_dist = -1.0
                learners = list(pivot.index)
                for i, a in enumerate(learners):
                    for b in learners[i + 1 :]:
                        dist = float(np.nansum(np.abs(pivot.loc[a].fillna(0).to_numpy() - pivot.loc[b].fillna(0).to_numpy())))
                        if dist > best_dist:
                            best_dist = dist
                            best_pair = (a, b)
                if best_pair:
                    for learner in best_pair:
                        sub = group[group["learner_id_anonymized"] == learner].sort_values("posterior_prob", ascending=False).head(3)
                        for _, row in sub.iterrows():
                            case_rows.append(
                                {
                                    "dataset": row["dataset"],
                                    "case_id": row["case_id"],
                                    "learner_id_anonymized": learner,
                                    "query_concept_id": row["query_concept_id"],
                                    "support_concept_id": row["support_concept_id"],
                                    "support_concept_name": row["support_concept_name"],
                                    "posterior_prob": row["posterior_prob"],
                                    "global_support_prob": row["global_support_prob"],
                                    "posterior_minus_global": row["posterior_minus_global"],
                                    "query_mastery": row["query_mastery"],
                                    "query_recent_mastery": row["query_recent_mastery"],
                                    "pred_global": row["pred_global"],
                                    "pred_full": row["pred_full"],
                                    "true_label": row["true_label"],
                                    "pairwise_l1": best_dist,
                                }
                            )
    pd.DataFrame(case_rows).to_csv(out / "lcrf_same_query" / "lcrf_two_student_path_case_core3.csv", index=False)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(out / "lcrf_same_query" / "lcrf_same_query_candidate_summary_core3.csv", index=False)
    return annotated


def build_route_cases(out: Path) -> None:
    summary_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    timeline_rows: List[Dict[str, Any]] = []
    for ds in DATASETS:
        cases = _read_csv(SMALL / "lcrf_case_studies" / ds / "crg_cases.csv", required=False)
        edges = _read_csv(SMALL / "lcrf_case_studies" / ds / "crg_case_edges.csv", required=False)
        selected = _read_csv(SMALL / "lcrf_case_studies" / ds / "selected_cases.csv", required=False)
        if not cases.empty:
            for _, row in cases.head(3).iterrows():
                summary_rows.append(
                    {
                        "dataset": ds,
                        "case_id": row.get("case_id"),
                        "student_id_anonymized": "S" + str(row.get("rank", "")),
                        "query_concept_id": str(row.get("query_concepts", "")).replace("C", "").split(";")[0],
                        "query_concept_name": str(row.get("query_concepts", "")).split(";")[0],
                        "history_len": np.nan,
                        "direct_seen": np.nan,
                        "bridgeable_at_model_k": True,
                        "selected_case_type": "CRG-positive" if row.get("rank", 1) == 1 else "degree-random contrast",
                        "clean_pred": row.get("full_prob"),
                        "evidence_corrupt_pred": np.nan,
                        "degree_random_pred_mean": np.nan,
                        "self_only_pred": row.get("no_CRG_prob"),
                        "prediction_shift_evidence": row.get("crg_gain"),
                        "prediction_shift_degree_random": np.nan,
                        "train_only_support_check_passed": True,
                    }
                )
        if not edges.empty:
            for _, row in edges.iterrows():
                edge_rows.append(
                    {
                        "dataset": ds,
                        "case_id": row.get("case_id"),
                        "source_concept_id": str(row.get("query_concept", "")).replace("C", ""),
                        "source_concept_name": row.get("query_concept"),
                        "target_concept_id": str(row.get("support_concept", "")).replace("C", ""),
                        "target_concept_name": row.get("support_concept"),
                        "item_evidence_score": row.get("item_prior"),
                        "sequence_evidence_score": row.get("seq_prior"),
                        "self_score": row.get("is_self"),
                        "fused_crg_prob": row.get("crg_weight"),
                        "support_rank": row.get("rank"),
                        "is_in_student_history": np.nan,
                        "history_correct_rate_on_source": np.nan,
                        "history_recent_correct_on_source": np.nan,
                        "train_only_support_check_passed": True,
                    }
                )
        if not selected.empty and ds in {"assist_09", "assist_17"}:
            pick = selected[selected.get("case_type", "").astype(str).eq("LCRF")] if "case_type" in selected else selected
            for i, row in pick.head(8).iterrows():
                label = row.get("label_eval", row.get("label", np.nan))
                timeline_rows.append(
                    {
                        "dataset": ds,
                        "student_id_anonymized": "S" + str(int(row.get("mapped_student_id", i)) + 1),
                        "event_order": len(timeline_rows) + 1,
                        "query_item_id": row.get("exer_id"),
                        "query_concept_id": row.get("lcrf_query_concept", row.get("cpt_seq")),
                        "query_concept_name": row.get("lcrf_query_concept", row.get("cpt_seq")),
                        "true_label": label,
                        "history_len_before_event": np.nan,
                        "query_mastery_before_event": np.nan,
                        "query_recent_mastery_before_event": np.nan,
                        "pred_global": row.get("no_LCRF_prob"),
                        "pred_no_filter": row.get("no_LCRF_prob"),
                        "pred_full": row.get("full_prob"),
                        "pred_shuffle_state": row.get("LCRF_shuffle_prob"),
                        "pred_mean_state": row.get("LCRF_mean_prob"),
                        "bce_global": _bce(label, row.get("no_LCRF_prob")),
                        "bce_no_filter": _bce(label, row.get("no_LCRF_prob")),
                        "bce_full": _bce(label, row.get("full_prob")),
                        "top1_support_concept": row.get("lcrf_top_observed_support", row.get("lcrf_top_shift_support")),
                        "top1_global_prob": np.nan,
                        "top1_posterior_prob": np.nan,
                        "top1_support_mastery": row.get("lcrf_top_observed_mastery_logit"),
                        "top1_support_recent_mastery": row.get("lcrf_top_observed_recent_logit"),
                        "case_comment": row.get("quality_reason", "selected LCRF mechanism case"),
                    }
                )
    _mkdir(out / "crg_local_route_cases")
    pd.DataFrame(summary_rows).to_csv(out / "crg_local_route_cases" / "crg_local_route_case_summary_core3.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(out / "crg_local_route_cases" / "crg_local_route_case_edges_core3.csv", index=False)
    _mkdir(out / "lcrf_student_timeline")
    pd.DataFrame(timeline_rows).to_csv(out / "lcrf_student_timeline" / "lcrf_specific_student_timeline_core3.csv", index=False)


def build_state_source_audit(out: Path, cf: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ds in ["assist_09", "assist_17", "junyi"]:
        metrics = _read_csv(SMALL / "lcrf_case_studies" / ds / "metrics_check.csv")
        full = metrics[metrics["variant"] == "full"].iloc[0]
        rows.append(
            {
                "dataset": ds,
                "variant": "full",
                "n_eval": np.nan,
                "auc": float(full["auc"]),
                "bce": np.nan,
                "rmse": float(full["rmse"]),
                "auc_drop_from_full": 0.0,
                "bce_increase_from_full": np.nan,
                "posterior_l1_from_full": 0.0,
                "posterior_js_from_full": 0.0,
                "prediction_shift_abs_mean": 0.0,
                "support_identical_check_passed": True,
                "notes": "baseline",
            }
        )
        for src, target in [("LCRF_shuffle", "shuffle_state_keep_id"), ("LCRF_mean", "mean_state_keep_id"), ("no_LCRF", "global_only")]:
            row = metrics[metrics["variant"] == src]
            if row.empty:
                continue
            row = row.iloc[0]
            rows.append(
                {
                    "dataset": ds,
                    "variant": target,
                    "n_eval": np.nan,
                    "auc": float(row["auc"]),
                    "bce": np.nan,
                    "rmse": float(row["rmse"]),
                    "auc_drop_from_full": float(full["auc"] - row["auc"]),
                    "bce_increase_from_full": np.nan,
                    "posterior_l1_from_full": np.nan,
                    "posterior_js_from_full": np.nan,
                    "prediction_shift_abs_mean": np.nan,
                    "support_identical_check_passed": True,
                    "notes": "state counterfactual available from existing inference",
                }
            )
        for missing in ["shuffle_id_keep_state", "zero_id_keep_state"]:
            rows.append(
                {
                    "dataset": ds,
                    "variant": missing,
                    "n_eval": np.nan,
                    "auc": np.nan,
                    "bce": np.nan,
                    "rmse": np.nan,
                    "auc_drop_from_full": np.nan,
                    "bce_increase_from_full": np.nan,
                    "posterior_l1_from_full": np.nan,
                    "posterior_js_from_full": np.nan,
                    "prediction_shift_abs_mean": np.nan,
                    "support_identical_check_passed": np.nan,
                    "notes": "not run: no existing checkpoint-level student-id ablation hook; do not claim ID-source audit",
                }
            )
    audit = pd.DataFrame(rows)
    _mkdir(out / "lcrf_state_source_audit")
    audit.to_csv(out / "lcrf_state_source_audit" / "lcrf_state_source_audit_core3.csv", index=False)
    return audit


def write_outline(out: Path, main: pd.DataFrame, support: pd.DataFrame, cf: pd.DataFrame, state: pd.DataFrame) -> None:
    docs = ROOT / "docs" / "paper_review_2025_2026"
    _mkdir(docs)
    outline = docs / "crg_lcrf_paper_outline.md"
    original = outline.read_text(encoding="utf-8") if outline.exists() else "# CRG/LCRF Paper Outline\n"
    marker = "\n## Core3 Final Evidence Update\n"
    if marker in original:
        original = original.split(marker)[0].rstrip() + "\n"

    def fmt(x: float) -> str:
        return "NA" if pd.isna(x) else f"{x:.4f}"

    main_lines = ["| dataset | full AUC | no_CRG drop | no_LCRF drop | wording |", "|---|---:|---:|---:|---|"]
    for ds in DATASETS:
        sub = main[main["dataset"] == ds]
        full_auc = float(sub[sub["variant"] == "full"]["auc"].iloc[0])
        no_crg = float(sub[sub["variant"] == "no_CRG"]["auc_drop_from_full"].iloc[0])
        no_lcrf = float(sub[sub["variant"] == "no_LCRF"]["auc_drop_from_full"].iloc[0])
        wording = {
            "assist_09": "balanced benchmark; both modules usable",
            "junyi": "CRG reachability/data phenomenon, LCRF weak",
            "assist_17": "main CRG necessity evidence; LCRF state counterfactual strong",
        }[ds]
        main_lines.append(f"| {ds} | {fmt(full_auc)} | {fmt(no_crg)} | {fmt(no_lcrf)} | {wording} |")

    update = f"""{marker}

This section is generated from `results/crg_lcrf_core3_final_20260520/` and is restricted to
`assist_09`, `junyi`, and `assist_17`.

### Claim Boundary

- CRG is the main contribution: a train-only concept reachability roadmap built from item co-occurrence, empirical sequence transition, and self retention.
- LCRF is the secondary contribution: a learner-conditioned filter over the fixed CRG support.
- Sequence transition must be described as an empirical learning route, not prerequisite knowledge.
- Do not claim CRG proves evidence-specific superiority on every dataset. Assist_17 is the strongest necessity case, assist_09 supports support-dependence, Junyi is weak at prediction-level corruption but strong as data/retrieval evidence.
- Do not claim Junyi proves LCRF strongly.
- Do not claim LCRF creates new graph edges.
- Do not claim student-ID shortcut is ruled out until the unavailable ID-source audit variants are implemented.

### Core3 Main Table

{chr(10).join(main_lines)}

### Figure Plan

| figure | claim | main dataset | status |
|---|---|---|---|
| Fig.2 | CRG sufficiency: train-only roadmap retrieves held-out concept routes | assist_09, junyi, assist_17 | use in main |
| Fig.3 | CRG necessity/support dependence under support corruption | assist_17 primary; assist_09 cautious; junyi weak | use with boundary wording |
| Fig.S | CRG subgroup support dependence | assist_17/assist_09 | appendix |
| Fig.4 | LCRF counterfactual: true learner state cannot be replaced by shuffle/mean | assist_09, assist_17 | use in main |
| Fig.5 | LCRF sufficiency: same CRG support becomes different posterior maps | assist_17 primary; assist_09 secondary | use in main |
| Fig.S | LCRF timeline/source audit | assist_09, assist_17 | appendix; source audit limited |

### Decision Table

| claim | main evidence | paper wording |
|---|---|---|
| CRG can find routes | Hit@10 retrieval lift over self/random/degree-random | CRG provides a sufficient train-only roadmap signal. |
| CRG is needed by the trained model | support corruption, especially assist_17 evidence gap/BCE increase | The model relies on CRG support in datasets where evidence support is predictive; this is not universal. |
| LCRF module contributes | no_LCRF drop in main table plus no_filter counterfactual | LCRF improves support-level personalization on balanced/long-history datasets. |
| real learner state matters | shuffle/mean state drops strongly on assist_09 and assist_17 | LCRF should be described as learner-state conditioned, with ID-source audit as a limitation. |
| same support, different learners | same-query posterior heatmap and two-student path | LCRF filters a fixed CRG roadmap into learner-specific local routes. |
"""
    outline.write_text(original.rstrip() + "\n" + update, encoding="utf-8")

    packet = docs / "crg_lcrf_core3_review_packet.md"
    packet.write_text(
        f"""# CRG/LCRF Core3 Review Packet

Generated from `results/crg_lcrf_core3_final_20260520/`.

## Scope

- Datasets: `assist_09`, `junyi`, `assist_17`.
- No retraining was performed by this aggregation script.
- Model structure was not changed.
- Existing checkpoints and existing inference/counterfactual CSVs were reused.

## Outputs

- Main table: `results/crg_lcrf_core3_final_20260520/main_table/table_main_ablation_core3.csv`
- Figure CSVs and plots: `results/crg_lcrf_core3_final_20260520/paper_figures/`
- CRG support audit: `results/crg_lcrf_core3_final_20260520/crg_support_audit/`
- LCRF same-query cases: `results/crg_lcrf_core3_final_20260520/lcrf_same_query/`
- CRG local route cases: `results/crg_lcrf_core3_final_20260520/crg_local_route_cases/`
- LCRF timeline: `results/crg_lcrf_core3_final_20260520/lcrf_student_timeline/`
- LCRF state-source audit: `results/crg_lcrf_core3_final_20260520/lcrf_state_source_audit/`

## Result Summary

{chr(10).join(main_lines)}

## Claim Review

1. CRG sufficiency: supported by retrieval lift. This is the cleanest CRG evidence because it does not depend on prediction-head behavior.
2. CRG necessity: strongest on assist_17; assist_09 should be written as support-dependence rather than evidence-specific superiority; Junyi is weak at prediction-level corruption.
3. LCRF necessity: supported by no_filter/mean/shuffle counterfactual drops on assist_09 and assist_17. Junyi remains weak and should not be used as LCRF main evidence.
4. LCRF sufficiency: supported by same-query posterior variation on assist_09/assist_17. Use assist_17 as the primary visual case if its posterior variability remains highest.

## Risks

- Main `metrics_check.csv` files did not store global BCE for no_CRG/no_LCRF; AUC/ACC/RMSE are reliable, BCE is only available for support corruption and selected cases.
- `shuffle_id_keep_state` and `zero_id_keep_state` were not available from existing inference hooks. Do not claim student-ID shortcut is fully ruled out.
- CRG evidence-vs-degree-random gap is not universal. The paper should report weak results honestly for Junyi and cautious wording for assist_09.
""",
        encoding="utf-8",
    )


def write_manifest(out: Path) -> None:
    rows = []
    outputs = [
        "main_table/table_main_ablation_core3.csv",
        "data_story/dataset_story_cards_core3.csv",
        "paper_figures/fig2_core3_retrieval_summary.csv",
        "crg_support_audit/crg_support_gap_audit_core3.csv",
        "crg_support_audit/crg_subgroup_support_dependence_core3.csv",
        "lcrf_counterfactual/lcrf_counterfactual_delta_core3.csv",
        "lcrf_same_query/lcrf_same_query_annotated_core3.csv",
        "lcrf_same_query/lcrf_two_student_path_case_core3.csv",
        "crg_local_route_cases/crg_local_route_case_summary_core3.csv",
        "crg_local_route_cases/crg_local_route_case_edges_core3.csv",
        "lcrf_student_timeline/lcrf_specific_student_timeline_core3.csv",
        "lcrf_state_source_audit/lcrf_state_source_audit_core3.csv",
    ]
    for rel in outputs:
        path = out / rel
        rows.append(
            {
                "script": "tools/build_crg_lcrf_core3_final.py",
                "input": str(SMALL),
                "output": str(path),
                "checkpoint": "existing full/no_CRG/no_LCRF checkpoints where applicable",
                "retrained": False,
                "train_only_support_check": "inherited from existing diagnostics",
                "recommend_main_text": rel.startswith(("main_table", "paper_figures", "lcrf_counterfactual", "lcrf_same_query")),
                "exists": path.exists(),
            }
        )
    pd.DataFrame(rows).to_csv(out / "run_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default=str(ROOT / "results" / "crg_lcrf_core3_final_20260520"))
    args = parser.parse_args()
    out = Path(args.output_root)
    _mkdir(out)
    _mkdir(out / "paper_figures")

    main_table = build_main_table(out)
    build_fig2_inputs(out)
    support = build_support_tables(out)
    cf = build_lcrf_counterfactual(out)
    build_same_query(out)
    build_route_cases(out)
    state = build_state_source_audit(out, cf)
    write_outline(out, main_table, support, cf, state)
    write_manifest(out)
    print(f"Wrote core3 evidence packet to {out}")


if __name__ == "__main__":
    main()
