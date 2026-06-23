#!/usr/bin/env python3
"""Re-aggregate EXISTING CRG/LCRF result CSVs into a main-line evidence ledger.

This script does NOT run the model, does NOT retrain, and does NOT change any
experiment value.  It only re-reads result CSVs already produced under
``results/`` and re-groups them to answer four questions the reframed paper
needs to settle honestly:

  Q1 Prevalence:  how large is the direct-unseen (concept-evidence-gap) regime?
  Q2 Difficulty:  is direct-unseen actually HARDER than direct-seen (AUC/BCE)?
  Q3 Specificity: do CRG's *specific* train-only relations matter beyond a
                  degree-matched random support, and is that effect located in
                  the unseen regime?  (evidence_minus_degree, by subgroup)
  Q4 LCRF clean:  what is the learner-state effect once the student-identity
                  scramble confound is removed?  (no_filter vs mean/shuffle)

Outputs a console ledger plus a tidy CSV/Markdown summary under
``results/mainline_evidence_ledger/``.

Run locally:  python tools/diagnose_mainline_evidence.py
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE3 = os.path.join(ROOT, "results", "crg_lcrf_core3_final_20260520")
MAINPROB = os.path.join(ROOT, "results", "main_problem_experiments_20260523")
OUT_DIR = os.path.join(ROOT, "results", "mainline_evidence_ledger")

STORY = os.path.join(CORE3, "data_story", "dataset_story_cards_core3.csv")
SUBDEP = os.path.join(CORE3, "crg_support_audit", "crg_subgroup_support_dependence_core3.csv")
LCRF_CF = os.path.join(CORE3, "lcrf_counterfactual", "lcrf_counterfactual_delta_core3.csv")
COVERAGE = os.path.join(MAINPROB, "coverage_conditioned_prediction_table.csv")

DATASETS = ("assist_09", "junyi", "assist_17")


def _read(path: str) -> List[dict]:
    if not os.path.exists(path):
        print(f"[warn] missing: {path}")
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def q1_prevalence() -> List[dict]:
    rows = _read(STORY)
    out = []
    for r in rows:
        out.append({
            "dataset": r["dataset"],
            "direct_unseen_rate": _f(r, "direct_unseen_rate"),
            "bridge_only_rate": _f(r, "bridge_only_rate"),
            "single_concept_rate": _f(r, "single_concept_rate"),
            "seq_edge_density": _f(r, "seq_edge_density"),
            "history_len_median": _f(r, "history_len_median"),
        })
    return out


def q2_difficulty() -> List[dict]:
    """Clean (ratio==0) AUC/BCE per subgroup -> is unseen harder than seen?"""
    rows = _read(SUBDEP)
    # key: (dataset, subgroup) -> clean auc/bce averaged over seeds
    acc: Dict[tuple, Dict[str, list]] = defaultdict(lambda: {"auc": [], "bce": [], "n": []})
    for r in rows:
        if _f(r, "corruption_ratio") != 0.0:
            continue
        sg = r.get("subgroup", "")
        if sg not in ("direct_seen", "direct_unseen", "direct_unseen_bridgeable", "all"):
            continue
        k = (r["dataset"], sg)
        ca, cb = _f(r, "clean_auc"), _f(r, "clean_bce")
        n = _f(r, "n_eval")
        if ca is not None:
            acc[k]["auc"].append(ca)
        if cb is not None:
            acc[k]["bce"].append(cb)
        if n is not None:
            acc[k]["n"].append(n)
    out = []
    for (ds, sg), d in sorted(acc.items()):
        out.append({
            "dataset": ds, "subgroup": sg,
            "n_eval": int(mean(d["n"])) if d["n"] else None,
            "clean_auc": round(mean(d["auc"]), 4) if d["auc"] else None,
            "clean_bce": round(mean(d["bce"]), 4) if d["bce"] else None,
        })
    return out


def q3_specificity() -> List[dict]:
    """evidence_minus_degree at full corruption, by subgroup (seen vs unseen).

    A value near 0 => specific train-only relations do not matter beyond degree.
    Positive => the specific relations carry information beyond degree structure.
    """
    rows = _read(SUBDEP)
    acc: Dict[tuple, Dict[str, list]] = defaultdict(
        lambda: {"emd_auc": [], "emd_bce": [], "ev_auc_drop": [], "ev_bce_inc": []}
    )
    for r in rows:
        if r.get("corruption_type") != "evidence_support_corruption":
            continue
        if _f(r, "corruption_ratio") != 1.0:
            continue
        sg = r.get("subgroup", "")
        if sg not in ("direct_seen", "direct_unseen", "direct_unseen_bridgeable", "all"):
            continue
        k = (r["dataset"], sg)
        for col, key in (
            ("evidence_minus_degree_random_auc_drop", "emd_auc"),
            ("evidence_minus_degree_random_bce_increase", "emd_bce"),
            ("auc_drop_from_clean", "ev_auc_drop"),
            ("bce_increase_from_clean", "ev_bce_inc"),
        ):
            v = _f(r, col)
            if v is not None:
                acc[k][key].append(v)
    out = []
    for (ds, sg), d in sorted(acc.items()):
        out.append({
            "dataset": ds, "subgroup": sg,
            "evidence_auc_drop": round(mean(d["ev_auc_drop"]), 4) if d["ev_auc_drop"] else None,
            "evidence_bce_increase": round(mean(d["ev_bce_inc"]), 4) if d["ev_bce_inc"] else None,
            "evidence_minus_degree_auc": round(mean(d["emd_auc"]), 4) if d["emd_auc"] else None,
            "evidence_minus_degree_bce": round(mean(d["emd_bce"]), 4) if d["emd_bce"] else None,
        })
    return out


def q3b_coverage_gain_location() -> List[dict]:
    """Where does the ablation gain live: direct_seen vs direct_unseen subgroup?"""
    rows = _read(COVERAGE)
    out = []
    for r in rows:
        sg = r.get("subgroup", "")
        if sg not in ("direct_seen", "direct_unseen_bridgeable", "weak_direct_evidence", "high_route_mass"):
            continue
        out.append({
            "dataset": r["dataset"], "subgroup": sg,
            "comparison": r.get("comparison", ""),
            "n_eval": int(_f(r, "n_eval") or 0),
            "delta_auc": round(_f(r, "delta_auc"), 4) if _f(r, "delta_auc") is not None else None,
            "delta_auc_ci_low": round(_f(r, "delta_auc_ci_low"), 4) if _f(r, "delta_auc_ci_low") is not None else None,
            "delta_auc_ci_high": round(_f(r, "delta_auc_ci_high"), 4) if _f(r, "delta_auc_ci_high") is not None else None,
            "delta_bce": round(_f(r, "delta_bce"), 4) if _f(r, "delta_bce") is not None else None,
        })
    return out


def q4_lcrf_clean() -> List[dict]:
    rows = _read(LCRF_CF)
    out = []
    for r in rows:
        out.append({
            "dataset": r["dataset"], "variant": r["variant"],
            "auc_drop_from_full": round(_f(r, "auc_drop_from_full"), 4) if _f(r, "auc_drop_from_full") is not None else None,
            "support_identical": r.get("support_identical_check_passed", ""),
            "note": r.get("notes", ""),
        })
    return out


def _read_many(root: str, filename: str) -> List[dict]:
    rows: List[dict] = []
    if not root or not os.path.isdir(root):
        return rows
    for dirpath, _, files in os.walk(root):
        if filename not in files:
            continue
        rows.extend(_read(os.path.join(dirpath, filename)))
    return rows


def q5_new_e2_gap_calibration(new_run_root: str) -> List[dict]:
    rows = _read_many(new_run_root, "gap_calibration_metrics_all.csv")
    out: List[dict] = []
    for r in rows:
        if r.get("target_history_bucket") != "0":
            continue
        if r.get("variant") not in {"full", "no_CRG", "degree_random_support"}:
            continue
        out.append({
            "dataset": r.get("dataset", ""),
            "bucket": "0",
            "variant": r.get("variant", ""),
            "n_eval": int(_f(r, "n_eval") or 0),
            "auc": round(_f(r, "auc"), 4) if _f(r, "auc") is not None else None,
            "bce": round(_f(r, "bce"), 4) if _f(r, "bce") is not None else None,
            "delta_bce_variant_minus_full": round(_f(r, "delta_bce_variant_minus_full"), 4)
            if _f(r, "delta_bce_variant_minus_full") is not None
            else None,
            "delta_bce_ci_low": round(_f(r, "delta_bce_ci_low"), 4)
            if _f(r, "delta_bce_ci_low") is not None
            else None,
        })
    return out


def q6_new_e4_decoupled_support(new_run_root: str) -> List[dict]:
    rows = _read_many(new_run_root, "decoupled_support_metrics_all.csv")
    out: List[dict] = []
    for r in rows:
        if r.get("subgroup") != "direct_unseen":
            continue
        if r.get("variant") not in {"s_support_random_same_total", "s_support_degree_matched_random"}:
            continue
        out.append({
            "dataset": r.get("dataset", ""),
            "subgroup": r.get("subgroup", ""),
            "variant": r.get("variant", ""),
            "decouple_support_runtime": r.get("decouple_support_runtime", ""),
            "a_masks_unchanged": r.get("a_masks_unchanged", ""),
            "auc_drop_from_clean": round(_f(r, "auc_drop_from_clean"), 4)
            if _f(r, "auc_drop_from_clean") is not None
            else None,
            "bce_increase_from_clean": round(_f(r, "bce_increase_from_clean"), 4)
            if _f(r, "bce_increase_from_clean") is not None
            else None,
            "bce_ci_low": round(_f(r, "bce_increase_ci_low"), 4)
            if _f(r, "bce_increase_ci_low") is not None
            else None,
        })
    return out


def _print_table(title: str, rows: List[dict]) -> None:
    print(f"\n### {title}")
    if not rows:
        print("(no data)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def _write_md(sections: Dict[str, List[dict]]) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    md_path = os.path.join(OUT_DIR, "mainline_evidence_ledger.md")
    with open(md_path, "w") as fh:
        fh.write("# Main-line evidence ledger (re-aggregated from existing results)\n\n")
        fh.write("Generated by `tools/diagnose_mainline_evidence.py`. "
                 "No experiment value is changed; this only re-groups existing result CSVs.\n")
        for title, rows in sections.items():
            fh.write(f"\n## {title}\n\n")
            if not rows:
                fh.write("_(no data)_\n")
                continue
            cols = list(rows[0].keys())
            fh.write("| " + " | ".join(cols) + " |\n")
            fh.write("| " + " | ".join("---" for _ in cols) + " |\n")
            for r in rows:
                fh.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
        # also dump each section as csv
        for title, rows in sections.items():
            if not rows:
                continue
            slug = title.lower().split(":")[0].strip()
            slug = "".join(ch if ch.isalnum() else "_" for ch in slug).strip("_")
            with open(os.path.join(OUT_DIR, f"{slug}.csv"), "w", newline="") as cf:
                w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run-root", default=os.environ.get("MAINLINE_NEW_RUN_ROOT", ""))
    args = parser.parse_args()
    sections = {
        "Q1 Prevalence of the concept-evidence-gap regime": q1_prevalence(),
        "Q2 Difficulty: clean AUC/BCE by coverage subgroup": q2_difficulty(),
        "Q3 Specificity: evidence-vs-degree by subgroup (full corruption)": q3_specificity(),
        "Q3b Where the ablation gain lives (coverage-conditioned dAUC)": q3b_coverage_gain_location(),
        "Q4 LCRF clean (no_filter) vs student-scramble (mean/shuffle)": q4_lcrf_clean(),
    }
    if args.new_run_root:
        sections["Q5 New E2: bucket=0 gap calibration"] = q5_new_e2_gap_calibration(args.new_run_root)
        sections["Q6 New E4: decoupled support perturbation"] = q6_new_e4_decoupled_support(args.new_run_root)
    print("=" * 78)
    print("MAIN-LINE EVIDENCE LEDGER  (re-aggregation of existing results only)")
    if args.new_run_root:
        print(f"NEW RUN ROOT: {args.new_run_root}")
    print("=" * 78)
    for title, rows in sections.items():
        _print_table(title, rows)
    md = _write_md(sections)
    print(f"\nWrote: {md}")
    print(f"       {OUT_DIR}/*.csv")


if __name__ == "__main__":
    main()
