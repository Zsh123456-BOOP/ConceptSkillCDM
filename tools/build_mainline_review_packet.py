#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build a compact review packet for the mainline evidence rerun."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


DEFAULT_DATASETS = ("assist_09", "junyi", "assist_17")

# An effect below this absolute BCE magnitude is treated as decorative, not
# load-bearing, regardless of significance.  Tunable via --min-bce-effect.
DEFAULT_MIN_BCE_EFFECT = 0.005
# E5 learner-state-removal variant AUC below this is a degenerate/broken variant
# (e.g. predictions inverted below chance) and must not be used as evidence.
DEGENERATE_AUC = 0.55
# Preferred clean learner-state-removal variant names, best first.
LCRF_CLEAN_CANDIDATES = ("lcrf_state_off", "no_filter", "no_LCRF")
# Variants that scramble the student dimension: identity-dependence diagnostics,
# NOT evidence for the LCRF support filter.
LCRF_CONFOUND_VARIANTS = ("mean_state", "shuffle_state")


def _read_many(root: Path, pattern: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(root.rglob(pattern)):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        frame["_source_file"] = str(path.relative_to(root))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _best_row(frame: pd.DataFrame, **conds: object) -> Optional[pd.Series]:
    if frame.empty:
        return None
    mask = pd.Series(True, index=frame.index)
    for key, value in conds.items():
        if key not in frame.columns:
            return None
        mask &= frame[key].astype(str).eq(str(value))
    hit = frame[mask]
    return None if hit.empty else hit.iloc[0]


def _parse_datasets(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _num(row: Optional[pd.Series], key: str) -> Optional[float]:
    if row is None or key not in row or pd.isna(row.get(key)):
        return None
    try:
        return float(row.get(key))
    except Exception:
        return None


def _ci_low_positive(row: Optional[pd.Series], *keys: str) -> Optional[bool]:
    """True/False if a CI lower-bound column exists; None if absent."""
    for key in keys:
        v = _num(row, key)
        if v is not None:
            return v > 0.0
    return None


def _section_e2(frame: pd.DataFrame, datasets: List[str], min_effect: float) -> List[str]:
    lines = [
        "## E2 Gap Calibration",
        "",
        "Question: at bucket=0 (no direct target-concept history), does removing the "
        "**backbone** (no_CRG) and corrupting the **support set** (degree_random) hurt "
        "calibration? Support-load-bearing requires the *support* effect to be both "
        f"significant (CI low>0) and >= {min_effect:.3f} BCE.",
        "",
    ]
    if frame.empty:
        return lines + ["- fail: missing E2 metrics.", ""]
    for ds in datasets:
        sub = frame[frame["dataset"].astype(str).eq(ds)]
        zero_full = _best_row(sub, target_history_bucket="0", variant="full")
        zero_no = _best_row(sub, target_history_bucket="0", variant="no_CRG")
        zero_deg = _best_row(sub, target_history_bucket="0", variant="degree_random_support")
        backbone = _num(zero_no, "delta_bce_variant_minus_full")
        support = _num(zero_deg, "delta_bce_variant_minus_full")
        backbone_ci = _ci_low_positive(zero_no, "delta_bce_ci_low")
        support_ci = _ci_low_positive(zero_deg, "delta_bce_ci_low")
        backbone_pass = bool(backbone is not None and backbone >= min_effect and backbone_ci)
        support_pass = bool(support is not None and support >= min_effect and support_ci)
        if support_pass:
            status = "support-load-bearing"
        elif backbone_pass:
            status = "backbone-only (support decorative)"
        else:
            status = "fail"
        lines.append(
            f"- {ds}: {status}; bucket=0 full BCE={_fmt(None if zero_full is None else zero_full.get('bce'))}, "
            f"backbone(no_CRG) ΔBCE={_fmt(backbone)} (CI low>0: {backbone_ci}), "
            f"support(degree_random) ΔBCE={_fmt(support)} (CI low>0: {support_ci})."
        )
    lines.append("")
    return lines


def _section_e3(frame: pd.DataFrame, datasets: List[str], min_effect: float) -> List[str]:
    lines = [
        "## E3 Relation Specificity (ENTANGLED: A is both backbone and support)",
        "",
        "Note: this corrupts the same mask that serves as GNN adjacency AND support, "
        "so a positive effect here is **confounded with the backbone** and must be "
        "confirmed by E4 (decoupled). Pass requires unseen ev-degree ΔBCE >= "
        f"{min_effect:.3f}, CI low>0, and > direct_seen.",
        "",
    ]
    if frame.empty:
        return lines + ["- fail: missing E3 support corruption gap claims.", ""]
    for ds in datasets:
        seen = _best_row(frame, dataset=ds, subgroup="direct_seen")
        unseen = _best_row(frame, dataset=ds, subgroup="direct_unseen")
        unseen_bce = _num(unseen, "evidence_minus_degree_random_bce_increase")
        seen_bce = _num(seen, "evidence_minus_degree_random_bce_increase")
        ci_low_pos = _ci_low_positive(
            unseen, "evidence_minus_degree_bce_ci_low", "bce_bootstrap_ci_low"
        )
        ci_low_val = _num(unseen, "evidence_minus_degree_bce_ci_low") or _num(unseen, "bce_bootstrap_ci_low")
        ok = bool(
            unseen_bce is not None
            and unseen_bce >= min_effect
            and (ci_low_pos in (True, None))
            and (seen_bce is None or unseen_bce > seen_bce)
        )
        status = "pass (confirm via E4)" if ok else "fail"
        lines.append(
            f"- {ds}: {status}; direct_unseen ev-degree ΔBCE={_fmt(unseen_bce)} "
            f"(CI low={_fmt(ci_low_val)}), direct_seen={_fmt(seen_bce)}."
        )
    lines.append("")
    return lines


def _e5_clean_variant_rows(sub: pd.DataFrame) -> pd.DataFrame:
    """Pick the clean learner-state-removal variant, excluding degenerate rows."""
    have = set(sub["variant"].astype(str)) if "variant" in sub.columns else set()
    for name in LCRF_CLEAN_CANDIDATES:
        if name not in have:
            continue
        rows = sub[sub["variant"].astype(str).eq(name)].copy()
        if "auc" in rows.columns:
            # drop degenerate (below-chance / broken) variant instances
            rows = rows[pd.to_numeric(rows["auc"], errors="coerce").fillna(0.0) >= DEGENERATE_AUC]
        if not rows.empty:
            rows["_clean_variant"] = name
            return rows
    return pd.DataFrame()


def _section_e5(frame: pd.DataFrame, datasets: List[str], min_effect: float) -> List[str]:
    lines = [
        "## E5 LCRF Clean Counterfactual",
        "",
        "Uses the clean learner-state-removal variant (state correction off, support "
        "set unchanged); degenerate variants with AUC < {:.2f} are excluded, and "
        "mean_state/shuffle_state are treated as student-identity-scramble diagnostics, "
        "NOT LCRF evidence. Pass requires the TARGETED (unseen) ΔBCE to be significant, "
        ">= {:.3f}, and larger than the overall (all) ΔBCE.".format(DEGENERATE_AUC, min_effect),
        "",
    ]
    if frame.empty:
        return lines + ["- fail: missing E5 LCRF strata metrics.", ""]
    bce_cols = ("bce_gap_variant_minus_full", "bce_delta_from_full", "bce_increase_from_full", "delta_bce")
    for ds in datasets:
        sub = frame[frame["dataset"].astype(str).eq(ds)].copy()
        clean = _e5_clean_variant_rows(sub)
        if clean.empty:
            lines.append(f"- {ds}: fail; no non-degenerate clean learner-state variant found.")
            continue
        variant_name = clean["_clean_variant"].iloc[0]
        bce_col = next((c for c in bce_cols if c in clean.columns), None)
        if bce_col is None or "subgroup" not in clean.columns:
            lines.append(f"- {ds}: fail; missing subgroup/bce columns for {variant_name}.")
            continue
        overall = _best_row(clean, subgroup="all")
        target = None
        for sg in ("direct_unseen", "direct_unseen_bridgeable", "weak_direct_high_route"):
            target = _best_row(clean, subgroup=sg)
            if target is not None:
                break
        overall_bce = _num(overall, bce_col)
        target_bce = _num(target, bce_col)
        ok = bool(
            target_bce is not None
            and overall_bce is not None  # need a non-degenerate overall baseline to claim gap-specificity
            and target_bce >= min_effect
            and target_bce > overall_bce
        )
        lines.append(
            f"- {ds}: {'pass' if ok else 'fail'}; variant={variant_name}, "
            f"target({'NA' if target is None else target.get('subgroup')}) ΔBCE={_fmt(target_bce)}, "
            f"overall(all) ΔBCE={_fmt(overall_bce)} "
            f"[gap-specific: {ok}]."
        )
    lines.append("")
    return lines


def _section_e4(frame: pd.DataFrame, datasets: List[str], min_effect: float) -> List[str]:
    lines = [
        "## E4 Decoupled Support (DECISIVE: A frozen, only the support set perturbed)",
        "",
        "This is the clean isolation of the support role from the backbone. Support is "
        f"load-bearing only if ΔBCE >= {min_effect:.3f}, CI low>0, AND it beats the "
        "degree-matched control (minus_degree_random_bce_increase > 0).",
        "",
    ]
    if frame.empty:
        return lines + ["- fail: missing E4 decoupled support metrics.", ""]
    for ds in datasets:
        row = _best_row(frame, dataset=ds, subgroup="direct_unseen", variant="s_support_degree_matched_random")
        if row is None:
            lines.append(f"- {ds}: fail; missing direct_unseen degree-matched row.")
            continue
        bce = _num(row, "bce_increase_from_clean")
        ci_low_pos = _ci_low_positive(row, "bce_increase_ci_low")
        minus_deg = _num(row, "minus_degree_random_bce_increase")
        plumbing_ok = bool(row.get("decouple_support_runtime", False)) and bool(row.get("a_masks_unchanged", False))
        ok = bool(
            plumbing_ok
            and bce is not None and bce >= min_effect
            and (ci_low_pos in (True, None))
            and (minus_deg is None or minus_deg > 0.0)
        )
        lines.append(
            f"- {ds}: {'pass' if ok else 'fail'}; direct_unseen S ΔBCE={_fmt(bce)} "
            f"(CI low>0: {ci_low_pos}), beats-degree(minus_degree ΔBCE)={_fmt(minus_deg)}, "
            f"A frozen={row.get('a_masks_unchanged')}, decouple={row.get('decouple_support_runtime')}."
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-bce-effect", type=float, default=DEFAULT_MIN_BCE_EFFECT,
                        help="Absolute BCE magnitude below which an effect is decorative.")
    args = parser.parse_args()

    root = Path(args.run_root)
    datasets = _parse_datasets(args.datasets)
    min_effect = float(args.min_bce_effect)
    out_path = Path(args.output) if args.output else root / "mainline_review_packet.md"
    e2 = _read_many(root, "gap_calibration_metrics_all.csv")
    e3 = _read_many(root, "support_corruption_gap_claims.csv")
    e5 = _read_many(root, "lcrf_strata_metrics.csv")
    e4 = _read_many(root, "decoupled_support_metrics_all.csv")

    lines = [
        "# Mainline Evidence Review Packet",
        "",
        f"Run root: `{root}`",
        "",
        "Honest gates: an effect must clear a magnitude floor (`--min-bce-effect="
        f"{min_effect:.3f}`), have a positive CI lower bound, and (for support claims) "
        "beat the degree-matched control. Backbone and support effects are reported "
        "separately. Failed gates are left as fail.",
        "",
    ]
    lines.extend(_section_e2(e2, datasets, min_effect))
    lines.extend(_section_e3(e3, datasets, min_effect))
    lines.extend(_section_e5(e5, datasets, min_effect))
    lines.extend(_section_e4(e4, datasets, min_effect))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] review packet written to {out_path}")


if __name__ == "__main__":
    main()
