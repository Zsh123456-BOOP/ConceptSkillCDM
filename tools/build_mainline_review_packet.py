#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build a compact review packet for the mainline evidence rerun."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


DATASETS = ("assist_09", "junyi", "assist_17")


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


def _section_e2(frame: pd.DataFrame) -> List[str]:
    lines = ["## E2 Gap Calibration", ""]
    if frame.empty:
        return lines + ["- fail: missing E2 metrics.", ""]
    for ds in DATASETS:
        sub = frame[frame["dataset"].astype(str).eq(ds)]
        zero_full = _best_row(sub, target_history_bucket="0", variant="full")
        zero_no = _best_row(sub, target_history_bucket="0", variant="no_CRG")
        zero_deg = _best_row(sub, target_history_bucket="0", variant="degree_random_support")
        pass_no = bool(
            zero_no is not None
            and "delta_bce_ci_low" in zero_no
            and pd.notna(zero_no["delta_bce_ci_low"])
            and float(zero_no["delta_bce_ci_low"]) > 0.0
        )
        pass_deg = bool(
            zero_deg is not None
            and "delta_bce_ci_low" in zero_deg
            and pd.notna(zero_deg["delta_bce_ci_low"])
            and float(zero_deg["delta_bce_ci_low"]) > 0.0
        )
        status = "pass" if pass_no or pass_deg else "fail"
        lines.append(
            f"- {ds}: {status}; bucket=0 full BCE={_fmt(None if zero_full is None else zero_full.get('bce'))}, "
            f"no_CRG ΔBCE={_fmt(None if zero_no is None else zero_no.get('delta_bce_variant_minus_full'))}, "
            f"degree_random ΔBCE={_fmt(None if zero_deg is None else zero_deg.get('delta_bce_variant_minus_full'))}."
        )
    lines.append("")
    return lines


def _section_e3(frame: pd.DataFrame) -> List[str]:
    lines = ["## E3 Relation Specificity", ""]
    if frame.empty:
        return lines + ["- fail: missing E3 support corruption gap claims.", ""]
    for ds in DATASETS:
        seen = _best_row(frame, dataset=ds, subgroup="direct_seen")
        unseen = _best_row(frame, dataset=ds, subgroup="direct_unseen")
        unseen_bce = None if unseen is None else unseen.get("evidence_minus_degree_random_bce_increase")
        seen_bce = None if seen is None else seen.get("evidence_minus_degree_random_bce_increase")
        ci_low = None if unseen is None else unseen.get("bce_bootstrap_ci_low")
        ok = bool(
            unseen_bce is not None
            and pd.notna(unseen_bce)
            and float(unseen_bce) > 0.0
            and (seen_bce is None or pd.isna(seen_bce) or float(unseen_bce) > float(seen_bce))
        )
        status = "pass" if ok else "fail"
        lines.append(
            f"- {ds}: {status}; direct_unseen ev-degree ΔBCE={_fmt(unseen_bce)} "
            f"(CI low={_fmt(ci_low)}), direct_seen={_fmt(seen_bce)}."
        )
    lines.append("")
    return lines


def _section_e5(frame: pd.DataFrame) -> List[str]:
    lines = ["## E5 LCRF Clean Counterfactual", ""]
    if frame.empty:
        return lines + ["- fail: missing E5 LCRF strata metrics.", ""]
    for ds in DATASETS:
        sub = frame[
            frame["dataset"].astype(str).eq(ds)
            & frame["variant"].astype(str).eq("no_filter")
        ].copy()
        if sub.empty:
            lines.append(f"- {ds}: fail; missing no_filter rows.")
            continue
        if "subgroup" in sub.columns:
            target = sub[sub["subgroup"].astype(str).isin(["direct_unseen_bridgeable", "weak_direct_high_route", "direct_unseen"])]
            if target.empty:
                target = sub
        else:
            target = sub
        bce_col = (
            "bce_gap_variant_minus_full"
            if "bce_gap_variant_minus_full" in target.columns
            else "bce_delta_from_full"
            if "bce_delta_from_full" in target.columns
            else "bce_increase_from_full"
            if "bce_increase_from_full" in target.columns
            else "delta_bce"
            if "delta_bce" in target.columns
            else None
        )
        row = target.sort_values(bce_col).iloc[-1] if bce_col else target.iloc[0]
        bce_delta = None if bce_col is None else row.get(bce_col)
        ok = bce_delta is not None and pd.notna(bce_delta) and float(bce_delta) > 0.0
        lines.append(
            f"- {ds}: {'pass' if ok else 'fail'}; subgroup={row.get('subgroup', 'NA')}, "
            f"no_filter ΔBCE={_fmt(bce_delta)}, "
            f"ΔAUC={_fmt(row.get('auc_gap_full_minus_variant', row.get('auc_delta_from_full', row.get('auc_drop_from_full', None))))}."
        )
    lines.append("")
    return lines


def _section_e4(frame: pd.DataFrame) -> List[str]:
    lines = ["## E4 Decoupled Support", ""]
    if frame.empty:
        return lines + ["- fail: missing E4 decoupled support metrics.", ""]
    for ds in DATASETS:
        row = _best_row(frame, dataset=ds, subgroup="direct_unseen", variant="s_support_degree_matched_random")
        if row is None:
            lines.append(f"- {ds}: fail; missing direct_unseen degree-matched row.")
            continue
        ok = bool(
            bool(row.get("decouple_support_runtime", False))
            and bool(row.get("a_masks_unchanged", False))
            and pd.notna(row.get("bce_increase_from_clean"))
            and float(row.get("bce_increase_from_clean")) > 0.0
        )
        lines.append(
            f"- {ds}: {'pass' if ok else 'fail'}; direct_unseen S degree-random ΔBCE="
            f"{_fmt(row.get('bce_increase_from_clean'))}, A frozen={row.get('a_masks_unchanged')}, "
            f"decouple_support={row.get('decouple_support_runtime')}."
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.run_root)
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
        "This packet only aggregates new result CSVs. Failed gates are left as fail.",
        "",
    ]
    lines.extend(_section_e2(e2))
    lines.extend(_section_e3(e3))
    lines.extend(_section_e5(e5))
    lines.extend(_section_e4(e4))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] review packet written to {out_path}")


if __name__ == "__main__":
    main()
