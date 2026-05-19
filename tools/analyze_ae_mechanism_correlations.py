#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyze whether CRG/LCRF mechanism diagnostics track prediction gains.

This is a post-hoc statistics helper for the paper-style mechanism evidence.
It does not train models.  It consumes a case-study export directory produced
by ``tools/export_ae_case_studies.py`` and writes compact CSV/Markdown files
that can be plotted by R.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


A_METRICS = (
    "a_edge_evidence_mass",
    "a_edge_item_mass",
    "a_edge_seq_mass",
    "a_top_edge_weight_sum",
    "a_top_edge_weight_max",
    "a_top_edge_entropy",
    "query_row_global_readout_delta_abs",
    "roadmap_abs_logit",
)

E_METRICS = (
    "query_row_posterior_kl",
    "query_row_posterior_delta_abs",
    "e_top_observed_abs_delta",
    "e_observed_shift_abs",
    "e_observed_state_abs",
    "e_observed_support_count_sum",
    "e_counterfactual_gain_min",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=6)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_num(frame: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prepare_common(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if col.endswith("_prob") or col.endswith("_gain") or col.endswith("_abs_error"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "query_row_global_readout_delta" in out:
        out["query_row_global_readout_delta_abs"] = pd.to_numeric(
            out["query_row_global_readout_delta"], errors="coerce"
        ).abs()
    roadmap_cols = [c for c in ("roadmap_macro_logit", "roadmap_item_logit", "roadmap_sequence_logit") if c in out]
    if roadmap_cols:
        vals = out[roadmap_cols].apply(pd.to_numeric, errors="coerce").abs()
        out["roadmap_abs_logit"] = vals.sum(axis=1)
    for col in ("a_rescue", "e_rescue", "e_counterfactual_rescue"):
        if col in out:
            out[col] = out[col].astype(str).str.lower().isin({"true", "1", "yes"})
    return out


def _spearman(frame: pd.DataFrame, metric: str, target: str) -> float:
    if metric not in frame or target not in frame:
        return np.nan
    sub = frame[[metric, target]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    if sub[target].dtype == bool:
        sub[target] = sub[target].astype(float)
    else:
        sub[target] = pd.to_numeric(sub[target], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3 or sub[metric].nunique() < 2 or sub[target].nunique() < 2:
        return np.nan
    return float(sub[metric].corr(sub[target], method="spearman"))


def _write_correlations(a_pool: pd.DataFrame, e_pool: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric in A_METRICS:
        for target in ("a_gain", "a_rescue"):
            rows.append(
                {
                    "module": "A",
                    "metric": metric,
                    "target": target,
                    "spearman": _spearman(a_pool, metric, target),
                    "n": int(a_pool[[metric, target]].dropna().shape[0]) if metric in a_pool and target in a_pool else 0,
                }
            )
    for metric in E_METRICS:
        for target in ("e_gain", "e_rescue", "e_counterfactual_rescue"):
            rows.append(
                {
                    "module": "E",
                    "metric": metric,
                    "target": target,
                    "spearman": _spearman(e_pool, metric, target),
                    "n": int(e_pool[[metric, target]].dropna().shape[0]) if metric in e_pool and target in e_pool else 0,
                }
            )
    corr = pd.DataFrame(rows)
    corr.to_csv(out_dir / "mechanism_correlations.csv", index=False)
    return corr


def _bin_table(frame: pd.DataFrame, module: str, metrics: Iterable[str], gain_col: str, rescue_col: str, bins: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for metric in metrics:
        if metric not in frame or gain_col not in frame:
            continue
        sub = frame[[metric, gain_col, rescue_col]].copy() if rescue_col in frame else frame[[metric, gain_col]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub[gain_col] = pd.to_numeric(sub[gain_col], errors="coerce")
        if rescue_col in sub:
            sub[rescue_col] = sub[rescue_col].astype(float)
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric, gain_col])
        if len(sub) < bins or sub[metric].nunique() < 2:
            continue
        try:
            sub["_bin"] = pd.qcut(sub[metric], q=min(bins, sub[metric].nunique()), duplicates="drop")
        except ValueError:
            continue
        grouped = sub.groupby("_bin", observed=True)
        for idx, (bin_label, grp) in enumerate(grouped, start=1):
            rows.append(
                {
                    "module": module,
                    "metric": metric,
                    "bin": idx,
                    "bin_label": str(bin_label),
                    "n": int(len(grp)),
                    "metric_mean": float(grp[metric].mean()),
                    "gain_mean": float(grp[gain_col].mean()),
                    "rescue_rate": float(grp[rescue_col].mean()) if rescue_col in grp else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _pool_summary(frame: pd.DataFrame, rescue_col: str) -> str:
    if frame.empty:
        return "n=0"
    parts = [f"n={len(frame)}"]
    if rescue_col in frame:
        rescue_rate = frame[rescue_col].astype(bool).mean()
        parts.append(f"{rescue_col}={rescue_rate:.3f}")
    return ", ".join(parts)


def _write_markdown(
    corr: pd.DataFrame,
    bins: pd.DataFrame,
    out_dir: Path,
    *,
    a_pool: pd.DataFrame,
    e_pool: pd.DataFrame,
    a_source: str,
    e_source: str,
) -> None:
    lines = ["# CRG/LCRF Mechanism Correlation Analysis", ""]
    lines.extend(
        [
            "## Diagnostic Pools",
            "",
            f"- A source: `{a_source}` ({_pool_summary(a_pool, 'a_rescue')})",
            f"- E source: `{e_source}` ({_pool_summary(e_pool, 'e_rescue')})",
            "",
        ]
    )
    if corr.empty:
        lines.append("No correlation rows were generated.")
    else:
        top = corr.dropna(subset=["spearman"]).copy()
        top["abs_spearman"] = top["spearman"].abs()
        top = top.sort_values("abs_spearman", ascending=False).head(12)
        lines.extend(
            [
                "## Strongest Spearman Associations",
                "",
                "| module | metric | target | spearman | n |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for r in top.itertuples(index=False):
            lines.append(f"| {r.module} | {r.metric} | {r.target} | {r.spearman:.4f} | {int(r.n)} |")
    if not bins.empty:
        lines.extend(["", "## Bin Tables", "", f"- Rows: {len(bins)}"])
    (out_dir / "mechanism_correlation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir
    out_dir = args.out_dir or (case_dir / "mechanism_correlation")
    out_dir.mkdir(parents=True, exist_ok=True)

    a_source = "a_mechanism_pool.csv" if (case_dir / "a_mechanism_pool.csv").exists() else "a_candidate_pool.csv"
    e_source = "e_mechanism_pool.csv" if (case_dir / "e_mechanism_pool.csv").exists() else "e_candidate_pool.csv"
    a_pool = _prepare_common(_read_csv(case_dir / a_source))
    e_pool = _prepare_common(_read_csv(case_dir / e_source))
    a_pool = _to_num(a_pool, A_METRICS)
    e_pool = _to_num(e_pool, E_METRICS)

    corr = _write_correlations(a_pool, e_pool, out_dir)
    bins = pd.concat(
        [
            _bin_table(a_pool, "A", A_METRICS, "a_gain", "a_rescue", int(args.bins)),
            _bin_table(e_pool, "E", E_METRICS, "e_gain", "e_rescue", int(args.bins)),
        ],
        ignore_index=True,
    )
    bins.to_csv(out_dir / "mechanism_metric_bins.csv", index=False)
    _write_markdown(corr, bins, out_dir, a_pool=a_pool, e_pool=e_pool, a_source=a_source, e_source=e_source)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
