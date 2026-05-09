#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Summarize whether the staged ERS/SLPR mechanism experiments are effective."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_csv", type=Path)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--min_a_drop", type=float, default=0.005)
    parser.add_argument("--min_e_drop", type=float, default=0.002)
    parser.add_argument("--min_evidence_gain", type=float, default=0.002)
    return parser.parse_args()


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: Optional[float], digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing result CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _best_by_variant(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if str(row.get("status", "")).lower() not in {"ok", "metrics_ok"}:
            continue
        variant = str(row.get("variant") or row.get("ablation") or "")
        auc = _to_float(row.get("test_auc"))
        if not variant or auc is None:
            continue
        old = best.get(variant)
        old_auc = _to_float(old.get("test_auc")) if old else None
        if old is None or old_auc is None or auc > old_auc:
            best[variant] = row
    return best


def _delta(full: Optional[float], ablated: Optional[float]) -> Optional[float]:
    if full is None or ablated is None:
        return None
    return full - ablated


def _classify(
    *,
    drop_no_a: Optional[float],
    drop_no_e: Optional[float],
    evidence_gain: Optional[float],
    args: argparse.Namespace,
) -> Tuple[str, str, str]:
    a_ok = (
        drop_no_a is not None
        and evidence_gain is not None
        and drop_no_a >= float(args.min_a_drop)
        and evidence_gain >= float(args.min_evidence_gain)
    )
    e_ok = drop_no_e is not None and drop_no_e >= float(args.min_e_drop)
    if a_ok and e_ok:
        verdict = "A_and_E_effective"
    elif a_ok:
        verdict = "A_effective_E_weak"
    elif e_ok:
        verdict = "E_effective_A_weak"
    else:
        verdict = "not_effective_or_inconclusive"
    return verdict, "yes" if a_ok else "no", "yes" if e_ok else "no"


def write_summary(rows: List[Dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> Tuple[Path, Path]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        phase = str(row.get("phase") or "")
        dataset = str(row.get("dataset") or "")
        if phase and dataset:
            grouped[(phase, dataset)].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (phase, dataset), grp in sorted(grouped.items()):
        best = _best_by_variant(grp)
        full_auc = _to_float(best.get("full", {}).get("test_auc"))
        no_a_auc = _to_float(best.get("no_A", {}).get("test_auc"))
        no_e_auc = _to_float(best.get("no_E", {}).get("test_auc"))
        item_auc = _to_float(best.get("A_item_only", {}).get("test_auc"))
        seq_auc = _to_float(best.get("A_seq_only", {}).get("test_auc"))
        uniform_auc = _to_float(best.get("A_uniform", {}).get("test_auc"))
        self_auc = _to_float(best.get("A_self_only", {}).get("test_auc"))
        prior_auc = _to_float(best.get("E_prior_only", {}).get("test_auc"))
        frozen_auc = _to_float(best.get("E_frozen_alpha", {}).get("test_auc"))

        drop_no_a = _delta(full_auc, no_a_auc)
        drop_no_e = _delta(full_auc, no_e_auc)
        evidence_gain = _delta(full_auc, uniform_auc)
        seq_minus_item = None if seq_auc is None or item_auc is None else seq_auc - item_auc
        e_posterior_gain = _delta(full_auc, prior_auc)
        e_gate_gain = _delta(full_auc, frozen_auc)
        verdict, a_effective, e_effective = _classify(
            drop_no_a=drop_no_a,
            drop_no_e=drop_no_e,
            evidence_gain=evidence_gain,
            args=args,
        )

        full_row = best.get("full", {})
        summary_rows.append(
            {
                "phase": phase,
                "dataset": dataset,
                "full_auc": _fmt(full_auc),
                "no_A_auc": _fmt(no_a_auc),
                "no_E_auc": _fmt(no_e_auc),
                "A_item_only_auc": _fmt(item_auc),
                "A_seq_only_auc": _fmt(seq_auc),
                "A_uniform_auc": _fmt(uniform_auc),
                "A_self_only_auc": _fmt(self_auc),
                "E_prior_only_auc": _fmt(prior_auc),
                "E_frozen_alpha_auc": _fmt(frozen_auc),
                "drop_no_A": _fmt(drop_no_a),
                "drop_no_E": _fmt(drop_no_e),
                "evidence_gain_vs_uniform": _fmt(evidence_gain),
                "seq_minus_item": _fmt(seq_minus_item),
                "E_posterior_gain_vs_prior_only": _fmt(e_posterior_gain),
                "E_gate_gain_vs_frozen_alpha": _fmt(e_gate_gain),
                "A_effective": a_effective,
                "E_effective": e_effective,
                "verdict": verdict,
                "support_final_size_mean": full_row.get("support_final_size_mean", ""),
                "support_item_survival_rate": full_row.get("support_item_survival_rate", ""),
                "support_seq_survival_rate": full_row.get("support_seq_survival_rate", ""),
                "query_row_posterior_kl": full_row.get("query_row_posterior_kl", ""),
                "query_row_posterior_delta_abs": full_row.get("query_row_posterior_delta_abs", ""),
                "alpha_std": full_row.get("alpha_std", ""),
                "best_full_log": full_row.get("log_file", ""),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "mechanism_effectiveness_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["phase", "dataset", "verdict"]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_md = out_dir / "mechanism_effectiveness_summary.md"
    lines = [
        "# Mechanism Effectiveness Summary",
        "",
        f"- A effective threshold: no_A drop >= {args.min_a_drop:.4f} and full - A_uniform >= {args.min_evidence_gain:.4f}",
        f"- E effective threshold: no_E drop >= {args.min_e_drop:.4f}",
        "",
        "| phase | dataset | full | drop_no_A | drop_no_E | evidence_gain | seq-item | verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {phase} | {dataset} | {full_auc} | {drop_no_A} | {drop_no_E} | "
            "{evidence_gain_vs_uniform} | {seq_minus_item} | {verdict} |".format(**row)
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_csv, summary_md


def main() -> None:
    args = parse_args()
    rows = read_rows(args.result_csv)
    out_dir = args.out_dir or args.result_csv.parent
    csv_path, md_path = write_summary(rows, out_dir, args)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
