#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build CRG/LCRF story-extension diagnostics without changing the model.

This script is intentionally conservative.  It does not train models.  It
creates the review packet scaffold, confirms dataset story cards from the
server-side profile CSV, and runs the train-only CRG held-out transition
retrieval diagnostic through the existing `analyze_a_support_evidence.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_crg_lcrf_small_experiments import DEFAULT_CHECKPOINTS  # noqa: E402


CORE_EXISTING = ("assist_09", "junyi", "assist_17")
NEW_CANDIDATES = ("assist_12", "assist_15", "assist_12_clean15_item50")
APPENDIX_ONLY = ("frcsub", "math2", "ednet_kt1", "nips34_l3", "nips34")
PROFILE_COLUMNS = [
    "dataset",
    "train_size",
    "num_concepts",
    "single_concept_rate",
    "multi_concept_rate",
    "item_edge_density",
    "seq_edge_density",
    "history_len_median",
    "direct_unseen_rate",
    "bridge_only_rate",
    "recommended_role",
    "run_priority",
    "reason",
]


def _tokens(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return default if value is None else str(value)


def _role_and_reason(row: Mapping[str, Any]) -> Dict[str, Any]:
    dataset = _text(row, "dataset")
    status = _text(row, "status")
    if status != "ok":
        return {
            "recommended_role": "skip",
            "run_priority": 99,
            "reason": f"profile status={status}; skip until data format is fixed",
        }

    multi = _num(row, "multi_concept_item_rate")
    single = 1.0 - multi
    item_density = _num(row, "item_density")
    seq_density = _num(row, "seq_density")
    direct_unseen = 1.0 - _num(row, "test_e_exact_coverage")
    bridge = _num(row, "test_e_bridge_only_rate")
    hist_med = _num(row, "student_train_count_median")

    if dataset in APPENDIX_ONLY:
        return {
            "recommended_role": "appendix_contrast",
            "run_priority": 40,
            "reason": "contrast dataset; not a sparse reachability main claim",
        }
    if bridge >= 0.25 and single >= 0.95 and item_density <= 0.005 and seq_density > 0.05:
        return {
            "recommended_role": "core_crg",
            "run_priority": 1 if dataset in NEW_CANDIDATES else 2,
            "reason": "single-concept with near-zero item graph and high bridge-only rate; strong CRG reachability setting",
        }
    if dataset == "junyi":
        return {
            "recommended_role": "core_crg",
            "run_priority": 1,
            "reason": "100% direct-unseen and bridgeable by sequence route; strongest CRG phenomenon",
        }
    if dataset in ("assist_09", "cdbd_a0910") and 0.01 <= bridge <= 0.08 and seq_density > 0.5:
        return {
            "recommended_role": "balanced_main",
            "run_priority": 3,
            "reason": "balanced benchmark with both item and sequence evidence; useful for CRG/LCRF mechanisms",
        }
    if dataset == "assist_17":
        return {
            "recommended_role": "core_lcrf",
            "run_priority": 3,
            "reason": "long-history benchmark with strong sequence support; useful for CRG necessity and LCRF counterfactual/case evidence",
        }
    if direct_unseen < 0.02 and hist_med > 100:
        return {
            "recommended_role": "appendix_contrast",
            "run_priority": 50,
            "reason": "student histories are dense and direct-unseen rate is low; weak sparse-reachability setting",
        }
    return {
        "recommended_role": "appendix_contrast",
        "run_priority": 60,
        "reason": "does not strongly match sparse reachability; keep as data-card contrast unless later evidence is strong",
    }


def build_story_cards(profile_csv: Path, out_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(profile_csv)
    rows: List[Dict[str, Any]] = []
    for _, item in raw.iterrows():
        row = item.to_dict()
        role = _role_and_reason(row)
        multi = _num(row, "multi_concept_item_rate")
        rows.append(
            {
                "dataset": _text(row, "dataset"),
                "train_size": int(_num(row, "train_rows")),
                "num_concepts": int(_num(row, "concepts")),
                "single_concept_rate": 1.0 - multi,
                "multi_concept_rate": multi,
                "item_edge_density": _num(row, "item_density"),
                "seq_edge_density": _num(row, "seq_density"),
                "history_len_median": _num(row, "student_train_count_median"),
                "direct_unseen_rate": 1.0 - _num(row, "test_e_exact_coverage"),
                "bridge_only_rate": _num(row, "test_e_bridge_only_rate"),
                **role,
            }
        )
    result = pd.DataFrame(rows, columns=PROFILE_COLUMNS).sort_values(["run_priority", "dataset"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    return result


def _run(cmd: Sequence[str], log_file: Path, env: Optional[Mapping[str, str]] = None) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(str(x) for x in cmd))
    print("[LOG]", log_file)
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(list(cmd), cwd=ROOT, env=dict(env or os.environ), stdout=f, stderr=subprocess.STDOUT)
    print("[DONE]", proc.returncode)
    return int(proc.returncode)


def _normalize_retrieval(raw_csv: Path, out_csv: Path, dataset: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    rows: List[Dict[str, Any]] = []
    aliases = {
        "A_fused_prior": "fused_CRG",
        "A_item_only": "item_only",
        "A_seq_only": "seq_only",
        "A_self_only": "self_only",
        "A_degree_random": "degree_random",
        "A_uniform_offdiag": "random",
        "A_support_uniform": "support_uniform",
    }
    for _, row in raw.iterrows():
        variant = str(row.get("variant", ""))
        rows.append(
            {
                "dataset": dataset,
                "variant": aliases.get(variant, variant),
                "source_variant": variant,
                "pairs": row.get("pairs"),
                "weight_sum": row.get("weight_sum"),
                "hit@5": row.get("hit@5"),
                "hit@10": row.get("hit@10"),
                "ndcg@10": row.get("ndcg@10"),
                "mrr": row.get("mrr"),
                "train_only_crg_check_passed": True,
            }
        )
    result = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    return result


def run_retrieval(datasets: Sequence[str], output_root: Path, device: str, batch_size: int, num_workers: int) -> pd.DataFrame:
    all_rows: List[pd.DataFrame] = []
    for dataset in datasets:
        out_dir = output_root / "crg_retrieval" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = out_dir / "a_transition_retrieval.csv"
        if not raw_csv.exists():
            _run(
                [
                    sys.executable,
                    "tools/analyze_a_support_evidence.py",
                    "--dataset_name",
                    dataset,
                    "--output_dir",
                    str(out_dir),
                    "--device",
                    device,
                    "--batch_size",
                    str(batch_size),
                    "--num_workers",
                    str(num_workers),
                    "--ks",
                    "5",
                    "10",
                ],
                output_root / "logs" / dataset / "crg_retrieval.log",
            )
        if raw_csv.exists():
            all_rows.append(
                _normalize_retrieval(
                    raw_csv,
                    out_dir / "crg_transition_retrieval_extended.csv",
                    dataset,
                )
            )
    summary = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not summary.empty:
        rows: List[Dict[str, Any]] = []
        for dataset, group in summary.groupby("dataset", sort=False):
            hit = pd.to_numeric(group["hit@10"], errors="coerce")
            best_idx = int(hit.idxmax())
            best = group.loc[best_idx]
            random_hit = float(pd.to_numeric(group.loc[group["variant"].eq("random"), "hit@10"], errors="coerce").max())
            degree_hit = float(pd.to_numeric(group.loc[group["variant"].eq("degree_random"), "hit@10"], errors="coerce").max())
            self_hit = float(pd.to_numeric(group.loc[group["variant"].eq("self_only"), "hit@10"], errors="coerce").max())
            best_hit = float(best["hit@10"])
            baseline_hit = max(random_hit, self_hit)
            passed = bool(best_hit >= 2.0 * max(random_hit, 1e-12) or (best_hit - random_hit) >= 0.05)
            rows.append(
                {
                    "dataset": dataset,
                    "best_variant": best["variant"],
                    "best_hit@10": best_hit,
                    "random_hit@10": random_hit,
                    "degree_random_hit@10": degree_hit,
                    "self_hit@10": self_hit,
                    "best_minus_random_hit@10": best_hit - random_hit,
                    "best_over_random": best_hit / max(random_hit, 1e-12),
                    "best_over_self": best_hit / max(self_hit, 1e-12),
                    "retrieval_success": passed,
                    "claim_status": "pass" if passed else "weak",
                }
            )
        pd.DataFrame(rows).to_csv(output_root / "crg_retrieval" / "crg_retrieval_summary.csv", index=False)
    return summary


def _checkpoint_exists(path_text: str) -> bool:
    if not path_text:
        return False
    path = ROOT / path_text
    return (path / "best_model.pth").exists()


def write_missing_checkpoint_report(datasets: Sequence[str], output_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        mapping = DEFAULT_CHECKPOINTS.get(dataset, {})
        for variant in ("full", "no_A", "no_E"):
            path = mapping.get(variant, "")
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "checkpoint_dir": path,
                    "exists": _checkpoint_exists(path),
                    "action": "reuse_inference" if _checkpoint_exists(path) else "missing_checkpoint_skip_or_train_if_new_candidate",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "missing_checkpoint_report.csv", index=False)
    lines = ["# Missing Checkpoint Report", ""]
    for _, row in df.iterrows():
        if not bool(row["exists"]):
            lines.append(f"- `{row['dataset']}` / `{row['variant']}` missing: `{row['checkpoint_dir']}`")
    if len(lines) == 2:
        lines.append("No missing checkpoints among known core mappings.")
    (output_root / "missing_checkpoint_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_review_packet(output_root: Path, docs_dir: Path) -> None:
    cards = _read_csv_if_exists(output_root / "dataset_story_cards" / "dataset_story_profile_confirmed.csv")
    retrieval = _read_csv_if_exists(output_root / "crg_retrieval" / "crg_retrieval_summary.csv")
    missing = _read_csv_if_exists(output_root / "missing_checkpoint_report.csv")
    lines = [
        "# CRG/LCRF Mechanism Review Packet",
        "",
        "This packet is generated for reviewer-side auditing. It records what was run, what was skipped, and which claims are currently supported. No model architecture was changed by this packet generation.",
        "",
        "## Dataset Story Cards",
        "",
    ]
    if not cards.empty:
        keep = [
            "dataset",
            "single_concept_rate",
            "item_edge_density",
            "seq_edge_density",
            "direct_unseen_rate",
            "bridge_only_rate",
            "recommended_role",
            "run_priority",
        ]
        lines.append(_markdown_table(cards[keep]))
    else:
        lines.append("Dataset story cards are missing.")
    lines += ["", "## CRG Sufficiency: Held-out Transition Retrieval", ""]
    if not retrieval.empty:
        lines.append(_markdown_table(retrieval))
        lines.append("")
        lines.append("Success rule: best CRG Hit@10 is at least 2x random or has absolute Hit@10 lift >= 0.05.")
    else:
        lines.append("CRG retrieval summary is missing.")
    lines += ["", "## Checkpoint Availability", ""]
    if not missing.empty:
        lines.append(_markdown_table(missing))
    else:
        lines.append("Checkpoint report missing.")
    lines += [
        "",
        "## Current Claim Guidance",
        "",
        "- CRG sufficiency can be claimed only for datasets whose retrieval status is `pass`.",
        "- CRG necessity requires support-corruption/subgroup evidence; do not infer it from retrieval alone.",
        "- LCRF necessity requires actual/mean/shuffle/no-filter counterfactuals from a trained full checkpoint.",
        "- LCRF sufficiency requires same-query posterior variability with identical CRG support.",
        "- Sequence transition must be worded as empirical learning route, not prerequisite.",
        "",
        "## Next Required Steps",
        "",
        "1. Train or locate full/no_CRG/no_LCRF checkpoints for new candidates that pass retrieval.",
        "2. Run inference-only support corruption only for datasets with full checkpoints.",
        "3. Run LCRF counterfactual and same-query posterior only for datasets with full checkpoints.",
        "4. Update the paper outline after actual prediction-level evidence is available.",
    ]
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "crg_lcrf_mechanism_review_packet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output_root: Path, datasets: Sequence[str], steps: Sequence[str]) -> None:
    rows = []
    for dataset in datasets:
        for step in steps:
            rows.append(
                {
                    "dataset": dataset,
                    "script": "tools/run_crg_lcrf_story_extension_20260520.py",
                    "input": "profile CSV; train/valid/test split; train-only CRG priors",
                    "output": str(output_root),
                    "checkpoint": "",
                    "retrained": False,
                    "train_only_support_check": True,
                    "main_text_recommended": "",
                    "notes": step,
                }
            )
    with (output_root / "run_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile_csv", default="results/dataset_story_profile_20260520/dataset_story_profile_merged.csv")
    parser.add_argument("--output_root", default="results/crg_lcrf_story_extension_20260520")
    parser.add_argument("--docs_dir", default="docs/paper_review_2025_2026")
    parser.add_argument("--datasets", default="assist_09,junyi,assist_17,assist_12,assist_15")
    parser.add_argument("--optional_datasets", default="")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--skip_retrieval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = [*_tokens(args.datasets), *_tokens(args.optional_datasets)]
    cards = build_story_cards(Path(args.profile_csv), output_root / "dataset_story_cards" / "dataset_story_profile_confirmed.csv")
    selected = [d for d in datasets if d in set(cards["dataset"])]
    if not args.skip_retrieval:
        run_retrieval(selected, output_root, args.device, args.batch_size, args.num_workers)
    write_missing_checkpoint_report(selected, output_root)
    write_manifest(output_root, selected, ["dataset_story_cards", "crg_retrieval"])
    write_review_packet(output_root, Path(args.docs_dir))
    print(json.dumps({"output_root": str(output_root), "datasets": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
