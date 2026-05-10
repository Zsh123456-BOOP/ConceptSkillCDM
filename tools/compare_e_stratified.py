"""Compare full E against a matched control with stratified per-sample diagnostics.

This is a read-only analysis helper. It reuses ``tools/analyze_ae_errors.py``
to export aligned test samples from two checkpoints, then reports where the
student-conditioned local posterior helps or hurts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment_utils import compute_metrics  # noqa: E402
from tools.analyze_ae_errors import analyze  # noqa: E402


SIGNAL_SPECS: Tuple[Tuple[str, str], ...] = (
    ("e_local_mastery_abs", "e_local_mastery_logit"),
    ("e_query_mastery_abs", "e_query_mastery_logit"),
    ("e_graph_mastery_abs", "e_graph_mastery_logit"),
    ("e_query_recent_mastery_abs", "e_query_recent_mastery_logit"),
    ("e_graph_recent_mastery_abs", "e_graph_recent_mastery_logit"),
    ("ae_abs", "ae_logit_residual"),
    ("posterior_prior_abs", "ae_posterior_prior_logit"),
    ("posterior_theta_abs", "ae_posterior_theta_logit"),
    ("posterior_kl", "query_row_posterior_kl"),
    ("posterior_delta", "query_row_posterior_delta_abs"),
    ("personal_message", "query_row_personal_message_delta"),
)


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    metrics = compute_metrics(labels, (probs > 0.5).astype(np.float32), probs)
    return float(metrics["auc"])


def _safe_mean(values: pd.Series) -> Optional[float]:
    if values.empty:
        return None
    out = float(values.mean())
    return out if np.isfinite(out) else None


def _check_alignment(full: pd.DataFrame, control: pd.DataFrame) -> None:
    if len(full) != len(control):
        raise ValueError(f"sample count mismatch: full={len(full)}, control={len(control)}")
    keys = [col for col in ("stu_id", "exer_id", "label") if col in full.columns and col in control.columns]
    for key in keys:
        if not (full[key].to_numpy() == control[key].to_numpy()).all():
            raise ValueError(f"sample alignment mismatch at column {key!r}")


def _prob_bce(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    probs = np.clip(probs.astype(np.float64), eps, 1.0 - eps)
    labels = labels.astype(np.float64)
    return -(labels * np.log(probs) + (1.0 - labels) * np.log1p(-probs))


def _metrics_for(frame: pd.DataFrame) -> Dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.float32)
    full_probs = frame["full_prob"].to_numpy(dtype=np.float32)
    control_probs = frame["control_prob"].to_numpy(dtype=np.float32)
    full_pred = (full_probs > 0.5).astype(np.float32)
    control_pred = (control_probs > 0.5).astype(np.float32)
    full_auc = _safe_auc(labels, full_probs)
    control_auc = _safe_auc(labels, control_probs)
    metrics = {
        "n": int(len(frame)),
        "positive_rate": _safe_mean(frame["label"]),
        "full_auc": full_auc,
        "control_auc": control_auc,
        "auc_delta": None if full_auc is None or control_auc is None else float(full_auc - control_auc),
        "full_acc": float((full_pred == labels).mean()) if len(frame) else None,
        "control_acc": float((control_pred == labels).mean()) if len(frame) else None,
        "bce_delta_mean": _safe_mean(frame["bce_delta"]),
        "bce_delta_median": float(frame["bce_delta"].median()) if len(frame) else None,
        "full_help_rate": _safe_mean((frame["bce_delta"] < 0.0).astype(np.float32)),
        "full_hurt_rate": _safe_mean((frame["bce_delta"] > 0.0).astype(np.float32)),
        "abs_prob_delta_mean": _safe_mean(frame["prob_delta"].abs()),
        "full_ae_abs_mean": _safe_mean(frame["full_ae_logit_residual"].abs()),
    }
    for signal_name, _ in SIGNAL_SPECS:
        if signal_name in frame:
            metrics[f"{signal_name}_mean"] = _safe_mean(frame[signal_name])
    return metrics


def _add_stratum(rows: List[Dict[str, Any]], name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    item = {"stratum": name}
    item.update(_metrics_for(frame))
    rows.append(item)


def _build_stratified_comparison(full: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    _check_alignment(full, control)
    frame = pd.DataFrame(
        {
            "label": full["label"].astype(np.float32),
            "full_prob": full["prob"].astype(np.float32),
            "control_prob": control["prob"].astype(np.float32),
            "q_count": full.get("q_count", pd.Series(np.zeros(len(full), dtype=np.int32))).astype(np.int32),
            "is_multi_concept": full.get("is_multi_concept", pd.Series(np.zeros(len(full), dtype=np.int32))).astype(
                np.int32
            ),
        }
    )
    frame["full_bce"] = (
        full["bce_total"].astype(np.float64)
        if "bce_total" in full
        else _prob_bce(frame["label"].to_numpy(), frame["full_prob"].to_numpy())
    )
    frame["control_bce"] = (
        control["bce_total"].astype(np.float64)
        if "bce_total" in control
        else _prob_bce(frame["label"].to_numpy(), frame["control_prob"].to_numpy())
    )
    frame["bce_delta"] = frame["full_bce"] - frame["control_bce"]
    frame["prob_delta"] = frame["full_prob"] - frame["control_prob"]

    for out_col, source_col in SIGNAL_SPECS:
        if source_col in full.columns:
            values = full[source_col].astype(np.float32)
            if source_col not in {"query_row_posterior_kl", "query_row_posterior_delta_abs", "query_row_personal_message_delta"}:
                values = values.abs()
            frame[out_col] = values
        else:
            frame[out_col] = 0.0
    frame["full_ae_logit_residual"] = full.get("ae_logit_residual", pd.Series(np.zeros(len(full)))).astype(np.float32)

    rows: List[Dict[str, Any]] = []
    _add_stratum(rows, "all", frame)
    _add_stratum(rows, "single_concept", frame[frame["is_multi_concept"] == 0])
    _add_stratum(rows, "multi_concept", frame[frame["is_multi_concept"] == 1])
    for q_count, grp in frame.groupby("q_count"):
        _add_stratum(rows, f"q_count={int(q_count)}", grp)

    for signal_name, _ in SIGNAL_SPECS:
        signal = frame[signal_name].astype(np.float64)
        if signal.notna().sum() == 0 or float(signal.max()) <= 0.0:
            continue
        for pct, label in ((0.90, "top10"), (0.80, "top20"), (0.50, "top50")):
            threshold = float(signal.quantile(pct))
            mask = signal >= threshold
            _add_stratum(rows, f"signal:{signal_name}:{label}", frame[mask])
        low_threshold = float(signal.quantile(0.80))
        _add_stratum(rows, f"signal:{signal_name}:bottom80", frame[signal < low_threshold])

    return pd.DataFrame(rows)


def _load_or_export_samples(
    *,
    samples_path: Optional[str],
    save_dir: Optional[str],
    output_dir: Path,
    role: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if samples_path:
        return pd.read_csv(samples_path)
    if not save_dir:
        raise ValueError(f"Either --{role}_samples or --{role}_save_dir is required.")
    role_dir = output_dir / f"{role}_analysis"
    ns = SimpleNamespace(
        save_dir=save_dir,
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        output_dir=str(role_dir),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        top_k=args.top_k,
        device=args.device,
    )
    analyze(ns)
    return pd.read_csv(role_dir / "ae_error_samples.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare E full/control per-sample effects by interpretable strata.")
    parser.add_argument("--full_save_dir", default=None)
    parser.add_argument("--control_save_dir", default=None)
    parser.add_argument("--full_samples", default=None)
    parser.add_argument("--control_samples", default=None)
    parser.add_argument("--dataset_name", default="assist_09")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full = _load_or_export_samples(
        samples_path=args.full_samples,
        save_dir=args.full_save_dir,
        output_dir=output_dir,
        role="full",
        args=args,
    )
    control = _load_or_export_samples(
        samples_path=args.control_samples,
        save_dir=args.control_save_dir,
        output_dir=output_dir,
        role="control",
        args=args,
    )
    summary = _build_stratified_comparison(full, control)
    csv_path = output_dir / "e_stratified_comparison.csv"
    json_path = output_dir / "e_stratified_comparison.json"
    summary.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
