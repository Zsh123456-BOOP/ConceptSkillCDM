#!/usr/bin/env python
"""Plot the compact graph and IRT diagnostics exported by the trainer."""

from __future__ import annotations

import argparse
import os
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")


def plot_graph(data: Dict[str, np.ndarray], save_dir: str, max_concepts: int = 50) -> None:
    matrices = data.get("global_relation_matrices")
    if matrices is None:
        print("No global_relation_matrices found; graph plot skipped.")
        return
    matrices = np.asarray(matrices)
    heads = min(int(matrices.shape[0]), 4)
    concept_count = min(int(matrices.shape[-1]), int(max_concepts))
    figure, axes = plt.subplots(1, heads, figsize=(4 * heads, 4), squeeze=False)
    for head in range(heads):
        axis = axes[0, head]
        image = axis.imshow(
            matrices[head, :concept_count, :concept_count],
            cmap="Blues",
            aspect="auto",
            vmin=0.0,
        )
        axis.set_title(f"Graph head {head + 1}")
        axis.set_xlabel("Support concept")
        axis.set_ylabel("Receiving concept")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path = os.path.join(save_dir, "global_concept_graph.png")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"Saved: {path}")


def plot_irt(data: Dict[str, np.ndarray], save_dir: str) -> None:
    series = (
        ("irt_logit_samples", "IRT logits"),
        ("irt_discrimination_samples", "Exercise discrimination"),
        ("irt_difficulty_samples", "Exercise difficulty"),
    )
    available = [(key, title) for key, title in series if key in data]
    if not available:
        print("No IRT samples found; IRT plot skipped.")
        return
    figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4), squeeze=False)
    for axis, (key, title) in zip(axes[0], available):
        values = np.asarray(data[key], dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        axis.hist(values, bins=30, color="#4C78A8", alpha=0.85)
        axis.axvline(values.mean(), color="#E45756", linestyle="--", label=f"mean={values.mean():.3f}")
        axis.set_title(title)
        axis.legend()
    figure.tight_layout()
    path = os.path.join(save_dir, "irt_diagnostics.png")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Graph-IRT component diagnostics.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_concepts", type=int, default=50)
    args = parser.parse_args()

    data_path = os.path.join(args.checkpoint_dir, "component_analysis_data.npz")
    if not os.path.exists(data_path):
        raise SystemExit(f"Component data not found: {data_path}")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    output_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_graph(data, output_dir, max_concepts=args.max_concepts)
    plot_irt(data, output_dir)


if __name__ == "__main__":
    main()
