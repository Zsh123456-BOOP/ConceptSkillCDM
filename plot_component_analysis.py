#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_component_analysis.py
生成组件有效性验证可视化图表：
1. 全局概念图热力图
2. Gate Alpha 分布图
3. 个性化图差异对比

使用方法：
    python plot_component_analysis.py --checkpoint_dir checkpoints/junyi_best_gpd_base
    python plot_component_analysis.py --checkpoint_dir checkpoints/assist_09_best_gpd_base
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_global_graph(data: dict, save_dir: str, max_concepts: int = 50) -> None:
    """绘制全局概念图热力图"""
    if "global_relation_matrices" not in data:
        print("No global relation matrices found.")
        return

    matrices = data["global_relation_matrices"]
    num_heads, num_concepts, _ = matrices.shape

    if num_concepts > max_concepts:
        matrices = matrices[:, :max_concepts, :max_concepts]
        num_concepts = max_concepts

    fig, axes = plt.subplots(1, min(num_heads, 4), figsize=(4 * min(num_heads, 4), 4))
    if num_heads == 1:
        axes = [axes]

    for idx in range(min(num_heads, 4)):
        ax = axes[idx]
        im = ax.imshow(matrices[idx], cmap="Blues", aspect="auto")
        ax.set_title(f"Head {idx + 1}", fontsize=12)
        ax.set_xlabel("Target Concept")
        ax.set_ylabel("Source Concept")
        plt.colorbar(im, ax=ax)

    plt.suptitle(f"Learned Global Concept Graph (first {num_concepts} concepts)", fontsize=14)
    plt.tight_layout()
    output_path = os.path.join(save_dir, "global_concept_graph.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    sparsity = (matrices < 0.01).mean()
    print(f"Graph sparsity (entries < 0.01): {sparsity:.2%}")


def plot_personal_graph_analysis(data: dict, save_dir: str) -> None:
    """绘制个性化图分析"""
    if "gate_alpha" in data:
        alpha = data["gate_alpha"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(alpha, bins=30, color="coral", edgecolor="black", alpha=0.7)
        ax.axvline(alpha.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {alpha.mean():.3f}")
        ax.set_xlabel("Gate Alpha Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Personal Graph Mixing Coefficient Distribution")
        ax.legend()
        plt.tight_layout()
        output_path = os.path.join(save_dir, "alpha_distribution.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")

    if "personal_matrices_samples" in data:
        samples = data["personal_matrices_samples"]
        num_show = min(4, len(samples))

        fig, axes = plt.subplots(1, num_show, figsize=(4 * num_show, 4))
        if num_show == 1:
            axes = [axes]

        for idx in range(num_show):
            ax = axes[idx]
            mat = samples[idx][:50, :50] if samples[idx].shape[0] > 50 else samples[idx]
            ax.imshow(mat, cmap="Purples", aspect="auto")
            ax.set_title(f"Student {idx + 1}", fontsize=12)
            ax.set_xlabel("Target Concept")
            ax.set_ylabel("Source Concept")

        plt.suptitle("Personal Graph Samples (Different Students)", fontsize=14)
        plt.tight_layout()
        output_path = os.path.join(save_dir, "personal_graph_samples.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")

        if len(samples) >= 2:
            diffs = []
            for idx in range(len(samples) - 1):
                diffs.append(np.abs(samples[idx] - samples[idx + 1]).mean())
            print(f"Average difference between student graphs: {np.mean(diffs):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate component analysis visualizations")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory containing component_analysis_data.npz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as checkpoint_dir)",
    )
    args = parser.parse_args()

    data_path = os.path.join(args.checkpoint_dir, "component_analysis_data.npz")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        print("Please run training with component analysis enabled first.")
        return

    data = dict(np.load(data_path, allow_pickle=True))
    print(f"Loaded data from {data_path}")
    print(f"Available keys: {list(data.keys())}")

    output_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Generating Global Graph Analysis ===")
    plot_global_graph(data, output_dir)

    print("\n=== Generating Personal Graph Analysis ===")
    plot_personal_graph_analysis(data, output_dir)

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
