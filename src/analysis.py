import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_skill_map(dataset_name, data_dir):
    """
    尝试加载知识点名称映射，用于图表显示。
    """
    skill_map = {}
    try:
        _ = os.path.join(data_dir, dataset_name)
    except Exception as e:
        print(f"[Warning] Could not load skill names: {e}")
    return skill_map


def run_structure_heatmap(analysis_data: Dict[str, np.ndarray], save_dir: str, top_k: int = 20) -> None:
    """
    可视化全局概念图的多头邻接热力图。
    """
    relation_matrices = analysis_data.get("global_relation_matrices")
    if relation_matrices is None:
        print("[Analysis] No global relation matrices found. Skipping structure heatmap.")
        return

    rels_np = np.asarray(relation_matrices)
    num_heads = rels_np.shape[0]
    limit = min(int(top_k), rels_np.shape[1])
    slice_idx = np.arange(limit)

    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 5))
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        ax = axes[h]
        data = rels_np[h][slice_idx, :][:, slice_idx]
        sns.heatmap(data, ax=ax, cmap="viridis", cbar=True, square=True)
        ax.set_title(f"Head {h} (Structure View)")
        ax.set_xlabel("Target Concept")
        ax.set_ylabel("Source Concept")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "analysis_structure_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Analysis] Structure Heatmap saved to: {save_path}")


def run_personal_gate_analysis(analysis_data: Dict[str, np.ndarray], save_dir: str) -> None:
    """
    可视化个性化 gate alpha 分布与个性化图样本。
    """
    gate_alpha = analysis_data.get("gate_alpha")
    personal_samples = analysis_data.get("personal_matrices_samples")

    if gate_alpha is None and personal_samples is None:
        print("[Analysis] No personal-graph analysis data found. Skipping.")
        return

    os.makedirs(save_dir, exist_ok=True)

    if gate_alpha is not None:
        gate_alpha = np.asarray(gate_alpha).reshape(-1)
        plt.figure(figsize=(8, 4))
        plt.hist(gate_alpha, bins=30, color="#3b82f6", alpha=0.85)
        plt.title("Personal Gate Alpha Distribution")
        plt.xlabel("alpha")
        plt.ylabel("count")
        plt.tight_layout()
        save_path = os.path.join(save_dir, "analysis_personal_gate_hist.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Analysis] Personal gate histogram saved to: {save_path}")

        stats = {
            "alpha_mean": float(gate_alpha.mean()),
            "alpha_std": float(gate_alpha.std()),
            "alpha_min": float(gate_alpha.min()),
            "alpha_max": float(gate_alpha.max()),
        }
        with open(os.path.join(save_dir, "analysis_personal_gate_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

    if personal_samples is not None:
        personal_samples = np.asarray(personal_samples)
        num_plots = min(4, personal_samples.shape[0])
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
        if num_plots == 1:
            axes = [axes]
        for idx in range(num_plots):
            sns.heatmap(personal_samples[idx], ax=axes[idx], cmap="magma", cbar=True, square=True)
            axes[idx].set_title(f"Personal Graph #{idx}")
            axes[idx].set_xlabel("Target Concept")
            axes[idx].set_ylabel("Source Concept")
        plt.tight_layout()
        save_path = os.path.join(save_dir, "analysis_personal_graph_samples.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Analysis] Personal graph samples saved to: {save_path}")
