#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_component_analysis.py
生成组件有效性验证可视化图表：
1. Prototype 热力图和相似度矩阵
2. 全局概念图热力图
3. Gate Alpha 分布图
4. 个性化图差异对比

使用方法：
    python plot_component_analysis.py --checkpoint_dir checkpoints/junyi_best_gpd_base
    python plot_component_analysis.py --checkpoint_dir checkpoints/assist_09_best_gpd_base
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无显示器环境

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_prototype_analysis(data: dict, save_dir: str) -> None:
    """绘制 Prototype 分析图"""
    
    # 1) 原型相似度矩阵
    if "prototype_similarity" in data:
        sim = data["prototype_similarity"]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(sim, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax.set_title("Prototype Similarity Matrix", fontsize=14)
        ax.set_xlabel("Prototype Index")
        ax.set_ylabel("Prototype Index")
        
        # 添加数值标注
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=10)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "prototype_similarity.png"), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(save_dir, 'prototype_similarity.png')}")
    
    # 2) 原型分配分布热力图
    if "prototype_assign" in data:
        assign = data["prototype_assign"]  # (N, K)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 计算每个原型的平均分配概率
        mean_assign = assign.mean(axis=0)
        
        ax.bar(range(len(mean_assign)), mean_assign, color='steelblue', edgecolor='black')
        ax.set_xlabel("Prototype Index")
        ax.set_ylabel("Average Assignment Probability")
        ax.set_title("Prototype Usage Distribution")
        ax.set_xticks(range(len(mean_assign)))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "prototype_usage.png"), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(save_dir, 'prototype_usage.png')}")


def plot_global_graph(data: dict, save_dir: str, max_concepts: int = 50) -> None:
    """绘制全局概念图热力图"""
    
    if "global_relation_matrices" not in data:
        print("No global relation matrices found.")
        return
    
    matrices = data["global_relation_matrices"]  # (H, C, C)
    H, C, _ = matrices.shape
    
    # 如果概念太多，只显示部分
    if C > max_concepts:
        matrices = matrices[:, :max_concepts, :max_concepts]
        C = max_concepts
    
    fig, axes = plt.subplots(1, min(H, 4), figsize=(4 * min(H, 4), 4))
    if H == 1:
        axes = [axes]
    
    for h in range(min(H, 4)):
        ax = axes[h]
        im = ax.imshow(matrices[h], cmap='Blues', aspect='auto')
        ax.set_title(f"Head {h+1}", fontsize=12)
        ax.set_xlabel("Target Concept")
        ax.set_ylabel("Source Concept")
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(f"Learned Global Concept Graph (first {C} concepts)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "global_concept_graph.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'global_concept_graph.png')}")
    
    # 图稀疏度分析
    sparsity = (matrices < 0.01).mean()
    print(f"Graph sparsity (entries < 0.01): {sparsity:.2%}")


def plot_personal_graph_analysis(data: dict, save_dir: str) -> None:
    """绘制个性化图分析"""
    
    # 1) Gate Alpha 分布
    if "gate_alpha" in data:
        alpha = data["gate_alpha"]
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(alpha, bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(alpha.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {alpha.mean():.3f}')
        ax.set_xlabel("Gate Alpha Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Personal Graph Mixing Coefficient Distribution")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "alpha_distribution.png"), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(save_dir, 'alpha_distribution.png')}")
    
    # 2) 个性化图差异对比（选择几个样本）
    if "personal_matrices_samples" in data:
        samples = data["personal_matrices_samples"]  # (N, C, C)
        num_show = min(4, len(samples))
        
        fig, axes = plt.subplots(1, num_show, figsize=(4 * num_show, 4))
        if num_show == 1:
            axes = [axes]
        
        for i in range(num_show):
            ax = axes[i]
            # 只显示部分概念
            mat = samples[i][:50, :50] if samples[i].shape[0] > 50 else samples[i]
            im = ax.imshow(mat, cmap='Purples', aspect='auto')
            ax.set_title(f"Student {i+1}", fontsize=12)
            ax.set_xlabel("Target Concept")
            ax.set_ylabel("Source Concept")
        
        plt.suptitle("Personal Graph Samples (Different Students)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "personal_graph_samples.png"), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(save_dir, 'personal_graph_samples.png')}")
        
        # 计算学生间差异
        if len(samples) >= 2:
            diffs = []
            for i in range(len(samples) - 1):
                diff = np.abs(samples[i] - samples[i+1]).mean()
                diffs.append(diff)
            print(f"Average difference between student graphs: {np.mean(diffs):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate component analysis visualizations")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint directory containing component_analysis_data.npz")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: same as checkpoint_dir)")
    args = parser.parse_args()
    
    data_path = os.path.join(args.checkpoint_dir, "component_analysis_data.npz")
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        print("Please run training with component analysis enabled first.")
        return
    
    # 加载数据
    data = dict(np.load(data_path, allow_pickle=True))
    print(f"Loaded data from {data_path}")
    print(f"Available keys: {list(data.keys())}")
    
    # 输出目录
    output_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成可视化
    print("\n=== Generating Prototype Analysis ===")
    plot_prototype_analysis(data, output_dir)
    
    print("\n=== Generating Global Graph Analysis ===")
    plot_global_graph(data, output_dir)
    
    print("\n=== Generating Personal Graph Analysis ===")
    plot_personal_graph_analysis(data, output_dir)
    
    print(f"\n✅ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
