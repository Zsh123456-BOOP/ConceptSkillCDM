import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
from sklearn.manifold import TSNE

def load_skill_map(dataset_name, data_dir):
    """
    尝试加载知识点名称映射，用于图表显示。
    """
    skill_map = {}
    try:
        # 简单适配逻辑，根据你的实际文件结构修改
        base_path = os.path.join(data_dir, dataset_name)
        # 尝试寻找 content.csv 或 skill_builder.csv
        # 这里仅作示例，如果找不到默认返回空字典
        pass
    except Exception as e:
        print(f"[Warning] Could not load skill names: {e}")
    return skill_map

def run_structure_heatmap(model, save_dir, top_k=20):
    """
    方案三：结构发现 (Structure Discovery)
    可视化多头注意力矩阵的热力图
    """
    print("\n[Analysis] Running Structure Discovery (Heatmap)...")
    model.eval()
    
    with torch.no_grad():
        concept_emb = model.concept_emb_shared.weight
        # 获取关系矩阵 (H, C, C)
        # 注意：这里调用 forward 会返回堆叠后的矩阵
        rels = model.relation_learning(concept_emb) # (H, C, C)
        
        # 转换为 numpy
        rels_np = rels.cpu().numpy()
        num_heads = rels_np.shape[0]
        
    # 为了可视化清晰，只取前 top_k 个活跃的概念（或者随机取）
    # 这里简单起见，取前 k 个
    limit = min(top_k, rels_np.shape[1])
    slice_idx = np.arange(limit)
    
    # 绘制每个 Head 的热力图
    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 5))
    if num_heads == 1: axes = [axes]
    
    for h in range(num_heads):
        ax = axes[h]
        data = rels_np[h][slice_idx, :][:, slice_idx]
        
        # 使用 seaborn 绘制热力图
        sns.heatmap(data, ax=ax, cmap="viridis", cbar=True, square=True)
        ax.set_title(f"Head {h} (Structure View)")
        ax.set_xlabel("Target Concept")
        ax.set_ylabel("Source Concept")
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'analysis_structure_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Analysis] Structure Heatmap saved to: {save_path}")


def run_manifold_visualization(model, dataset, save_dir, device, max_students=1000):
    """
    方案二：流形可视化 (Manifold & Clustering Story)
    使用 t-SNE 可视化学生向量，并根据其所属的原型进行染色
    """
    print("\n[Analysis] Running Manifold Visualization (t-SNE)...")
    
    if model.prototype_module is None:
        print("[Analysis] No prototype module found. Skipping t-SNE.")
        return

    model.eval()
    
    student_reprs = []
    prototype_labels = []
    
    # 随机采样一批学生
    num_total = getattr(model, 'num_students', 100)
    indices = np.random.choice(num_total, min(num_total, max_students), replace=False)
    
    batch_size = 128
    with torch.no_grad():
        # 获取共享概念嵌入和关系图
        concept_emb = model.concept_emb_shared.weight
        rel_matrices = model.relation_learning(concept_emb)
        
        for i in range(0, len(indices), batch_size):
            batch_idx = torch.tensor(indices[i:i+batch_size], device=device)
            
            # 通过 Encoder 获取学生向量
            ks = model.knowledge_encoder(batch_idx, rel_matrices, concept_emb)
            repr_batch = ks.mean(dim=1) # (B, D)
            
            # 获取原型归属
            _, assign = model.prototype_module(repr_batch) # (B, K)
            labels = assign.argmax(dim=1) # Hard assignment for coloring
            
            student_reprs.append(repr_batch.cpu().numpy())
            prototype_labels.append(labels.cpu().numpy())
            
    # 合并数据
    X = np.concatenate(student_reprs, axis=0) # (N, D)
    y = np.concatenate(prototype_labels, axis=0) # (N,)
    
    # 获取原型本身的向量，也画上去作为地标
    proto_vecs = model.prototype_module.prototypes.detach().cpu().numpy() # (K, D)
    
    # 组合学生和原型一起做 t-SNE，保证在同一个空间
    X_combined = np.concatenate([X, proto_vecs], axis=0)
    
    # 运行 t-SNE
    # perplexity 需小于样本数
    perp = min(30, len(X_combined) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_combined)
    
    # 分离
    X_students = X_embedded[:len(X)]
    X_protos = X_embedded[len(X):]
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 画学生点
    scatter = plt.scatter(X_students[:, 0], X_students[:, 1], c=y, cmap='tab10', alpha=0.6, s=30, label='Students')
    
    # 画原型中心点 (用大星号表示)
    plt.scatter(X_protos[:, 0], X_protos[:, 1], c='black', marker='*', s=300, edgecolor='white', linewidth=2, label='Prototypes')
    
    # 添加图例
    handles, _ = scatter.legend_elements()
    legend_labels = [f"Proto {i}" for i in range(len(handles))]
    plt.legend(handles, legend_labels, title="Cognitive Patterns")
    
    plt.title("Student Cognitive Manifold (t-SNE Visualization)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    save_path = os.path.join(save_dir, 'analysis_manifold_tsne.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Analysis] Manifold t-SNE saved to: {save_path}")


def run_ccs_experiment(model, dataset, save_dir, device):
    """
    反事实认知模拟 (原有的个体分析)
    """
    print("\n[Analysis] Running CCS (Counterfactual Cognitive Simulation)...")
    model.eval()
    
    target_student_idx = 0 
    if hasattr(model, 'num_students') and model.num_students > 0:
        target_student_idx = model.num_students // 2
    
    with torch.no_grad():
        sid = torch.tensor([target_student_idx], device=device)
        concept_emb = model.concept_emb_shared.weight
        rel_matrices = model.relation_learning(concept_emb)
        
        ks_nat = model.knowledge_encoder(sid, rel_matrices, concept_emb)
        repr_nat = ks_nat.mean(dim=1)
        
        if model.prototype_module is None:
            return

        mix_nat, assign_nat = model.prototype_module(repr_nat)
        original_proto = assign_nat.argmax().item()
        original_conf = assign_nat.max().item()
        
        mix_nat_broadcast = mix_nat.unsqueeze(1).expand(-1, model.num_concepts, -1)
        ks_final_nat = (1 - model.proto_lambda) * ks_nat + model.proto_lambda * mix_nat_broadcast
        theta_nat = model.prediction_head.theta_proj(ks_final_nat).squeeze(-1)
        mastery_nat = torch.sigmoid(theta_nat).squeeze(0).cpu().numpy()

        all_protos = model.prototype_module.prototypes
        dists = torch.norm(all_protos - all_protos[original_proto], dim=1)
        target_proto = dists.argmax().item()
        
        target_vec = all_protos[target_proto].unsqueeze(0)
        target_broadcast = target_vec.unsqueeze(1).expand(-1, model.num_concepts, -1)
        
        ks_final_counter = (1 - model.proto_lambda) * ks_nat + model.proto_lambda * target_broadcast
        theta_counter = model.prediction_head.theta_proj(ks_final_counter).squeeze(-1)
        mastery_counter = torch.sigmoid(theta_counter).squeeze(0).cpu().numpy()

    delta = mastery_counter - mastery_nat
    top_indices = np.argsort(np.abs(delta))[-15:]
    labels = [f"Concept {i}" for i in top_indices]
    values = delta[top_indices]
    colors = ['#ff9999' if v > 0 else '#66b3ff' for v in values]
    
    plt.figure(figsize=(10, 8))
    plt.barh(labels, values, color=colors)
    plt.axvline(0, color='grey', linewidth=0.8)
    plt.title(f"CCS Experiment: Student {target_student_idx}\nSwitching from Proto-{original_proto} (conf:{original_conf:.2f}) to Proto-{target_proto}")
    plt.xlabel("Change in Mastery Probability (Delta)")
    
    save_path = os.path.join(save_dir, 'analysis_ccs_result.png')
    plt.savefig(save_path)
    plt.close()
    
    # 顺便存JSON
    result_data = {
        "student_id": target_student_idx,
        "original_proto": original_proto,
        "counterfactual_proto": target_proto,
        "top_changes": {f"Concept_{idx}": float(val) for idx, val in zip(top_indices, values)}
    }
    with open(os.path.join(save_dir, 'analysis_ccs_data.json'), 'w') as f:
        json.dump(result_data, f, indent=4)
    
    print(f"[Analysis] CCS Result saved to: {save_path}")