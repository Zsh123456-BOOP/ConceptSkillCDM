import os
import sys
import math
import logging
import argparse
import csv
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

# ==========================================
# 0. 工具函数 (日志、保存、随机种子)
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logger(log_dir, name, dataset_name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f"{name}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(f'%(asctime)s [{dataset_name}] %(message)s')
    
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def save_result_csv(args, metrics, best_epoch):
    res_path = os.path.join(args.log_dir, "results_v3_fixed_bug.csv")
    file_exists = os.path.exists(res_path)
    
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model_name,
        "dataset": args.dataset,
        "test_auc": f"{metrics['auc']:.4f}",
        "test_acc": f"{metrics['acc']:.4f}",
        "test_rmse": f"{metrics['rmse']:.4f}",
        "best_epoch": best_epoch,
        "dim_stu": args.dim_student,
        "lr": args.lr,
        "dropout": args.dropout,
        "note": args.note
    }
    
    with open(res_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ==========================================
# 1. 数据处理
# ==========================================
class InteractionDataset(Dataset):
    def __init__(self, df, s2i, i2i, c2i):
        self.records = []
        for _, row in df.iterrows():
            s = s2i[int(row["student_id"])]
            i = i2i[int(row["item_id"])]
            c_str = str(row["concept_ids"]).replace('"', "").replace("'", "")
            c_list = [c2i[int(x)] for x in c_str.replace(",", ";").split(";") if x.strip()]
            label = float(row["correct"])
            self.records.append((s, i, torch.tensor(c_list, dtype=torch.long), torch.tensor(label, dtype=torch.float)))

    def __len__(self): return len(self.records)
    def __getitem__(self, idx): return self.records[idx]

def collate_fn(batch):
    s_ids = torch.tensor([b[0] for b in batch], dtype=torch.long)
    i_ids = torch.tensor([b[1] for b in batch], dtype=torch.long)
    labels = torch.stack([b[3] for b in batch])
    c_ptr = [0]
    all_c = []
    for b in batch:
        c_ptr.append(c_ptr[-1] + len(b[2]))
        all_c.extend(b[2].tolist())
    
    return {
        "student_ids": s_ids,
        "item_ids": i_ids,
        "concept_ids": torch.tensor(all_c, dtype=torch.long),
        "concept_ptr": torch.tensor(c_ptr, dtype=torch.long),
        "labels": labels
    }

def load_data(dataset_name, data_dir="data"):
    base_path = os.path.join(data_dir, dataset_name)
    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    valid_df = pd.read_csv(os.path.join(base_path, "valid.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "test.csv"))
    
    for df in [train_df, valid_df, test_df]:
        rename_map = {"stu_id": "student_id", "exer_id": "item_id", "label": "correct", "cpt_seq": "concept_ids"}
        df.rename(columns=rename_map, inplace=True)

    students, items, concepts = set(), set(), set()
    for df in [train_df, valid_df, test_df]:
        students.update(df["student_id"].astype(int).unique())
        items.update(df["item_id"].astype(int).unique())
        for c_str in df["concept_ids"]:
            clean = str(c_str).replace('"', "").replace("'", "").replace(",", ";")
            concepts.update([int(x) for x in clean.split(";") if x.strip()])
            
    s2i = {sid: i for i, sid in enumerate(sorted(students))}
    i2i = {iid: i for i, iid in enumerate(sorted(items))}
    c2i = {cid: i for i, cid in enumerate(sorted(concepts))}
    
    return (
        InteractionDataset(train_df, s2i, i2i, c2i),
        InteractionDataset(valid_df, s2i, i2i, c2i),
        InteractionDataset(test_df, s2i, i2i, c2i),
        {"num_students": len(s2i), "num_items": len(i2i), "num_concepts": len(c2i)}
    )

# ==========================================
# 2. 核心模型 (FullCDM V3.2 - 数值稳定版)
# ==========================================
class FullCDM(nn.Module):
    def __init__(self, num_students, num_items, num_concepts, args):
        super().__init__()
        self.args = args
        self.num_concepts = num_concepts
        
        # --- Embedding ---
        self.student_emb = nn.Embedding(num_students, args.dim_student)
        self.item_emb = nn.Embedding(num_items, args.dim_item)
        self.concept_emb = nn.Embedding(num_concepts, args.dim_concept)
        
        self.ln_s = nn.LayerNorm(args.dim_student)
        self.ln_i = nn.LayerNorm(args.dim_item)
        self.ln_c = nn.LayerNorm(args.dim_concept)
        self.dropout = nn.Dropout(args.dropout)
        
        nn.init.xavier_normal_(self.student_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.concept_emb.weight)

        # --- Module 1: Graph Discovery ---
        self.head_dim = args.dim_concept // 2
        self.W_Q = nn.Linear(args.dim_concept, args.dim_concept)
        self.W_K = nn.Linear(args.dim_concept, args.dim_concept)
        # 初始化稍大一点，防止一开始就除以极小值
        self.temp = nn.Parameter(torch.tensor(1.0)) 

        # --- Module 2: Disentanglement ---
        self.know_proj = nn.Sequential(
            nn.Linear(args.dim_student, args.dim_hidden),
            nn.LayerNorm(args.dim_hidden),
            nn.ReLU(),
            nn.Linear(args.dim_hidden, num_concepts)
        )
        self.skill_proj = nn.Sequential(
            nn.Linear(args.dim_student, args.dim_hidden),
            nn.ReLU(),
            nn.Linear(args.dim_hidden, args.dim_skill)
        )
        self.mastery_scale = nn.Parameter(torch.tensor(1.0))

        # --- Module 3: Item-aware aggregation & Prediction Heads ---
        # Item-Concept attention: 让每道题自己学“哪些知识点更关键”
        self.item_concept_mlp = nn.Sequential(
            nn.Linear(args.dim_item + args.dim_concept, args.dim_hidden),
            nn.Tanh(),
            nn.Linear(args.dim_hidden, 1),
        )
        # soft-min 的温度（越大越接近硬 min）
        self.softmin_beta = getattr(args, "softmin_beta", 5.0)

        self.guess_head = nn.Linear(args.dim_skill, 1)
        self.slip_head = nn.Linear(args.dim_skill, 1)

    def get_graph(self):
        E_c = self.ln_c(self.concept_emb.weight)
        N, D = E_c.shape
        
        Q = self.W_Q(E_c).view(N, 2, self.head_dim).transpose(0, 1)
        K = self.W_K(E_c).view(N, 2, self.head_dim).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        mask = torch.eye(N, device=E_c.device).bool().unsqueeze(0).expand(2, -1, -1)
        scores.masked_fill_(mask, -1e9)
        
        # [Bug修复 1] 保证 temp > 0.01，防止除以 0
        safe_temp = F.softplus(self.temp) + 0.01
        
        # [Bug修复 2] 钳位 scores，防止 exp 溢出
        scores = torch.clamp(scores / safe_temp, min=-10, max=10)
        
        A = torch.softmax(scores, dim=-1)
        return A 

    def forward(self, batch):
        s_ids = batch["student_ids"]
        i_ids = batch["item_ids"]
        c_ids = batch["concept_ids"]
        c_ptr = batch["concept_ptr"]
        
        e_s = self.ln_s(self.dropout(self.student_emb(s_ids)))
        e_i = self.ln_i(self.dropout(self.item_emb(i_ids)))
        
        # 1. Graph Learning
        adj = self.get_graph()
        adj_dag = adj[0] # Head 0: DAG
        
        # 2. Disentanglement & Propagation
        z_k_init = torch.sigmoid(self.know_proj(e_s))
        
        k = self.args.graph_topk
        topk_val, topk_idx = torch.topk(adj_dag, k=k, dim=-1)
        mask = torch.zeros_like(adj_dag)
        mask.scatter_(-1, topk_idx, 1.0)
        
        if self.training and self.args.graph_dropout > 0:
            mask = F.dropout(mask, p=self.args.graph_dropout, training=True)
            
        adj_prop = adj_dag * mask
        adj_prop = adj_prop / (adj_prop.sum(dim=-1, keepdim=True) + 1e-8)
        
        # [逻辑统一] 前置 -> 后继，不转置
        z_k_prop = torch.matmul(z_k_init, adj_prop) * self.mastery_scale
        z_k_final = z_k_init + z_k_prop
        
        z_skill = self.skill_proj(e_s)
        
        # 3. Prediction: item-aware soft-min aggregation over concepts
        batch_k_score = []
        beta = self.softmin_beta

        for b in range(len(s_ids)):
            start, end = c_ptr[b], c_ptr[b+1]
            if start == end:
                # 没有概念标注的题目，用中性默认值
                batch_k_score.append(torch.tensor(0.5, device=e_s.device))
                continue

            concepts = c_ids[start:end]                # [L]
            vals = z_k_final[b, concepts]             # [L] 学生在这些知识点上的掌握度

            # Item-Concept attention：对“当前题目”的每个相关知识点打权重
            item_vec = e_i[b].unsqueeze(0).expand(len(concepts), -1)        # [L, dim_item]
            concept_vecs = self.ln_c(self.concept_emb(concepts))            # [L, dim_concept]
            att_input = torch.cat([item_vec, concept_vecs], dim=-1)         # [L, dim_item+dim_concept]
            logits = self.item_concept_mlp(att_input).squeeze(-1)           # [L]
            weights = F.softmax(logits, dim=-1)                             # [L], sum=1

            # Weighted soft-min:
            # x_i = -beta * vals_i
            x = -beta * vals
            x_max = torch.max(x)
            # softmin = -1/beta * log( sum_i w_i * exp(x_i) )
            softmin_val = -(1.0 / beta) * (
                torch.log(torch.sum(weights * torch.exp(x - x_max)) + 1e-8) + x_max
            )
            batch_k_score.append(softmin_val)

        batch_k_score = torch.stack(batch_k_score)
        
        guess = torch.sigmoid(self.guess_head(z_skill)).squeeze(-1) * 0.3
        slip = torch.sigmoid(self.slip_head(z_skill)).squeeze(-1) * 0.3
        
        preds = batch_k_score + guess - slip
        return preds, adj, z_k_init, z_skill


class HybridKTCDM(nn.Module):
    """
    两路融合 + 图与解耦（增强版）：
    - 分支1：序列注意力分支（同一学生的交互序列，masked self-attention）
      事件表征升级为: [item_emb, concept_repr, graph-aware 知识向量, skill 向量]
      => 相当于：序列 KT + 图先验 + 解耦技巧 的联合表征
    - 分支2：静态诊断分支（知识向量 + 技巧向量 + item bias）
    - 图模块：在概念 embedding 上做两头注意力图 (head0=前置/DAG, head1=相似/Sym)
    - 知识传播：用 head0 图对 z_k_init 做前置->后继传播，得到 z_k_final 参与预测
    - 输出：fusion MLP(p_seq, k_diag(z_k_final), skill_score, item_bias, len_feat)
    - 返回：preds, adj, z_k_init, z_skill 供 loss_function 使用（图正则 + Orth）
    """
    def __init__(self, num_students, num_items, num_concepts, args):
        super().__init__()
        self.args = args
        self.num_concepts = num_concepts

        d_s = args.dim_student
        d_i = args.dim_item
        d_c = args.dim_concept
        d_h = args.dim_hidden
        d_skill = args.dim_skill

        # --------- 基础 Embedding 层 ---------
        self.student_emb = nn.Embedding(num_students, d_s)
        self.item_emb = nn.Embedding(num_items, d_i)
        self.concept_emb = nn.Embedding(num_concepts, d_c)

        self.ln_s = nn.LayerNorm(d_s)
        self.ln_i = nn.LayerNorm(d_i)
        self.ln_c = nn.LayerNorm(d_c)
        self.dropout = nn.Dropout(args.dropout)

        nn.init.xavier_normal_(self.student_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.concept_emb.weight)

        # ========= 图模块：概念关系发现（两头：DAG + 相似） =========
        # 单独的一套 graph heads（不要和序列分支的 num_heads 混用）
        self.graph_heads = 2
        self.graph_head_dim = d_c // self.graph_heads

        self.W_Q = nn.Linear(d_c, d_c)
        self.W_K = nn.Linear(d_c, d_c)
        # 温度，softplus 保证 >0
        self.temp = nn.Parameter(torch.tensor(1.0))
        # 知识传播缩放因子
        self.mastery_scale = nn.Parameter(torch.tensor(1.0))

        # ========= 分支2：静态诊断（知识 + 技巧） =========
        # Student -> knowledge over all concepts
        self.know_proj = nn.Sequential(
            nn.Linear(d_s, d_h),
            nn.ReLU(),
            nn.Linear(d_h, num_concepts)
        )

        # Student -> skill vector
        self.skill_proj = nn.Sequential(
            nn.Linear(d_s, d_skill),
            nn.ReLU()
        )
        self.skill_score_head = nn.Linear(d_skill, 1)

        # Item bias
        self.item_bias = nn.Embedding(num_items, 1)

        # ========= 分支1：序列注意力分支 =========
        # ### MOD: 事件表征从 (d_i + d_c) 升级为 (d_i + d_c + d_c + d_skill)
        # event 表示 = [item_emb, concept_repr, graph-aware concept-knowledge 向量, skill 向量]
        self.ev_proj = nn.Linear(d_i + d_c + d_c + d_skill, d_h)

        self.num_heads = 4
        self.head_dim = d_h // self.num_heads
        assert self.head_dim * self.num_heads == d_h, "dim_hidden 必须能被 num_heads 整除"

        self.W_q = nn.Linear(d_h, d_h)
        self.W_k = nn.Linear(d_h, d_h)
        self.W_v = nn.Linear(d_h, d_h)

        self.seq_ln = nn.LayerNorm(d_h)

        # 序列分支预测头： [h_seq, e_i, c_repr] -> p_seq
        self.seq_pred_mlp = nn.Sequential(
            nn.Linear(d_h + d_i + d_c, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 1),
            nn.Sigmoid()
        )

        # ========= 融合层 =========
        # 输入特征: [p_seq, k_diag(z_k_final), skill_score, item_bias, len_feat]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    # ===== 图构建：与 FullCDM 同逻辑 =====
    def get_graph(self):
        """
        返回 adj: [2, N, N]
        head0: DAG/前置关系
        head1: 相似关系（对称性正则）
        """
        E_c = self.ln_c(self.concept_emb.weight)   # [N, d_c]
        N, D = E_c.shape

        H = self.graph_heads
        Dh = self.graph_head_dim

        Q = self.W_Q(E_c).view(N, H, Dh).transpose(0, 1)   # [H, N, Dh]
        K = self.W_K(E_c).view(N, H, Dh).transpose(0, 1)   # [H, N, Dh]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)  # [H, N, N]

        # 屏蔽对角线：防止学成自环
        mask = torch.eye(N, device=E_c.device).bool().unsqueeze(0).expand(H, -1, -1)
        scores.masked_fill_(mask, -1e9)

        # 温度 + 数值稳定
        safe_temp = F.softplus(self.temp) + 0.01
        scores = torch.clamp(scores / safe_temp, min=-10, max=10)

        A = torch.softmax(scores, dim=-1)  # [H, N, N]
        return A

    def forward(self, batch):
        """
        batch:
            student_ids: [B]
            item_ids:    [B]
            concept_ids: [L_total]
            concept_ptr: [B+1]
        """
        device = batch["student_ids"].device

        s_ids = batch["student_ids"]   # [B]
        i_ids = batch["item_ids"]      # [B]
        c_ids = batch["concept_ids"]   # [L_total]
        c_ptr = batch["concept_ptr"]   # [B+1]

        B = s_ids.size(0)

        # --------- 基础 Embedding ---------
        e_s = self.ln_s(self.dropout(self.student_emb(s_ids)))  # [B, d_s]
        e_i = self.ln_i(self.dropout(self.item_emb(i_ids)))     # [B, d_i]

        # ====== 图 + 解耦部分 ======
        # 1) 图结构 (概念级，数据集级别常量)
        adj = self.get_graph()           # [2, N, N]
        adj_dag = adj[0]                # head0: 前置/DAG

        # 2) 学生知识 & 技巧（解耦）
        z_k_init = torch.sigmoid(self.know_proj(e_s))   # [B, N_concepts]
        z_skill = self.skill_proj(e_s)                  # [B, d_skill]

        # 3) 基于前置图做传播：z_k_final = z_k_init + z_k_init @ G
        k = self.args.graph_topk
        topk_val, topk_idx = torch.topk(adj_dag, k=k, dim=-1)   # 按行 top-k
        mask = torch.zeros_like(adj_dag)
        mask.scatter_(-1, topk_idx, 1.0)

        if self.training and self.args.graph_dropout > 0:
            mask = F.dropout(mask, p=self.args.graph_dropout, training=True)

        adj_prop = adj_dag * mask
        adj_prop = adj_prop / (adj_prop.sum(dim=-1, keepdim=True) + 1e-8)  # 行归一化

        # 前置 -> 后继（不转置）
        z_k_prop = torch.matmul(z_k_init, adj_prop) * self.mastery_scale   # [B, N_concepts]
        z_k_final = z_k_init + z_k_prop                                    # [B, N_concepts]

        # ====== 每个样本的概念 embedding & 知识向量（用于序列分支事件表征） ======
        concept_vecs = self.concept_emb(c_ids)                  # [L_total, d_c]

        concept_repr_list = []
        know_vec_list = []   # ### NEW: graph-aware 知识向量 in concept space

        for b in range(B):
            start, end = c_ptr[b], c_ptr[b+1]
            if start == end:
                concept_repr_list.append(torch.zeros(self.args.dim_concept, device=device))
                know_vec_list.append(torch.zeros(self.args.dim_concept, device=device))
            else:
                vecs_b = concept_vecs[start:end]        # [L_b, d_c]
                concept_repr_list.append(vecs_b.mean(dim=0))

                concepts_b = c_ids[start:end]           # [L_b]
                mastery_b = z_k_final[b, concepts_b]    # [L_b]

                # 使用掌握度作为权重，构造该题的“知识向量”
                w = mastery_b.clamp(min=0.0)
                if float(w.sum()) < 1e-6:
                    # 极端情况：全 0，则退化为平均
                    w = torch.ones_like(w) / w.numel()
                else:
                    w = w / (w.sum() + 1e-8)

                know_vec_b = (w.unsqueeze(-1) * vecs_b).sum(dim=0)  # [d_c]
                know_vec_list.append(know_vec_b)

        c_repr = torch.stack(concept_repr_list, dim=0)          # [B, d_c]
        c_repr = self.ln_c(c_repr)

        know_vec = torch.stack(know_vec_list, dim=0)            # [B, d_c]
        know_vec = self.ln_c(know_vec)                          # [B, d_c] 复用同一个 LN

        # ====== 静态诊断分支中的 k_diag 用 z_k_final ======
        k_diag_list = []
        for b in range(B):
            start, end = c_ptr[b], c_ptr[b+1]
            if start == end:
                k_diag_list.append(torch.tensor(0.5, device=device))
            else:
                concepts_b = c_ids[start:end]
                vals = z_k_final[b, concepts_b]                  # [L_b]
                k_diag_list.append(vals.mean())
        k_diag = torch.stack(k_diag_list, dim=0)                # [B]

        # 技巧得分 + item bias
        skill_score = torch.sigmoid(self.skill_score_head(z_skill)).squeeze(-1)  # [B]
        item_b = self.item_bias(i_ids).squeeze(-1)                               # [B]

        # ========= 分支1：序列注意力分支 =========
        # ### MOD: event 表示 = [e_i, c_repr, know_vec, z_skill]
        ev_feat = torch.cat([e_i, c_repr, know_vec, z_skill], dim=-1)  # [B, d_i + d_c + d_c + d_skill]
        ev = self.ev_proj(ev_feat)                                     # [B, d_h]

        H = self.num_heads
        D = self.head_dim

        Q = self.W_q(ev).view(B, H, D).transpose(0, 1)          # [H, B, D]
        K = self.W_k(ev).view(B, H, D).transpose(0, 1)          # [H, B, D]
        V = self.W_v(ev).view(B, H, D).transpose(0, 1)          # [H, B, D]

        # scores: [H, B, B]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

        # 构造 mask：只看同一个学生 & 只看自身及之前 (causal)
        same_stu = (s_ids.unsqueeze(0) == s_ids.unsqueeze(1))   # [B, B]
        causal = torch.tril(torch.ones(B, B, dtype=torch.bool, device=device), diagonal=0)
        attn_mask = same_stu & causal                           # [B, B]

        scores = scores.masked_fill(~attn_mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)                    # [H, B, B]

        # [H, B, B] @ [H, B, D] -> [H, B, D]
        h_hist = torch.matmul(attn, V)
        h_hist = h_hist.transpose(0, 1).contiguous().view(B, H * D)  # [B, d_h]
        h_hist = self.seq_ln(h_hist)

        # 序列分支预测： [h_hist, e_i, c_repr] -> p_seq
        seq_feat = torch.cat([h_hist, e_i, c_repr], dim=-1)     # [B, d_h + d_i + d_c]
        p_seq = self.seq_pred_mlp(seq_feat).squeeze(-1)         # [B]

        # ========= 融合特征 =========
        # 交互长度特征：该学生在当前 batch 内的“第几次做题”（log 缩放）
        len_feat = torch.zeros(B, device=device)
        for b in range(B):
            len_feat[b] = (s_ids[:b+1] == s_ids[b]).sum()
        len_feat = torch.log1p(len_feat) / 5.0                  # 粗略归一化到 [0,1) 附近

        fusion_in = torch.stack(
            [p_seq, k_diag, skill_score, item_b, len_feat],
            dim=-1                                             # [B, 5]
        )
        preds = self.fusion_mlp(fusion_in).squeeze(-1)          # [B]

        # 关键：这里返回 adj 而不是 None，让图正则真正上线
        # 为了 Orth：继续返回 z_k_init 和 z_skill（保持你原先 loss 的接口）
        return preds, adj, z_k_init, z_skill


class PlainCDM(nn.Module):
    """
    性能优先的简化版：
    - 不用图 (Graph)；
    - 保留学生知识/技巧解耦；
    - 题目感知的加权 mean 聚合；
    - 用一个小 MLP 预测头。
    """
    def __init__(self, num_students, num_items, num_concepts, args):
        super().__init__()
        self.args = args
        self.num_concepts = num_concepts

        # Embeddings
        self.student_emb = nn.Embedding(num_students, args.dim_student)
        self.item_emb = nn.Embedding(num_items, args.dim_item)
        self.concept_emb = nn.Embedding(num_concepts, args.dim_concept)

        self.ln_s = nn.LayerNorm(args.dim_student)
        self.ln_i = nn.LayerNorm(args.dim_item)
        self.ln_c = nn.LayerNorm(args.dim_concept)
        self.dropout = nn.Dropout(args.dropout)

        nn.init.xavier_normal_(self.student_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.concept_emb.weight)

        # 知识投影：Student -> [B, N_concepts]
        self.know_proj = nn.Sequential(
            nn.Linear(args.dim_student, args.dim_hidden),
            nn.ReLU(),
            nn.Linear(args.dim_hidden, num_concepts)
        )

        # 技巧向量：Student -> [B, dim_skill]
        self.skill_proj = nn.Sequential(
            nn.Linear(args.dim_student, args.dim_skill),
            nn.ReLU()
        )

        # Item-Concept attention 用来做加权 mean（而不是 soft-min）
        self.item_concept_mlp = nn.Sequential(
            nn.Linear(args.dim_item + args.dim_concept, args.dim_hidden),
            nn.Tanh(),
            nn.Linear(args.dim_hidden, 1),
        )

        # 技巧得分头
        self.skill_score_head = nn.Linear(args.dim_skill, 1)

        # 最终预测头：整合 [k_score, skill_score, item_bias]
        self.pred_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # item 偏置
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, batch):
        s_ids = batch["student_ids"]   # [B]
        i_ids = batch["item_ids"]      # [B]
        c_ids = batch["concept_ids"]   # [L_total]
        c_ptr = batch["concept_ptr"]   # [B+1]

        e_s = self.ln_s(self.dropout(self.student_emb(s_ids)))   # [B, d_s]
        e_i = self.ln_i(self.dropout(self.item_emb(i_ids)))      # [B, d_i]

        # 1. 知识 & 技巧
        z_k_init = torch.sigmoid(self.know_proj(e_s))            # [B, N_concepts]
        z_skill = self.skill_proj(e_s)                           # [B, d_skill]

        # 2. item-aware 加权 mean 聚合
        batch_k_score = []
        for b in range(len(s_ids)):
            start, end = c_ptr[b], c_ptr[b+1]
            if start == end:
                batch_k_score.append(torch.tensor(0.5, device=e_s.device))
                continue

            concepts = c_ids[start:end]                          # [L]
            vals = z_k_init[b, concepts]                         # [L]

            item_vec = e_i[b].unsqueeze(0).expand(len(concepts), -1)    # [L, d_i]
            concept_vecs = self.ln_c(self.concept_emb(concepts))        # [L, d_c]
            att_in = torch.cat([item_vec, concept_vecs], dim=-1)        # [L, d_i+d_c]
            att_logits = self.item_concept_mlp(att_in).squeeze(-1)      # [L]
            weights = F.softmax(att_logits, dim=-1)                      # [L]

            k_score = torch.sum(weights * vals)                          # 标准加权 mean
            batch_k_score.append(k_score)

        batch_k_score = torch.stack(batch_k_score)                       # [B]

        # 3. 技巧得分 + item bias
        skill_score = torch.sigmoid(self.skill_score_head(z_skill)).squeeze(-1)  # [B]
        item_b = self.item_bias(i_ids).squeeze(-1)                                # [B]

        # 4. 最终预测：3 维特征 -> MLP -> Sigmoid
        feat = torch.stack([batch_k_score, skill_score, item_b], dim=-1)  # [B, 3]
        preds = self.pred_mlp(feat).squeeze(-1)                           # [B]

        # 为了兼容现有 loss_function 的接口，这里仍然返回 z_k/z_skill，adj 置 None
        return preds, None, z_k_init, z_skill


# ==========================================
# 3. Loss & Training
# ==========================================
def loss_function(preds, labels, adj, z_k, z_skill, args):
    """
    返回:
      total: 标量 loss
      loss_dict: 各个组成部分，方便 log 打印
    """
    device = labels.device

    # -------- 1. Task Loss（主任务）---------
    preds = torch.clamp(preds, 1e-4, 1 - 1e-4)
    l_bce = F.binary_cross_entropy(preds, labels)

    # 先全部初始化为 0，保证任何情况下都有值
    l_ent_dag = torch.tensor(0.0, device=device)
    l_ent_sym = torch.tensor(0.0, device=device)
    l_dag     = torch.tensor(0.0, device=device)
    l_sym     = torch.tensor(0.0, device=device)
    l_orth    = torch.tensor(0.0, device=device)

    # -------- 2. 图相关 Loss（entropy + symmetry + 可选 DAG）---------
    if adj is not None:
        adj_dag, adj_sym = adj[0], adj[1]   # head0: 前置 / head1: 相似

        # === 2.1 目标熵约束：希望每行大约有 graph_topk 条“有效边” ===
        # 行熵 H(row) = -sum_j a_ij log a_ij
        H_dag = -torch.sum(adj_dag * torch.log(adj_dag + 1e-8), dim=-1)  # [N]
        H_sym = -torch.sum(adj_sym * torch.log(adj_sym + 1e-8), dim=-1)  # [N]

        # 理想情况：每行大约有 K 条差不多的非零边 -> 熵 ~ log K
        K = max(1, getattr(args, "graph_topk", 4))
        target_H_dag = math.log(float(K))
        target_H_sym = math.log(float(K))

        # 让实际熵逼近目标熵，而不是一味变小
        l_ent_dag = torch.mean((H_dag - target_H_dag) ** 2)
        l_ent_sym = torch.mean((H_sym - target_H_sym) ** 2)

        # === 2.2 Symmetry：只约束“相似头”是对称的 ===
        l_sym = torch.mean((adj_sym - adj_sym.transpose(-1, -2)) ** 2)

        # === 2.3 DAG Loss：只在「明确打开 + 图不大」时才算，避免炸 CUDA ===
        if args.lambda_dag > 0 and adj_dag.shape[-1] <= 200:
            if not (torch.isnan(adj_dag).any() or torch.isinf(adj_dag).any()):
                try:
                    l_dag = torch.trace(torch.matrix_exp(adj_dag)) - adj_dag.shape[-1]
                except RuntimeError:
                    # 数值不稳定时就当 0，让权重把影响压死
                    l_dag = torch.tensor(0.0, device=adj_dag.device)

    # -------- 3. 解耦 Orth：知识 vs 技巧 正交 ---------
    if (z_k is not None) and (z_skill is not None) and (args.lambda_orth > 0):
        if z_k.shape[1] > z_skill.shape[1]:
            z_k_red = z_k[:, :z_skill.shape[1]]
        else:
            z_k_red = z_k

        vn_k = F.normalize(z_k_red, dim=1)
        vn_s = F.normalize(z_skill, dim=1)
        cos = torch.sum(vn_k * vn_s, dim=1)
        l_orth = torch.mean(cos ** 2)

    # -------- 4. 总 Loss 汇总 ---------
    total = (
        l_bce
        + args.lambda_ent_dag * l_ent_dag
        + args.lambda_ent_sym * l_ent_sym
        + args.lambda_dag * l_dag
        + args.lambda_sym * l_sym
        + args.lambda_orth * l_orth
    )

    loss_dict = {
        "BCE":  float(l_bce.detach().cpu()),
        "EntD": float(l_ent_dag.detach().cpu()),
        "EntS": float(l_ent_sym.detach().cpu()),
        "DAG":  float(l_dag.detach().cpu()),
        "Sym":  float(l_sym.detach().cpu()),
        "Orth": float(l_orth.detach().cpu()),
        "Total": float(total.detach().cpu()),
    }

    return total, loss_dict


def run_dataset(dataset_name, args):
    print(f"\n🚀 Running {dataset_name} ...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    train_ds, valid_ds, test_ds, info = load_data(dataset_name)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    
    log_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = get_logger(args.log_dir, log_name, dataset_name)
    
    logger.info(f"Stats: {info['num_students']} Stu, {info['num_items']} Items, {info['num_concepts']} Concepts")
    logger.info(
    "HyperParams: "
    f"model={args.model_name}, "
    f"batch={args.batch_size}, epochs={args.epochs}, "
    f"lr={args.lr}, wd={args.weight_decay}, "
    f"dim_stu={args.dim_student}, dim_item={args.dim_item}, dim_cpt={args.dim_concept}, "
    f"graph_topk={args.graph_topk}, drop={args.dropout}, gdrop={args.graph_dropout}, "
    f"λ_ent_dag={getattr(args, 'lambda_ent_dag', 0.0)}, "
    f"λ_ent_sym={getattr(args, 'lambda_ent_sym', 0.0)}, "
    f"λ_sym={getattr(args, 'lambda_sym', 0.0)}, "
    f"λ_dag={getattr(args, 'lambda_dag', 0.0)}, "
    f"λ_orth={getattr(args, 'lambda_orth', 0.0)}"
    )

    # ===== 选择模型结构 =====
    if args.model_name in ["FullCDM_V3", "FullCDM_V3_Fixed"]:
        model = FullCDM(info['num_students'], info['num_items'], info['num_concepts'], args).to(device)
    elif args.model_name == "PlainCDM":
        model = PlainCDM(info['num_students'], info['num_items'], info['num_concepts'], args).to(device)
    elif args.model_name == "HybridKTCDM":
        model = HybridKTCDM(info['num_students'], info['num_items'], info['num_concepts'], args).to(device)
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")
    
    # 在日志和终端里都标一条当前使用的模型结构
    model_desc = f"Using model structure: {args.model_name}"
    print(model_desc)
    logger.info(model_desc)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_auc = 0.0
    best_acc = 0.0
    best_epoch = 0
    patience_cnt = 0
    save_path = os.path.join(args.log_dir, f"{dataset_name}_best_v3_fixed.pt")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_dl:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            
            preds, adj, z_k, z_s = model(batch)
            loss, loss_dict = loss_function(preds, batch["labels"], adj, z_k, z_s, args)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # ===== 验证 =====
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in valid_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds, _, _, _ = model(batch)
                y_true.extend(batch["labels"].cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
        
        orth_val = loss_dict.get("Orth", 0.0)
        avg_train_loss = total_loss / len(train_dl)

        logger.info(
            "Ep %02d | Loss: %.4f | Val AUC: %.4f ACC: %.4f | "
            "BCE: %.4f EntD: %.4f EntS: %.4f DAG: %.4f Sym: %.4f Orth: %.4f"
            % (
                epoch,
                avg_train_loss,
                auc, acc,
                loss_dict["BCE"],
                loss_dict["EntD"],
                loss_dict["EntS"],
                loss_dict["DAG"],
                loss_dict["Sym"],
                loss_dict["Orth"],
            )
        )

        
        # ===== 刷新 best 时额外打一条 log =====
        if auc > best_auc:
            best_auc = auc
            best_acc = acc
            best_epoch = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            logger.info(
                f"🔥 New best on {dataset_name}: "
                f"AUC={best_auc:.4f}, ACC={best_acc:.4f} @ epoch {best_epoch:02d}"
            )
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(
                    f"Early Stopping triggered at epoch {epoch:02d}. "
                    f"Best Val: AUC={best_auc:.4f}, ACC={best_acc:.4f} @ epoch {best_epoch:02d}"
                )
                break
    
    # ===== 测试集 =====
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds, _, _, _ = model(batch)
            y_true.extend(batch["labels"].cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    test_auc = roc_auc_score(y_true, y_pred)
    test_acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
    test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    logger.info(
        f"🌟 Final Test ({dataset_name}): "
        f"AUC={test_auc:.4f}, ACC={test_acc:.4f}, RMSE={test_rmse:.4f} | "
        f"Best Val: AUC={best_auc:.4f}, ACC={best_acc:.4f} @ epoch {best_epoch:02d}"
    )
    
    metrics = {"auc": test_auc, "acc": test_acc, "rmse": test_rmse}
    save_result_csv(args, metrics, best_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--model_name", type=str, default="HybridKTCDM")
    parser.add_argument("--note", type=str, default="MinAgg_3PL_SepEnt")
    
    parser.add_argument("--dim_student", type=int, default=64)
    parser.add_argument("--dim_item", type=int, default=64)
    parser.add_argument("--dim_concept", type=int, default=64)
    parser.add_argument("--dim_hidden", type=int, default=64)
    parser.add_argument("--dim_skill", type=int, default=16)
    
    parser.add_argument("--softmin_beta", type=float, default=5.0)

    parser.add_argument("--graph_topk", type=int, default=8)
    parser.add_argument("--graph_dropout", type=float, default=0.2)
    
    parser.add_argument("--lambda_dag", type=float, default=0.005)
    parser.add_argument("--lambda_sym", type=float, default=0.02)
    parser.add_argument("--lambda_ent_dag", type=float, default=0.005)
    parser.add_argument("--lambda_ent_sym", type=float, default=0.005)
    parser.add_argument("--lambda_orth", type=float, default=0.01)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.dataset == "assist_17" and args.dim_student == 64:
        args.dim_student = 128
        args.dim_concept = 128
        args.lr = 8e-4
        args.batch_size = 2048
        
    run_dataset(args.dataset, args)