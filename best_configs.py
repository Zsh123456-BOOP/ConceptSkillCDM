#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
best_configs.py
共享的最佳超参数配置，供 run_all_datasets.py 和 run_ablation.py 使用。

修改此文件后，两个脚本都会自动使用新配置。
"""

from typing import Dict, Any

# ========================================================================
# BEST_CFG 参数说明
# ========================================================================
#
# 【训练参数】
#   seed                  : 随机种子，保证实验可复现
#   batch_size            : 批大小
#   epochs                : 最大训练轮数
#   learning_rate         : 学习率
#   weight_decay          : L2 正则权重（AdamW）
#   dropout               : Dropout 概率
#   early_stop_patience   : 早停耐心（验证集无提升的轮数）
#   patience              : 学习率调度器耐心
#   save_interval         : 每隔多少 epoch 保存一次 checkpoint
#   num_workers           : DataLoader 工作进程数
#   no_cuda               : 是否禁用 GPU
#
# 【模型结构】
#   knowledge_dim         : 概念/知识状态的嵌入维度（Module 1/2 共用）
#   exercise_dim          : 题目嵌入维度（Module 2/3 共用）
#   skill_dim             : 学生技能潜在向量维度（Module 3 - MF 分支）
#   num_gnn_layers        : GNN 传播层数（Module 1）
#   num_relation_heads    : 多头关系学习的头数（Module 1 - 概念图）
#
# 【Module 1 - 概念结构建模】
#   use_personal_graph    : 是否启用个性化图（E），为每个学生学习独立的概念关系
#   lambda_sparse         : 全局概念图行熵稀疏正则权重（A），值越大图越稀疏
#   lambda_sparse_personal: 个性化图行熵稀疏正则权重（E），仅 use_personal_graph=True 时生效
#   lambda_alpha          : 个性化图混合系数 alpha 惩罚权重，仅 use_personal_graph=True 时生效
#
# 【Module 3 - Soft Prototype 校正】
#   num_prototypes        : 原型数量（C），0 表示禁用
#   proto_tau             : 原型分配的温度系数
#   proto_lambda          : 原型校正混合权重，knowledge_state = (1-λ)*原始 + λ*原型校正
#   lambda_proto_div      : 原型多样性正则权重（鼓励原型向量正交）
#   lambda_proto_usage    : 原型均匀使用正则权重（鼓励各原型被均匀分配）
#
# 【Module 3 - MF 残差分支】
#   exercise_l2_lambda    : MF/IRT 参数 L2 正则权重
#
# 【数据过滤】
#   min_stu_interactions  : 学生最少交互数（冷启动过滤）
#   min_exer_interactions : 题目最少交互数（冷门题过滤），0 表示不过滤
#   min_poison_count      : 毒题阈值（全对/全错比例异常），0 表示不过滤
#
# 【消融开关（旧版，兼容）】
#   disable_soft_prototype: 禁用 Soft Prototype（等价于 num_prototypes=0）
#   ablate_skill_encoder  : 消融 Skill Encoder（Module 3 的 MF 分支）
#   ablate_soft_prototype : 消融 Soft Prototype（Module 3 的 C）
#
# 【其他】
#   model_variant         : 模型变体标识，用于区分不同配置的实验
# ========================================================================

BEST_CFG: Dict[str, Dict[str, Any]] = {
    "junyi": {
        # === 训练参数 ===
        "seed": 42,
        "batch_size": 512,
        "epochs": 100,
        "learning_rate": 0.003,
        "weight_decay": 0.001,
        "dropout": 0.4,
        "early_stop_patience": 3,
        "patience": 1,
        "save_interval": 10,
        "num_workers": 4,
        "no_cuda": False,

        # === 模型结构 ===
        "knowledge_dim": 128,            # Module 1/2 概念嵌入维度
        "exercise_dim": 128,             # Module 2/3 题目嵌入维度
        "skill_dim": 64,                 # Module 3 学生 MF 潜在维度
        "num_gnn_layers": 2,             # Module 1 GNN 层数
        "num_relation_heads": 4,         # Module 1 多头关系数

        # === Module 1: 概念图正则 ===
        "lambda_sparse": 1,              # 全局概念图稀疏正则
        "lambda_sparse_personal": 0.1, # 个性化图稀疏正则（需 use_personal_graph=True）
        "lambda_alpha": 0.1,            # 个性化混合系数惩罚（需 use_personal_graph=True）
        "use_personal_graph": True,      # 是否启用个性化图

        # === Module 3: Soft Prototype ===
        "num_prototypes": 3,             # 原型数量
        "proto_tau": 1.0,                # 原型温度
        "proto_lambda": 5,               # 原型校正混合权重
        "lambda_proto_div": 0.1,         # 原型多样性正则
        "lambda_proto_usage": 0.1,       # 原型均匀使用正则
        "disable_soft_prototype": False,

        # === Module 3: MF 正则 ===
        "exercise_l2_lambda": 5e-5,      # MF/IRT 参数 L2 正则

        # === 数据过滤 ===
        "min_stu_interactions": 15,
        "min_exer_interactions": 0,
        "min_poison_count": 0,

        # === 消融开关（兼容旧版）===
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,

        # === 其他 ===
        "model_variant": "gpd_base",
    },

    "assist_09": {
        # === 训练参数 ===
        "seed": 42,
        "batch_size": 128,
        "epochs": 100,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "dropout": 0.20,
        "early_stop_patience": 3,
        "patience": 5,
        "save_interval": 10,
        "num_workers": 4,
        "no_cuda": False,

        # === 模型结构 ===
        "knowledge_dim": 64,             # Module 1/2 概念嵌入维度
        "exercise_dim": 64,              # Module 2/3 题目嵌入维度
        "skill_dim": 128,                # Module 3 学生 MF 潜在维度
        "num_gnn_layers": 1,             # Module 1 GNN 层数
        "num_relation_heads": 4,         # Module 1 多头关系数

        # === Module 1: 概念图正则 ===
        "lambda_sparse": 1,              # 全局概念图稀疏正则
        "lambda_sparse_personal": 0.1, # 个性化图稀疏正则（需 use_personal_graph=True）
        "lambda_alpha": 0.1,            # 个性化混合系数惩罚（需 use_personal_graph=True）
        "use_personal_graph": True,      # 是否启用个性化图

        # === Module 3: Soft Prototype ===
        "num_prototypes": 3,             # 原型数量
        "proto_tau": 1.0,                # 原型温度
        "proto_lambda": 5,               # 原型校正混合权重
        "lambda_proto_div": 0.1,         # 原型多样性正则
        "lambda_proto_usage": 0.1,       # 原型均匀使用正则
        "disable_soft_prototype": False,

        # === Module 3: MF 正则 ===
        "exercise_l2_lambda": 5e-5,      # MF/IRT 参数 L2 正则

        # === 数据过滤 ===
        "min_stu_interactions": 15,
        "min_exer_interactions": 0,
        "min_poison_count": 0,

        # === 消融开关（兼容旧版）===
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,

        # === 其他 ===
        "model_variant": "gpd_base",
    },
}

# 默认种子列表
DEFAULT_SEEDS = [42]
