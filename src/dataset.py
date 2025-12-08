import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


class CognitiveDiagnosisDataset(Dataset):
    """认知诊断数据集类"""

    def __init__(
            self,
            csv_file: str,
            stu_id_map: Dict[int, int],
            exer_id_map: Dict[int, int],
            cpt_id_map: Dict[int, int]
    ):
        """
        初始化数据集

        Args:
            csv_file: CSV文件路径 或 已经读取好的 DataFrame
            stu_id_map: 学生ID映射字典 {原始ID: 新ID}
            exer_id_map: 习题ID映射字典 {原始ID: 新ID}
            cpt_id_map: 知识点ID映射字典 {原始ID: 新ID}
        """
        self.stu_id_map = stu_id_map
        self.exer_id_map = exer_id_map
        self.cpt_id_map = cpt_id_map
        self.num_concepts = len(cpt_id_map)

        # 读取数据：支持路径或 DataFrame
        if isinstance(csv_file, str):
            self.data = pd.read_csv(csv_file)
        else:
            # 认为是 DataFrame
            self.data = csv_file.copy().reset_index(drop=True)

        # 重新映射ID并转换为tensor
        self.student_ids = torch.LongTensor([
            stu_id_map[sid] for sid in self.data['stu_id'].values
        ])
        self.exercise_ids = torch.LongTensor([
            exer_id_map[eid] for eid in self.data['exer_id'].values
        ])
        self.labels = torch.FloatTensor(self.data['label'].values)

        # 处理知识点序列，转换为one-hot编码
        self.concept_matrix = self._process_concepts(self.data['cpt_seq'].values)

    def _process_concepts(self, concept_seqs) -> torch.Tensor:
        """
        将知识点序列转换为one-hot编码矩阵

        Args:
            concept_seqs: 知识点序列数组

        Returns:
            形状为 (数据长度, 知识点总数) 的one-hot张量
        """
        batch_size = len(concept_seqs)
        concept_matrix = torch.zeros(batch_size, self.num_concepts)

        for i, seq in enumerate(concept_seqs):
            # 处理字符串格式的知识点序列
            if pd.isna(seq):  # 处理可能的NaN值
                continue

            # 将字符串分割为知识点ID列表
            if isinstance(seq, str):
                concept_ids = [int(cid) for cid in seq.split(',')]
            else:
                concept_ids = [int(seq)]

            # 使用映射后的ID设置one-hot编码
            for cid in concept_ids:
                mapped_cid = self.cpt_id_map.get(cid)
                if mapped_cid is not None:
                    concept_matrix[i, mapped_cid] = 1

        return concept_matrix

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (student_id, exercise_id, concept_vector, label) 元组
        """
        return (
            self.student_ids[idx],
            self.exercise_ids[idx],
            self.concept_matrix[idx],
            self.labels[idx]
        )


def build_id_mappings(sources: list) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    从所有数据源中构建ID映射

    Args:
        sources: 列表，每个元素可以是 CSV 文件路径(str) 或 pandas.DataFrame

    Returns:
        (stu_id_map, exer_id_map, cpt_id_map) 三个映射字典
    """
    all_stu_ids = set()
    all_exer_ids = set()
    all_cpt_ids = set()

    # 收集所有唯一ID
    for src in sources:
        if isinstance(src, str):
            df = pd.read_csv(src)
        else:
            df = src

        # 收集学生ID和习题ID
        all_stu_ids.update(df['stu_id'].unique())
        all_exer_ids.update(df['exer_id'].unique())

        # 收集知识点ID
        for seq in df['cpt_seq'].values:
            if pd.isna(seq):
                continue
            if isinstance(seq, str):
                concept_ids = [int(cid) for cid in seq.split(',')]
            else:
                concept_ids = [int(seq)]
            all_cpt_ids.update(concept_ids)

    # 创建映射字典：原始ID -> 连续的新ID (从0开始)
    stu_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_stu_ids))}
    exer_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_exer_ids))}
    cpt_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_cpt_ids))}

    return stu_id_map, exer_id_map, cpt_id_map


def build_q_matrix(
        sources: list,
        exer_id_map: Dict[int, int],
        cpt_id_map: Dict[int, int]
) -> torch.Tensor:
    """
    构建Q矩阵 (习题-知识点关联矩阵)

    Args:
        sources: 列表，每个元素可以是 CSV 文件路径(str) 或 pandas.DataFrame
        exer_id_map: 习题ID映射字典 {原始ID: 新ID}
        cpt_id_map: 知识点ID映射字典 {原始ID: 新ID}

    Returns:
        Q矩阵，形状为 (习题总数, 知识点总数)
    """
    num_exercises = len(exer_id_map)
    num_concepts = len(cpt_id_map)

    # 初始化Q矩阵
    q_matrix = torch.zeros(num_exercises, num_concepts)

    # 用于存储习题和知识点的对应关系
    exercise_concepts = {}

    # 从所有文件 / DataFrame 中收集习题-知识点关系
    for src in sources:
        if isinstance(src, str):
            df = pd.read_csv(src)
        else:
            df = src

        for _, row in df.iterrows():
            exer_id = row['exer_id']
            cpt_seq = row['cpt_seq']

            # 获取映射后的习题ID
            mapped_exer_id = exer_id_map[exer_id]

            # 解析知识点序列
            if pd.isna(cpt_seq):
                continue

            if isinstance(cpt_seq, str):
                concept_ids = [int(cid) for cid in cpt_seq.split(',')]
            else:
                concept_ids = [int(cpt_seq)]

            # 存储该习题对应的知识点
            if mapped_exer_id not in exercise_concepts:
                exercise_concepts[mapped_exer_id] = set()

            for cid in concept_ids:
                if cid in cpt_id_map:
                    exercise_concepts[mapped_exer_id].add(cpt_id_map[cid])

    # 填充Q矩阵
    for exer_id, concepts in exercise_concepts.items():
        for cpt_id in concepts:
            q_matrix[exer_id, cpt_id] = 1

    return q_matrix


def create_dataloaders(
        train_file: str,
        val_file: str,
        test_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
        min_stu_interactions: int = 0,
        min_exer_interactions: int = 0,
        min_poison_count: int = 0,
        logger=None,
        dataset_name: str = None,   # ★ 新增：数据集名字（assist_09 / junyi / assist_17）
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    创建训练、验证和测试的数据加载器，并执行统一的数据清洗

    Args:
        train_file: 训练集CSV文件路径
        val_file: 验证集CSV文件路径
        test_file: 测试集CSV文件路径
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        shuffle_train: 是否打乱训练集
        min_stu_interactions: 学生最小答题数阈值（<该值的学生将被过滤，0 表示不启用）
        min_exer_interactions: 习题最小被作答数阈值（<该值的题目将被过滤，0 表示不启用）
        min_poison_count: 毒题检测所需的最小作答次数（0 表示不启用）
        logger: 可选的 logger，对清洗过程进行记录

    Returns:
        (train_loader, val_loader, test_loader, info_dict) 元组
        info_dict包含数据集的统计信息和映射字典
    """
    dataset_prefix = f"[{dataset_name}]" if dataset_name else ""

    def log(msg: str):
        full_msg = f"{dataset_prefix} {msg}" if dataset_prefix else msg
        if logger is not None:
            logger.info(full_msg)
        else:
            print(full_msg)


    # 1. 读取原始 CSV
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    # ============ 统一的清洗逻辑 ============

    # --- 学生冷启动过滤 ---
    if min_stu_interactions > 0:
        combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
        stu_counts = combined.groupby("stu_id").size()
        keep_users = set(
            stu_counts[stu_counts >= min_stu_interactions].index
        )
        removed_users = len(stu_counts) - len(keep_users)
        log(
            f"[数据清洗] 学生交互次数 < {min_stu_interactions} 次的学生：共移除 {removed_users} 个学生（仅统计用户数）"
        )

        def _filter_by_students(df):
            before = len(df)
            df_new = df[df["stu_id"].isin(keep_users)].reset_index(drop=True)
            return df_new, before - len(df_new)

        dropped_total = 0
        train_df, d = _filter_by_students(train_df); dropped_total += d
        val_df, d = _filter_by_students(val_df); dropped_total += d
        test_df, d = _filter_by_students(test_df); dropped_total += d

        log(f"[数据清洗] 学生冷启动过滤：在 train/valid/test 中共删除 {dropped_total} 条记录。")
    else:
        log("[数据清洗] 跳过学生冷启动过滤（min_stu_interactions <= 0）。")

    # --- 题目冷门过滤 ---
    if min_exer_interactions > 0:
        combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
        exer_counts = combined.groupby("exer_id").size()
        keep_items = set(
            exer_counts[exer_counts >= min_exer_interactions].index
        )
        removed_items = len(exer_counts) - len(keep_items)
        log(
            f"[数据清洗] 题目作答次数 < {min_exer_interactions} 次的题目：共移除 {removed_items} 道题（仅统计题目数）"
        )

        def _filter_by_items(df):
            before = len(df)
            df_new = df[df["exer_id"].isin(keep_items)].reset_index(drop=True)
            return df_new, before - len(df_new)

        dropped_total = 0
        train_df, d = _filter_by_items(train_df); dropped_total += d
        val_df, d = _filter_by_items(val_df); dropped_total += d
        test_df, d = _filter_by_items(test_df); dropped_total += d

        log(f"[数据清洗] 题目冷门过滤：在 train/valid/test 中共删除 {dropped_total} 条记录。")
    else:
        log("[数据清洗] 跳过题目冷门过滤（min_exer_interactions <= 0）。")

    # --- 毒题清洗（Acc=0 或 1 且作答数 >= min_poison_count） ---
    if min_poison_count > 0:
        combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
        item_stats = combined.groupby("exer_id")["label"].agg(
            count="count", correct_rate="mean"
        ).reset_index()

        poison_mask = (
                (item_stats["count"] >= min_poison_count)
                & ((item_stats["correct_rate"] <= 0.0)
                   | (item_stats["correct_rate"] >= 1.0))
        )
        poison_items = item_stats.loc[poison_mask, :]
        poison_ids = set(poison_items["exer_id"].tolist())

        if len(poison_ids) > 0:
            log(
                f"[数据清洗] 检测到 {len(poison_ids)} 道“毒题”"
                f"（作答次数 ≥ {min_poison_count} 且正确率为 0 或 1），将从所有数据集中移除。"
            )
            keep_non_toxic = set(
                item_stats[~item_stats["exer_id"].isin(poison_ids)]["exer_id"]
            )

            def _filter_non_toxic(df):
                before = len(df)
                df_new = df[df["exer_id"].isin(keep_non_toxic)].reset_index(drop=True)
                return df_new, before - len(df_new)

            dropped_total = 0
            train_df, d = _filter_non_toxic(train_df); dropped_total += d
            val_df, d = _filter_non_toxic(val_df); dropped_total += d
            test_df, d = _filter_non_toxic(test_df); dropped_total += d

            log(f"[数据清洗] 毒题清洗：在 train/valid/test 中共删除 {dropped_total} 条记录。")
        else:
            log("[数据清洗] 在当前 min_poison_count 设置下未检测到“毒题”。")
    else:
        log("[数据清洗] 跳过毒题清洗（min_poison_count <= 0）。")

    # 2. 构建ID映射 & Q矩阵（基于清洗后的 DataFrame）
    csv_sources = [train_df, val_df, test_df]

    log("正在构建ID映射...")
    stu_id_map, exer_id_map, cpt_id_map = build_id_mappings(csv_sources)

    log("正在构建Q矩阵...")
    q_matrix = build_q_matrix(csv_sources, exer_id_map, cpt_id_map)

    # ========= 统计每题的概念数量（全局 + 各数据集） =========
    # 每道题对应的概念数：按 Q 矩阵行求和
    concepts_per_exercise = q_matrix.sum(dim=1)  # (num_exercises,)

    def _log_concept_stats(split_name: str, df: pd.DataFrame):
        """对某个 split 统计：每道题关联的概念数量分布"""
        if df is None or len(df) == 0:
            log(f"[{split_name}] 数据集为空，跳过概念数统计。")
            return

        # 把该 split 中出现的题目映射到内部ID
        raw_exer_ids = df["exer_id"].unique().tolist()
        mapped_ids = []
        for eid in raw_exer_ids:
            if eid in exer_id_map:
                mapped_ids.append(exer_id_map[eid])

        if len(mapped_ids) == 0:
            log(f"[{split_name}] 无有效题目ID，跳过概念数统计。")
            return

        mapped_ids = np.array(mapped_ids, dtype=int)

        # 取出这些题目的概念数
        counts = concepts_per_exercise[mapped_ids].cpu().numpy()
        n_items = len(counts)

        mean_c = float(counts.mean())
        min_c = int(counts.min())
        max_c = int(counts.max())

        # 1 概念、多概念 题目比例
        num_1 = int((counts == 1).sum())
        num_ge2 = int((counts >= 2).sum())
        p1 = num_1 / n_items
        p_ge2 = num_ge2 / n_items

        log(
            f"[{split_name}] 概念数统计：题目数={n_items}, "
            f"平均概念数={mean_c:.4f}, min={min_c}, max={max_c}, "
            f"1概念={num_1} ({p1:.2%}), ≥2概念={num_ge2} ({p_ge2:.2%})"
        )

        # 如需更细的直方图分布，也可以顺手打一行
        hist = {}
        for k in range(min_c, max_c + 1):
            hist[k] = int((counts == k).sum())
        log(f"[{split_name}] 概念数直方图（概念数: 题目数量）: {hist}")

    log("[统计] 开始统计清洗后每道题的概念数量分布...")
    _log_concept_stats("Train", train_df)
    _log_concept_stats("Valid", val_df)
    _log_concept_stats("Test", test_df)
    log("[统计] 概念数量统计完成。")

    # 3. 创建数据集（直接传 DataFrame，避免重复读盘）
    log("正在加载数据集...")
    train_dataset = CognitiveDiagnosisDataset(train_df, stu_id_map, exer_id_map, cpt_id_map)
    val_dataset = CognitiveDiagnosisDataset(val_df, stu_id_map, exer_id_map, cpt_id_map)
    test_dataset = CognitiveDiagnosisDataset(test_df, stu_id_map, exer_id_map, cpt_id_map)

    # 4. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 5. 收集信息
    info_dict = {
        'num_students': len(stu_id_map),
        'num_exercises': len(exer_id_map),
        'num_concepts': len(cpt_id_map),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'stu_id_map': stu_id_map,
        'exer_id_map': exer_id_map,
        'cpt_id_map': cpt_id_map,
        'stu_id_reverse_map': {v: k for k, v in stu_id_map.items()},
        'exer_id_reverse_map': {v: k for k, v in exer_id_map.items()},
        'cpt_id_reverse_map': {v: k for k, v in cpt_id_map.items()},
        'q_matrix': q_matrix,
        # 新增：每道题的“概念数量”，与 exer_id 内部索引对齐
        'concepts_per_exercise': concepts_per_exercise,
    }

    return train_loader, val_loader, test_loader, info_dict


# 使用示例
if __name__ == "__main__":
    # 创建数据加载器
    train_loader, val_loader, test_loader, info = create_dataloaders(
        train_file='./data/assist-09/process_data/train.csv',
        val_file='./data/assist-09/process_data/valid.csv',
        test_file='./data/assist-09/process_data/test.csv',
        batch_size=64,
        num_workers=4
    )

    # 测试数据加载
    print("\n" + "=" * 50)
    print("测试数据加载:")
    for batch_idx, (stu_ids, exer_ids, concepts, labels) in enumerate(train_loader):
        print(f"\n批次 {batch_idx + 1}:")
        print(f"学生ID形状: {stu_ids.shape}")
        print(f"习题ID形状: {exer_ids.shape}")
        print(f"知识点矩阵形状: {concepts.shape}")
        print(f"标签形状: {labels.shape}")

        # 验证Q矩阵和concept矩阵的一致性
        for i in range(min(3, len(exer_ids))):
            exer_id = exer_ids[i].item()
            q_concepts = info['q_matrix'][exer_id].nonzero().squeeze()
            data_concepts = concepts[i].nonzero().squeeze()

            if q_concepts.dim() == 0:
                q_concepts = q_concepts.unsqueeze(0)
            if data_concepts.dim() == 0:
                data_concepts = data_concepts.unsqueeze(0)

            print(f"\n样本 {i + 1}:")
            print(f"  习题ID: {exer_id}")
            print(f"  Q矩阵中的知识点: {q_concepts.tolist()}")
            print(f"  数据中的知识点: {data_concepts.tolist()}")
            print(f"  是否一致: {torch.equal(q_concepts.sort()[0], data_concepts.sort()[0])}")

        # 只显示第一个批次
        break
