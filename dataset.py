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
            csv_file: CSV文件路径
            stu_id_map: 学生ID映射字典 {原始ID: 新ID}
            exer_id_map: 习题ID映射字典 {原始ID: 新ID}
            cpt_id_map: 知识点ID映射字典 {原始ID: 新ID}
        """
        self.stu_id_map = stu_id_map
        self.exer_id_map = exer_id_map
        self.cpt_id_map = cpt_id_map
        self.num_concepts = len(cpt_id_map)

        # 读取CSV文件
        self.data = pd.read_csv(csv_file)

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


def build_id_mappings(csv_files: list) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    从所有CSV文件中构建ID映射

    Args:
        csv_files: CSV文件路径列表

    Returns:
        (stu_id_map, exer_id_map, cpt_id_map) 三个映射字典
    """
    all_stu_ids = set()
    all_exer_ids = set()
    all_cpt_ids = set()

    # 收集所有唯一ID
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

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
        csv_files: list,
        exer_id_map: Dict[int, int],
        cpt_id_map: Dict[int, int]
) -> torch.Tensor:
    """
    构建Q矩阵 (习题-知识点关联矩阵)

    Args:
        csv_files: CSV文件路径列表
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

    # 从所有文件中收集习题-知识点关系
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

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
        shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    创建训练、验证和测试的数据加载器

    Args:
        train_file: 训练集CSV文件路径
        val_file: 验证集CSV文件路径
        test_file: 测试集CSV文件路径
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        shuffle_train: 是否打乱训练集

    Returns:
        (train_loader, val_loader, test_loader, info_dict) 元组
        info_dict包含数据集的统计信息和映射字典
    """
    csv_files = [train_file, val_file, test_file]

    # 构建ID映射
    print("正在构建ID映射...")
    stu_id_map, exer_id_map, cpt_id_map = build_id_mappings(csv_files)

    # 构建Q矩阵
    print("正在构建Q矩阵...")
    q_matrix = build_q_matrix(csv_files, exer_id_map, cpt_id_map)

    # 创建数据集
    print("正在加载数据集...")
    train_dataset = CognitiveDiagnosisDataset(train_file, stu_id_map, exer_id_map, cpt_id_map)
    val_dataset = CognitiveDiagnosisDataset(val_file, stu_id_map, exer_id_map, cpt_id_map)
    test_dataset = CognitiveDiagnosisDataset(test_file, stu_id_map, exer_id_map, cpt_id_map)

    # 创建数据加载器
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

    # 收集数据集信息
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
        'q_matrix': q_matrix
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
