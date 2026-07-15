import math
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset


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

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (student_id, exercise_id, label) 元组
        """
        return (
            self.student_ids[idx],
            self.exercise_ids[idx],
            self.labels[idx]
        )


def _load_source_df(src) -> pd.DataFrame:
    if isinstance(src, str):
        return pd.read_csv(src)
    return src.copy() if isinstance(src, pd.DataFrame) else pd.DataFrame(src)


def _iter_source_dfs(sources: Optional[list]):
    for src in sources or []:
        if src is None:
            continue
        yield _load_source_df(src)


def _parse_concept_seq(seq) -> List[int]:
    if isinstance(seq, (list, tuple, set, np.ndarray)):
        return [int(cid) for cid in seq]
    if pd.isna(seq):
        return []
    if isinstance(seq, str):
        return [int(cid) for cid in seq.split(",") if str(cid).strip()]
    return [int(seq)]


def build_id_mappings(
        train_sources: list,
        metadata_sources: Optional[list] = None,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    基于 train-only 数据源构建 ID 映射。

    metadata_sources 仅用于显式补充 train 中已出现习题的概念元数据；
    它不会扩展 student/item 的 seen 空间，也不会隐式使用 val/test。

    Args:
        train_sources: 训练数据源列表，每个元素可以是 CSV 文件路径(str) 或 pandas.DataFrame
        metadata_sources: 可选的显式题目元数据源，仅用于补充 train-seen exercise 的 concept 集合

    Returns:
        (stu_id_map, exer_id_map, cpt_id_map) 三个映射字典
    """
    all_stu_ids = set()
    all_exer_ids = set()
    all_cpt_ids = set()

    for df in _iter_source_dfs(train_sources):
        all_stu_ids.update(df['stu_id'].unique())
        all_exer_ids.update(df['exer_id'].unique())
        for seq in df['cpt_seq'].values:
            all_cpt_ids.update(_parse_concept_seq(seq))

    train_seen_exercises = set(all_exer_ids)
    for df in _iter_source_dfs(metadata_sources):
        if "exer_id" not in df.columns or "cpt_seq" not in df.columns:
            continue
        meta_df = df[df["exer_id"].isin(train_seen_exercises)]
        for seq in meta_df["cpt_seq"].values:
            all_cpt_ids.update(_parse_concept_seq(seq))

    # 创建映射字典：原始ID -> 连续的新ID (从0开始)
    stu_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_stu_ids))}
    exer_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_exer_ids))}
    cpt_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_cpt_ids))}

    return stu_id_map, exer_id_map, cpt_id_map


def build_q_matrix(
        train_sources: list,
        exer_id_map: Dict[int, int],
        cpt_id_map: Dict[int, int],
        metadata_sources: Optional[list] = None,
) -> torch.Tensor:
    """
    构建Q矩阵 (习题-知识点关联矩阵)

    Args:
        train_sources: 训练数据源列表，每个元素可以是 CSV 文件路径(str) 或 pandas.DataFrame
        exer_id_map: 习题ID映射字典 {原始ID: 新ID}
        cpt_id_map: 知识点ID映射字典 {原始ID: 新ID}
        metadata_sources: 可选的显式题目元数据，仅用于补充 train-seen 习题的概念关系

    Returns:
        Q矩阵，形状为 (习题总数, 知识点总数)
    """
    num_exercises = len(exer_id_map)
    num_concepts = len(cpt_id_map)

    # 初始化Q矩阵
    q_matrix = torch.zeros(num_exercises, num_concepts)

    # 用于存储习题和知识点的对应关系
    exercise_concepts = {}

    def _collect_relations_from_df(df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            exer_id = row['exer_id']
            cpt_seq = row['cpt_seq']

            if exer_id not in exer_id_map:
                continue

            # 获取映射后的习题ID
            mapped_exer_id = exer_id_map[exer_id]

            # 解析知识点序列
            concept_ids = _parse_concept_seq(cpt_seq)

            # 存储该习题对应的知识点
            if mapped_exer_id not in exercise_concepts:
                exercise_concepts[mapped_exer_id] = set()

            for cid in concept_ids:
                if cid in cpt_id_map:
                    exercise_concepts[mapped_exer_id].add(cpt_id_map[cid])

    for df in _iter_source_dfs(train_sources):
        _collect_relations_from_df(df)

    for df in _iter_source_dfs(metadata_sources):
        if "exer_id" not in df.columns or "cpt_seq" not in df.columns:
            continue
        _collect_relations_from_df(df)

    # 填充Q矩阵
    for exer_id, concepts in exercise_concepts.items():
        for cpt_id in concepts:
            q_matrix[exer_id, cpt_id] = 1

    return q_matrix


def build_student_concept_response_stats(
    train_df: pd.DataFrame,
    stu_id_map: Dict[int, int],
    exer_id_map: Dict[int, int],
    q_matrix: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Aggregate train-only response evidence without materializing row-by-concept data.

    The returned tensors contain raw sufficient statistics.  Training removes
    the current row from these totals before the model consumes them; valid and
    test rows use the complete training totals.  Keeping counts and correct
    sums separate makes that leave-one-out contract exact and auditable.
    """
    num_students = len(stu_id_map)
    num_concepts = int(q_matrix.size(1))
    student_concept_count = np.zeros((num_students, num_concepts), dtype=np.float32)
    student_concept_correct = np.zeros((num_students, num_concepts), dtype=np.float32)
    concept_count = np.zeros(num_concepts, dtype=np.float32)
    concept_correct = np.zeros(num_concepts, dtype=np.float32)
    concepts_by_exercise = [
        torch.nonzero(row > 0, as_tuple=False).reshape(-1).cpu().numpy()
        for row in q_matrix
    ]

    global_count = 0.0
    global_correct = 0.0
    for raw_student, raw_exercise, raw_label in zip(
        train_df["stu_id"].values,
        train_df["exer_id"].values,
        train_df["label"].values,
    ):
        student = stu_id_map.get(raw_student)
        exercise = exer_id_map.get(raw_exercise)
        if student is None or exercise is None:
            continue
        label = float(raw_label)
        concepts = concepts_by_exercise[exercise]
        if concepts.size == 0:
            continue
        student_concept_count[student, concepts] += 1.0
        student_concept_correct[student, concepts] += label
        concept_count[concepts] += 1.0
        concept_correct[concepts] += label
        global_count += 1.0
        global_correct += label

    if global_count <= 0.0:
        raise ValueError("response evidence requires at least one mapped training row")
    if np.any(student_concept_correct < 0.0) or np.any(
        student_concept_correct > student_concept_count
    ):
        raise ValueError("student-concept response statistics are inconsistent")

    return {
        "student_concept_count": torch.from_numpy(student_concept_count),
        "student_concept_correct": torch.from_numpy(student_concept_correct),
        "concept_count": torch.from_numpy(concept_count),
        "concept_correct": torch.from_numpy(concept_correct),
        "global_count": torch.tensor(global_count, dtype=torch.float32),
        "global_correct": torch.tensor(global_correct, dtype=torch.float32),
    }


def _uniform_offdiag_prior(num_concepts: int) -> torch.Tensor:
    if num_concepts <= 0:
        raise ValueError(f"num_concepts must be positive, got {num_concepts}")
    if num_concepts == 1:
        return torch.ones(1, 1, dtype=torch.float32)
    eye = torch.eye(num_concepts, dtype=torch.float32)
    offdiag = 1.0 - eye
    return offdiag / float(num_concepts - 1)


def _row_normalize_offdiag_counts(
        counts: torch.Tensor,
        *,
        fallback_prior: Optional[torch.Tensor] = None,
        shrink: float = 0.0,
) -> torch.Tensor:
    C = int(counts.size(0))
    if C == 1:
        return torch.ones(1, 1, dtype=torch.float32)
    counts = counts.detach().float().clone()
    eye = torch.eye(C, dtype=torch.float32, device=counts.device)
    offdiag = 1.0 - eye
    counts = counts * offdiag
    if fallback_prior is None:
        fallback = _uniform_offdiag_prior(C).to(device=counts.device, dtype=counts.dtype)
    else:
        fallback = fallback_prior.detach().float().to(device=counts.device, dtype=counts.dtype) * offdiag
        fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    smooth = max(0.0, float(shrink))
    row_sum = counts.sum(dim=-1, keepdim=True)
    if smooth > 0.0:
        prior = (counts + smooth * fallback) / (row_sum + smooth).clamp(min=1e-12)
    else:
        prior = counts / row_sum.clamp(min=1.0)
        prior = torch.where(row_sum > 0, prior, fallback)
    prior = prior * offdiag
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return prior.to(dtype=torch.float32)


def _keep_observed_support_only(prior: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    """Keep smoothed scores only on observed evidence edges; do not turn smoothing into support."""
    prior = prior.detach().float() * observed_mask.detach().float()
    row_sum = prior.sum(dim=-1, keepdim=True)
    return torch.where(row_sum > 0, prior / row_sum.clamp(min=1e-12), torch.zeros_like(prior))


def _observed_offdiag_support(*priors: torch.Tensor) -> torch.Tensor:
    if not priors:
        raise ValueError("at least one prior matrix is required")
    first = priors[0].detach().float()
    if first.dim() != 2 or first.size(0) != first.size(1):
        raise ValueError(f"prior matrices must be square, got {tuple(first.shape)}")
    C = int(first.size(0))
    support = torch.zeros(C, C, dtype=torch.bool)
    for prior in priors:
        p = prior.detach().float()
        if tuple(p.shape) != (C, C):
            raise ValueError(f"all prior matrices must share shape {(C, C)}, got {tuple(p.shape)}")
        support |= p > 0.0
    if C > 0:
        support &= ~torch.eye(C, dtype=torch.bool)
    return support


def make_degree_random_prior(
        *priors: torch.Tensor,
        seed: int = 20260513,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Random off-diagonal support with the same row-wise degree as observed evidence."""
    support = _observed_offdiag_support(*priors)
    C = int(support.size(0))
    sampled = torch.zeros(C, C, dtype=torch.float32)
    if C > 1:
        gen = torch.Generator().manual_seed(int(seed) + C * 1009)
        for row_idx in range(C):
            degree = int(support[row_idx].sum().item())
            if degree <= 0:
                continue
            candidates = torch.tensor([idx for idx in range(C) if idx != row_idx], dtype=torch.long)
            perm = torch.randperm(candidates.numel(), generator=gen)[:degree]
            cols = candidates[perm]
            sampled[row_idx, cols] = torch.rand(degree, generator=gen).clamp(min=1e-4)
    row_sum = sampled.sum(dim=-1, keepdim=True)
    prior = torch.where(row_sum > 0, sampled / row_sum.clamp(min=1e-12), torch.zeros_like(sampled))
    possible = float(C * max(0, C - 1))
    edge_count = float((sampled > 0.0).sum().item())
    return prior.to(dtype=torch.float32), {
        "degree_random_edge_count": edge_count,
        "degree_random_density": edge_count / possible if possible > 0 else 0.0,
        "degree_random_entropy": _prior_entropy(prior),
        "degree_random_seed": float(seed),
    }


def _prior_entropy(prior: torch.Tensor) -> float:
    if prior.numel() <= 1:
        return 0.0
    p = prior.detach().float().clamp(min=1e-12)
    return float((-(p * p.log()).sum(dim=-1)).mean().item())


def build_item_cooccurrence_prior(q_matrix: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    q = q_matrix.detach().float()
    if q.dim() != 2:
        raise ValueError(f"q_matrix must be 2D, got {tuple(q.shape)}")
    C = int(q.size(1))
    if C == 1:
        prior = torch.ones(1, 1, dtype=torch.float32)
        return prior, {
            "item_observed_edge_count": 0.0,
            "item_prior_density": 0.0,
            "item_prior_entropy": 0.0,
        }
    concept_occurs = (q > 0).float()
    counts = concept_occurs.t().matmul(concept_occurs)
    eye = torch.eye(C, dtype=torch.float32, device=counts.device)
    counts = counts * (1.0 - eye)
    observed_mask = (counts > 0).float()
    prior = _row_normalize_offdiag_counts(counts, shrink=0.0)
    prior = _keep_observed_support_only(prior, observed_mask)
    possible = float(C * (C - 1))
    observed = float((counts > 0).float().sum().item())
    return prior, {
        "item_observed_edge_count": observed,
        "item_prior_density": observed / possible if possible > 0 else 0.0,
        "item_prior_entropy": _prior_entropy(prior),
    }


def build_student_coexposure_prior(
    train_sources: list,
    cpt_id_map: Dict[int, int],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build a permutation-invariant train-only student co-exposure prior.

    Each student contributes equal total support mass per observed concept,
    regardless of history length. This uses only ``stu_id`` and ``cpt_seq``;
    correctness labels and physical CSV row order are irrelevant.
    """
    concept_count = len(cpt_id_map)
    if concept_count <= 0:
        raise ValueError("cpt_id_map must not be empty")
    if concept_count == 1:
        prior = torch.ones(1, 1, dtype=torch.float32)
        return prior, {
            "exposure_student_count": 0.0,
            "exposure_multi_concept_student_count": 0.0,
            "exposure_mean_concepts_per_student": 0.0,
            "exposure_pair_mass": 0.0,
            "exposure_observed_edge_count": 0.0,
            "exposure_prior_density": 0.0,
            "exposure_prior_entropy": 0.0,
        }

    student_concepts: Dict[object, set] = {}
    for dataframe in _iter_source_dfs(train_sources):
        if "stu_id" not in dataframe.columns or "cpt_seq" not in dataframe.columns:
            continue
        for student_id, student_rows in dataframe.groupby("stu_id", sort=False):
            concepts = student_concepts.setdefault(student_id, set())
            for sequence in student_rows["cpt_seq"].values:
                concepts.update(
                    cpt_id_map[concept_id]
                    for concept_id in _parse_concept_seq(sequence)
                    if concept_id in cpt_id_map
                )

    histories = [sorted(concepts) for concepts in student_concepts.values() if concepts]
    row_indices: List[int] = []
    concept_indices: List[int] = []
    weights: List[float] = []
    for row_index, concepts in enumerate(histories):
        if len(concepts) < 2:
            continue
        weight = 1.0 / math.sqrt(float(len(concepts) - 1))
        row_indices.extend([row_index] * len(concepts))
        concept_indices.extend(concepts)
        weights.extend([weight] * len(concepts))
    incidence = csr_matrix(
        (weights, (row_indices, concept_indices)),
        shape=(len(histories), concept_count),
        dtype=np.float32,
    )
    counts = torch.from_numpy((incidence.transpose() @ incidence).toarray()).float()

    eye = torch.eye(concept_count, dtype=torch.float32)
    counts *= 1.0 - eye
    observed_mask = (counts > 0).float()
    prior = _row_normalize_offdiag_counts(counts, shrink=0.0)
    prior = _keep_observed_support_only(prior, observed_mask)
    possible = float(concept_count * (concept_count - 1))
    observed = float((counts > 0).float().sum().item())
    multi_concept_histories = sum(len(concepts) >= 2 for concepts in histories)
    mean_concepts = (
        float(sum(len(concepts) for concepts in histories)) / float(len(histories))
        if histories
        else 0.0
    )
    return prior, {
        "exposure_student_count": float(len(histories)),
        "exposure_multi_concept_student_count": float(multi_concept_histories),
        "exposure_mean_concepts_per_student": mean_concepts,
        "exposure_pair_mass": float(counts.sum().item()),
        "exposure_observed_edge_count": observed,
        "exposure_prior_density": observed / possible if possible > 0 else 0.0,
        "exposure_prior_entropy": _prior_entropy(prior),
    }


def create_dataloaders(
        train_file: str,
        val_file: str,
        test_file: Optional[str],
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
        min_stu_interactions: int = 0,
        min_exer_interactions: int = 0,
        logger=None,
        dataset_name: Optional[str] = None,
        graph_prior_mode: str = "evidence",
        seed: int = 42,
        load_test: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], dict]:
    """
    创建训练、验证和测试的数据加载器，并执行统一的数据清洗

    Args:
        train_file: 训练集CSV文件路径
        val_file: 验证集CSV文件路径
        test_file: 测试集CSV文件路径；纯训练阶段可传 None
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        shuffle_train: 是否打乱训练集
        min_stu_interactions: 学生最小答题数阈值（<该值的学生将被过滤，0 表示不启用）
        min_exer_interactions: 习题最小被作答数阈值（<该值的题目将被过滤，0 表示不启用）
        logger: 可选的 logger，对清洗过程进行记录
        seed: 训练 DataLoader 的独立随机种子
        load_test: 是否读取测试集；训练/选模时必须为 False

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
    if load_test and not test_file:
        raise ValueError("test_file is required when load_test=True")
    test_df = pd.read_csv(test_file) if load_test else None

    required_columns = {"stu_id", "exer_id", "cpt_seq", "label"}
    split_frames = [("train", train_df), ("valid", val_df)]
    if test_df is not None:
        split_frames.append(("test", test_df))
    for split_name, dataframe in split_frames:
        missing = sorted(required_columns - set(dataframe.columns))
        if missing:
            raise ValueError(f"{split_name} split is missing required columns: {missing}")
        labels = pd.to_numeric(dataframe["label"], errors="coerce")
        if labels.isna().any() or not labels.between(0.0, 1.0).all():
            raise ValueError(f"{split_name} labels must be finite values in [0, 1]")
        dataframe["label"] = labels.astype(float)
        if dataframe[["stu_id", "exer_id"]].isna().any().any():
            raise ValueError(f"{split_name} stu_id/exer_id values must not be missing")

    if train_df.empty:
        raise ValueError("training split is empty")

    # ============ 统一的清洗逻辑 ============

    # --- 学生冷启动过滤 ---
    if min_stu_interactions > 0:
        stu_counts = train_df.groupby("stu_id").size()
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
        train_df, dropped = _filter_by_students(train_df)
        dropped_total += dropped
        val_df, dropped = _filter_by_students(val_df)
        dropped_total += dropped
        if test_df is not None:
            test_df, dropped = _filter_by_students(test_df)
            dropped_total += dropped

        split_scope = "train/valid/test" if test_df is not None else "train/valid"
        log(f"[数据清洗] 学生冷启动过滤：在 {split_scope} 中共删除 {dropped_total} 条记录。")
    else:
        log("[数据清洗] 跳过学生冷启动过滤（min_stu_interactions <= 0）。")

    # --- 题目冷门过滤 ---
    if min_exer_interactions > 0:
        exer_counts = train_df.groupby("exer_id").size()
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
        train_df, dropped = _filter_by_items(train_df)
        dropped_total += dropped
        val_df, dropped = _filter_by_items(val_df)
        dropped_total += dropped
        if test_df is not None:
            test_df, dropped = _filter_by_items(test_df)
            dropped_total += dropped

        split_scope = "train/valid/test" if test_df is not None else "train/valid"
        log(f"[数据清洗] 题目冷门过滤：在 {split_scope} 中共删除 {dropped_total} 条记录。")
    else:
        log("[数据清洗] 跳过题目冷门过滤（min_exer_interactions <= 0）。")

    if train_df.empty:
        raise ValueError("training split is empty after interaction filtering")
    if train_df["label"].nunique(dropna=True) < 2:
        raise ValueError("training split must contain both outcome classes")

    # 2. 构建ID映射 & Q矩阵（严格基于 train-only）
    train_sources = [train_df]

    log("正在构建ID映射...")
    stu_id_map, exer_id_map, cpt_id_map = build_id_mappings(train_sources=train_sources)

    log("正在构建Q矩阵...")
    q_matrix = build_q_matrix(train_sources=train_sources, exer_id_map=exer_id_map, cpt_id_map=cpt_id_map)

    def _zero_item_prior(num_concepts: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        return torch.zeros(num_concepts, num_concepts, dtype=torch.float32), {
            "item_observed_edge_count": 0.0,
            "item_prior_density": 0.0,
            "item_prior_entropy": 0.0,
        }

    def _zero_exposure_prior(num_concepts: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        return torch.zeros(num_concepts, num_concepts, dtype=torch.float32), {
            "exposure_student_count": 0.0,
            "exposure_multi_concept_student_count": 0.0,
            "exposure_mean_concepts_per_student": 0.0,
            "exposure_pair_mass": 0.0,
            "exposure_observed_edge_count": 0.0,
            "exposure_prior_density": 0.0,
            "exposure_prior_entropy": 0.0,
        }

    prior_mode = str(graph_prior_mode or "evidence").strip().lower()
    valid_prior_modes = {
        "evidence",
        "item_only",
        "exposure_only",
        "degree_random",
    }
    if prior_mode not in valid_prior_modes:
        raise ValueError(f"graph_prior_mode must be one of {sorted(valid_prior_modes)}, got {graph_prior_mode!r}")
    num_concepts = len(cpt_id_map)

    def _build_evidence_priors() -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, Dict[str, float]]:
        item_prior, item_stats = build_item_cooccurrence_prior(q_matrix)
        exposure_prior, exposure_stats = build_student_coexposure_prior(
            train_sources=train_sources,
            cpt_id_map=cpt_id_map,
        )
        return item_prior, item_stats, exposure_prior, exposure_stats

    if prior_mode == "degree_random":
        evidence_item, item_prior_stats, evidence_exposure, exposure_prior_stats = _build_evidence_priors()
        item_prior_matrix, random_stats = make_degree_random_prior(evidence_item, evidence_exposure)
        item_prior_stats = {**item_prior_stats, **random_stats}
        exposure_prior_matrix, _ = _zero_exposure_prior(num_concepts)
        log("[Graph Prior] mode=degree_random: using row-degree-matched random support control prior.")
    else:
        if prior_mode == "exposure_only":
            item_prior_matrix, item_prior_stats = _zero_item_prior(num_concepts)
            log("[Graph Prior] mode=exposure_only: item co-occurrence evidence disabled.")
        else:
            item_prior_matrix, item_prior_stats = build_item_cooccurrence_prior(q_matrix)

        if prior_mode == "item_only":
            exposure_prior_matrix, exposure_prior_stats = _zero_exposure_prior(num_concepts)
            log("[Graph Prior] mode=item_only: student co-exposure evidence disabled.")
        else:
            exposure_prior_matrix, exposure_prior_stats = build_student_coexposure_prior(
                train_sources=train_sources,
                cpt_id_map=cpt_id_map,
            )
    graph_prior_stats = {**item_prior_stats, **exposure_prior_stats}
    graph_prior_stats["graph_prior_mode"] = prior_mode
    log(
        "[Graph Prior] train-only evidence: "
        f"mode={prior_mode}, "
        f"item_edges={graph_prior_stats['item_observed_edge_count']:.0f}, "
        f"item_density={graph_prior_stats['item_prior_density']:.4f}, "
        f"item_entropy={graph_prior_stats['item_prior_entropy']:.4f}, "
        f"exposure_students={graph_prior_stats['exposure_student_count']:.0f}, "
        f"exposure_mean_concepts={graph_prior_stats['exposure_mean_concepts_per_student']:.2f}, "
        f"exposure_pair_mass={graph_prior_stats['exposure_pair_mass']:.1f}, "
        f"exposure_edges={graph_prior_stats['exposure_observed_edge_count']:.0f}, "
        f"exposure_density={graph_prior_stats['exposure_prior_density']:.4f}, "
        f"exposure_entropy={graph_prior_stats['exposure_prior_entropy']:.4f}"
    )

    def _filter_seen_support(df: pd.DataFrame, split_name: str):
        before = len(df)
        if before == 0:
            log(f"[数据清洗] {split_name} split 为空，跳过 train-seen 过滤。")
            return df.copy().reset_index(drop=True), before, before, 1.0

        keep_mask = (
            df["stu_id"].isin(stu_id_map.keys())
            & df["exer_id"].isin(exer_id_map.keys())
        )
        df_new = df[keep_mask].reset_index(drop=True)
        after = len(df_new)
        coverage = after / before if before > 0 else 1.0
        dropped = before - after
        log(
            f"[数据清洗] {split_name} train-seen 过滤：before={before}, after={after}, "
            f"dropped={dropped}, coverage={coverage:.2%}"
        )
        return df_new, before, after, coverage

    val_df, val_total_rows_raw, val_seen_rows, val_seen_coverage = _filter_seen_support(val_df, "Valid")
    if val_df.empty:
        raise ValueError("validation split has no train-seen student-item rows")
    if val_df["label"].nunique(dropna=True) < 2:
        raise ValueError("validation split must contain both outcome classes after train-seen filtering")
    if test_df is not None:
        test_df, test_total_rows_raw, test_seen_rows, test_seen_coverage = _filter_seen_support(
            test_df,
            "Test",
        )
        if test_df.empty:
            raise ValueError("test split has no train-seen student-item rows")
        if test_df["label"].nunique(dropna=True) < 2:
            raise ValueError("test split must contain both outcome classes after train-seen filtering")

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
    if test_df is not None:
        _log_concept_stats("Test", test_df)
    log("[统计] 概念数量统计完成。")

    response_evidence_stats = build_student_concept_response_stats(
        train_df=train_df,
        stu_id_map=stu_id_map,
        exer_id_map=exer_id_map,
        q_matrix=q_matrix,
    )
    observed_student_concepts = response_evidence_stats["student_concept_count"] > 0
    log(
        "[Response Evidence] train-only sufficient statistics: "
        f"observed_student_concepts={int(observed_student_concepts.sum().item())}, "
        f"coverage={float(observed_student_concepts.float().mean().item()):.2%}, "
        f"global_rate={float(response_evidence_stats['global_correct'].item() / response_evidence_stats['global_count'].item()):.4f}"
    )

    # 3. 创建数据集（直接传 DataFrame，避免重复读盘）
    log("正在加载数据集...")
    train_dataset = CognitiveDiagnosisDataset(train_df, stu_id_map, exer_id_map, cpt_id_map)
    val_dataset = CognitiveDiagnosisDataset(val_df, stu_id_map, exer_id_map, cpt_id_map)
    test_dataset = (
        CognitiveDiagnosisDataset(test_df, stu_id_map, exer_id_map, cpt_id_map)
        if test_df is not None
        else None
    )

    # 4. 创建 DataLoader
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_generator = torch.Generator()
    train_generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        generator=train_generator,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    # 5. 收集信息
    info_dict = {
        'num_students': len(stu_id_map),
        'num_exercises': len(exer_id_map),
        'num_concepts': len(cpt_id_map),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'stu_id_map': stu_id_map,
        'exer_id_map': exer_id_map,
        'cpt_id_map': cpt_id_map,
        'stu_id_reverse_map': {v: k for k, v in stu_id_map.items()},
        'exer_id_reverse_map': {v: k for k, v in exer_id_map.items()},
        'cpt_id_reverse_map': {v: k for k, v in cpt_id_map.items()},
        'q_matrix': q_matrix,
        'item_prior_matrix': item_prior_matrix,
        'exposure_prior_matrix': exposure_prior_matrix,
        'response_evidence_stats': response_evidence_stats,
        'graph_prior_stats': graph_prior_stats,
        'exposure_prior_disabled': bool(prior_mode in {"item_only", "degree_random"}),
        'item_prior_disabled': bool(prior_mode == "exposure_only"),
        'graph_prior_mode': prior_mode,
        # 每道题的概念数量，与 exer_id 内部索引对齐。
        'concepts_per_exercise': concepts_per_exercise,
        'train_only_split_hygiene': True,
        'test_sealed_during_training': not load_test,
        'val_total_rows_raw': int(val_total_rows_raw),
        'val_seen_rows': int(val_seen_rows),
        'val_seen_coverage': float(val_seen_coverage),
    }
    if test_dataset is not None:
        info_dict.update(
            {
                'test_size': len(test_dataset),
                'test_total_rows_raw': int(test_total_rows_raw),
                'test_seen_rows': int(test_seen_rows),
                'test_seen_coverage': float(test_seen_coverage),
            }
        )

    return train_loader, val_loader, test_loader, info_dict
