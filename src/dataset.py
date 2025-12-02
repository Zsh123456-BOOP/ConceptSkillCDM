import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CognitiveDataset(Dataset):
    """
    基础 Dataset：
    只存 (stu_idx, exer_idx, label) 三元组，
    Q-matrix 在 DataProcessor 里统一处理并在 collate 时注入。
    """
    def __init__(self, triplets):
        """
        triplets: List[(stu_idx, exer_idx, label)]
        """
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        stu_idx, exer_idx, label = self.triplets[idx]
        return stu_idx, exer_idx, label


class CognitiveDataProcessor:
    """
    数据处理与清洗：
    - 读取 train/valid/test 以及可选的分层测试集
    - 按 config 中的阈值统一执行清洗策略（所有数据集一致）
    - junyi 额外做 Transductive 用户对齐
    - 统一 ID 映射，构建 Q-matrix
    - 生成 triplets 和 DataLoader
    """
    def __init__(self, args, logger):
        self.data_dir = os.path.join(args.data_root, args.dataset)
        self.args = args
        self.logger = logger

        self.logger.info(f">>> Processing Data for Dataset: {args.dataset} ...")
        self._load_and_process()

    # ================== 主流程 ==================
    def _load_and_process(self):
        # 1. 读取原始 CSV
        self.train_data = self._safe_read_csv(self.args.train_file)
        self.test_data = self._safe_read_csv(self.args.test_file)

        valid_path = os.path.join(self.data_dir, self.args.valid_file)
        if not os.path.exists(valid_path):
            valid_path = os.path.join(self.data_dir, 'val.csv')
        if os.path.exists(valid_path):
            self.valid_data = pd.read_csv(valid_path)
        else:
            raise FileNotFoundError(f"Validation file not found in {self.data_dir}")

        # 分层测试集（可选）
        self.high_test = self._safe_read_csv(self.args.high_test_file, optional=True)
        self.med_test = self._safe_read_csv(self.args.medium_test_file, optional=True)
        self.low_test = self._safe_read_csv(self.args.low_test_file, optional=True)

        # ============ 辅助函数：批量过滤 ============
        def apply_filter_to_all_splits(keep_ids, col_name, log_desc):
            dropped_count = 0
            for attr in ['train_data', 'valid_data', 'test_data',
                         'high_test', 'med_test', 'low_test']:
                df = getattr(self, attr)
                if df is not None:
                    before = len(df)
                    df_new = df[df[col_name].isin(keep_ids)].reset_index(drop=True)
                    after = len(df_new)
                    setattr(self, attr, df_new)
                    if before != after:
                        dropped_count += (before - after)
            self.logger.info(f"[DataClean] {log_desc}: Dropped {dropped_count} rows across all splits.")

        # ============ 统一的清洗逻辑（由 config 控制） ============
        combined = pd.concat(
            [self.train_data, self.valid_data, self.test_data],
            ignore_index=True
        )

        # --- 学生冷启动过滤（所有数据集生效） ---
        if self.args.min_stu_interactions > 0:
            stu_counts = combined.groupby("stu_id").size()
            keep_users = set(
                stu_counts[stu_counts >= self.args.min_stu_interactions].index
            )
            self.logger.info(
                f"[DataClean] Filtering Students < {self.args.min_stu_interactions} interactions "
                f"(Removed {len(stu_counts) - len(keep_users)} users)"
            )
            apply_filter_to_all_splits(
                keep_users, "stu_id",
                f"Cold-Start Users (<{self.args.min_stu_interactions})"
            )
        else:
            self.logger.info(
                "[DataClean] [Skip] Student cold-start filtering disabled "
                "(min_stu_interactions <= 0)."
            )

        # --- 题目冷门过滤（所有数据集生效） ---
        if self.args.min_exer_interactions > 0:
            combined = pd.concat(
                [self.train_data, self.valid_data, self.test_data],
                ignore_index=True
            )
            exer_counts = combined.groupby("exer_id").size()
            keep_items = set(
                exer_counts[exer_counts >= self.args.min_exer_interactions].index
            )
            self.logger.info(
                f"[DataClean] Filtering Items < {self.args.min_exer_interactions} interactions "
                f"(Removed {len(exer_counts) - len(keep_items)} items)"
            )
            apply_filter_to_all_splits(
                keep_items, "exer_id",
                f"Cold Items (<{self.args.min_exer_interactions})"
            )
        else:
            self.logger.info(
                "[DataClean] [Skip] Item cold filtering disabled "
                "(min_exer_interactions <= 0)."
            )

        # --- 毒题清洗（所有数据集生效，是否开启由 min_poison_count 控制） ---
        if self.args.min_poison_count > 0:
            combined = pd.concat(
                [self.train_data, self.valid_data, self.test_data],
                ignore_index=True
            )
            item_stats = combined.groupby("exer_id")["label"].agg(
                count="count", correct_rate="mean"
            ).reset_index()

            poison_mask = (
                (item_stats["count"] >= self.args.min_poison_count)
                & ((item_stats["correct_rate"] <= 0.0)
                   | (item_stats["correct_rate"] >= 1.0))
            )
            poison_items = item_stats.loc[poison_mask, :]
            poison_ids = set(poison_items["exer_id"].tolist())

            if len(poison_ids) > 0:
                self.logger.info(
                    f"[DataClean] Detected {len(poison_ids)} toxic items "
                    f"(count >= {self.args.min_poison_count} & Acc=0.0 or 1.0). Removing..."
                )
                keep_non_toxic = set(
                    item_stats[~item_stats["exer_id"].isin(poison_ids)]["exer_id"]
                )
                apply_filter_to_all_splits(
                    keep_non_toxic, "exer_id", "Toxic Items"
                )
            else:
                self.logger.info(
                    "[DataClean] No toxic items detected under current min_poison_count."
                )
        else:
            self.logger.info(
                "[DataClean] [Skip] Toxic item filtering disabled "
                "(min_poison_count <= 0)."
            )

        # ============ 统一 Student ID 类型 ============
        self.logger.info("Normalizing Student IDs to strings...")
        for attr in ['train_data', 'valid_data', 'test_data',
                     'high_test', 'med_test', 'low_test']:
            df = getattr(self, attr)
            if df is not None and not df.empty:
                df['stu_id'] = df['stu_id'].astype(str)

        # ============ Junyi 用户对齐 (Transductive，仅 junyi) ============
        if self.args.dataset == "junyi":
            self.logger.info(
                "[Alignment] Filtering unseen users in Valid/Test sets "
                "(Transductive Setting)..."
            )
            train_users = set(self.train_data['stu_id'].unique())

            def filter_unseen(df, name):
                if df is None or df.empty:
                    return df
                original_len = len(df)
                df_filtered = df[df['stu_id'].isin(train_users)].reset_index(drop=True)
                new_len = len(df_filtered)
                if original_len != new_len:
                    self.logger.info(
                        f"   -> [{name}] Dropped {original_len - new_len} rows "
                        "from unseen users."
                    )
                return df_filtered

            self.valid_data = filter_unseen(self.valid_data, "valid")
            self.test_data = filter_unseen(self.test_data, "test")
            if self.high_test is not None:
                self.high_test = filter_unseen(self.high_test, "high")
            if self.med_test is not None:
                self.med_test = filter_unseen(self.med_test, "med")
            if self.low_test is not None:
                self.low_test = filter_unseen(self.low_test, "low")

        # ============ 构建 ID 映射与概念集合 ============
        all_data = pd.concat([self.train_data, self.valid_data, self.test_data])

        self.stu_ids = sorted(all_data['stu_id'].unique())
        self.exer_ids = sorted(all_data['exer_id'].unique())

        all_concepts = set()
        if 'cpt_seq' in all_data.columns:
            cpt_series = all_data['cpt_seq'].dropna().astype(str)
            for cpt_seq in cpt_series:
                clean_seq = cpt_seq.strip('"').strip("'")
                if clean_seq:
                    try:
                        concepts = list(map(int, clean_seq.split(',')))
                        all_concepts.update(concepts)
                    except Exception:
                        pass
        self.cpt_ids = sorted(all_concepts)

        self.stu2idx = {id_: i for i, id_ in enumerate(self.stu_ids)}
        self.exer2idx = {id_: i for i, id_ in enumerate(self.exer_ids)}
        self.cpt2idx = {id_: i for i, id_ in enumerate(self.cpt_ids)}

        self.num_students = len(self.stu_ids)
        self.num_exercises = len(self.exer_ids)
        self.num_concepts = len(self.cpt_ids)

        self.logger.info(
            f"Final Stats: Stu={self.num_students}, "
            f"Exer={self.num_exercises}, Cpt={self.num_concepts}"
        )

        # ============ 三元组构建 ============
        self.logger.info("Processing Triplets...")
        self.train_triplets = self._process_triplets_fast(self.train_data)
        self.valid_triplets = self._process_triplets_fast(self.valid_data)
        self.test_triplets = self._process_triplets_fast(self.test_data)

        self.high_triplets = self._process_triplets_fast(self.high_test) if self.high_test is not None else []
        self.med_triplets = self._process_triplets_fast(self.med_test) if self.med_test is not None else []
        self.low_triplets = self._process_triplets_fast(self.low_test) if self.low_test is not None else []

        # ============ Q-matrix ============
        self.logger.info("Building Q-Matrix...")
        self.Q_matrix = self._build_q_matrix(all_data)  # (num_exercises, num_concepts)

        # 学生统计（用于日志 & 分位数）
        self.student_stats = self._compute_student_stats()

    # ================== 工具函数 ==================
    def _safe_read_csv(self, filename, optional=False):
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        if optional:
            return None
        raise FileNotFoundError(f"Required file not found: {path}")

    def _build_q_matrix(self, all_data):
        """
        构建 Q-matrix: (num_exercises, num_concepts)
        Q[e_idx, c_idx] = 1 表示该题目涉及该知识点。
        """
        Q = torch.zeros(self.num_exercises, self.num_concepts)
        if 'cpt_seq' not in all_data.columns:
            return Q
        unique_exer = all_data[['exer_id', 'cpt_seq']].dropna().drop_duplicates('exer_id')
        for _, row in unique_exer.iterrows():
            e_idx = self.exer2idx.get(row['exer_id'])
            if e_idx is None:
                continue
            clean_seq = str(row['cpt_seq']).strip('"').strip("'")
            if not clean_seq:
                continue
            try:
                concepts = list(map(int, clean_seq.split(',')))
                for c in concepts:
                    c_idx = self.cpt2idx.get(c)
                    if c_idx is not None:
                        Q[e_idx, c_idx] = 1
            except Exception:
                continue
        return Q

    def _process_triplets_fast(self, data):
        """
        将 DataFrame 转为 (stu_idx, exer_idx, label) 列表。
        """
        if data is None or data.empty:
            return []
        s_idxs = data['stu_id'].map(self.stu2idx)
        e_idxs = data['exer_id'].map(self.exer2idx)

        mask = s_idxs.notna() & e_idxs.notna()
        s_list = s_idxs[mask].astype(int).tolist()
        e_list = e_idxs[mask].astype(int).tolist()
        l_list = data.loc[mask, 'label'].astype(int).tolist()

        return list(zip(s_list, e_list, l_list))

    def _compute_student_stats(self):
        """
        统计每个学生的 (correct_count, wrong_count, total_count)，
        并基于训练集历史正确率估计分位数阈值 q_low / q_high（写入 args）。
        """
        stats = torch.zeros(self.num_students, 3)
        if not self.train_triplets:
            self.q_low = 0.4
            self.q_high = 0.7
            setattr(self.args, "q_low", self.q_low)
            setattr(self.args, "q_high", self.q_high)
            self.logger.warning(
                "[Grouping] No train_triplets found. "
                "Using default thresholds q_low=0.4, q_high=0.7"
            )
            return stats

        triplets_arr = torch.tensor(self.train_triplets, dtype=torch.long)
        s_ids = triplets_arr[:, 0]
        labels = triplets_arr[:, 2]
        ones = torch.ones_like(labels, dtype=torch.float32)

        # total
        stats[:, 2].scatter_add_(0, s_ids, ones)
        # correct
        correct_mask = (labels == 1)
        stats[:, 0].scatter_add_(0, s_ids[correct_mask], ones[correct_mask])
        # wrong
        wrong_mask = (labels == 0)
        stats[:, 1].scatter_add_(0, s_ids[wrong_mask], ones[wrong_mask])

        total = stats[:, 2]
        valid_mask = total > 0
        if valid_mask.sum() >= 3:
            acc = stats[:, 0] / (total + 1e-6)
            valid_acc = acc[valid_mask]

            low_q = float(torch.quantile(
                valid_acc,
                torch.tensor(self.args.low_quantile, dtype=torch.float32)
            ))
            high_q = float(torch.quantile(
                valid_acc,
                torch.tensor(self.args.high_quantile, dtype=torch.float32)
            ))

            self.q_low = low_q
            self.q_high = high_q
            self.logger.info(
                f"[Grouping] Using accuracy quantiles for groups: "
                f"low_q={self.q_low:.3f} (p={self.args.low_quantile}), "
                f"high_q={self.q_high:.3f} (p={self.args.high_quantile})"
            )
        else:
            self.q_low = 0.4
            self.q_high = 0.7
            self.logger.warning(
                "[Grouping] Not enough students with interactions to estimate quantiles. "
                "Fallback to fixed thresholds q_low=0.4, q_high=0.7"
            )

        setattr(self.args, "q_low", self.q_low)
        setattr(self.args, "q_high", self.q_high)

        return stats

    # ================== DataLoader 接口 ==================
    def get_loaders(self):
        """
        返回：
            train_loader, valid_loader, test_loader,
            high_loader, med_loader, low_loader

        每个 batch: (stu_ids, exer_ids, labels, q_mask)
        """
        Q_matrix = self.Q_matrix  # (num_exercises, num_concepts)

        def collate(batch, q_matrix=Q_matrix):
            """
            batch: List[(stu_idx, exer_idx, label)]
            """
            stu_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
            exer_ids = torch.tensor([x[1] for x in batch], dtype=torch.long)
            labels = torch.tensor([x[2] for x in batch], dtype=torch.float32)

            # q_mask: (B, num_concepts) = Q_matrix[exer_ids]
            q_mask = q_matrix[exer_ids] if q_matrix is not None else None

            return stu_ids, exer_ids, labels, q_mask

        kw = {
            'batch_size': self.args.batch_size,
            'collate_fn': collate,
            'num_workers': 0,
            'pin_memory': False,
        }

        train_loader = DataLoader(
            CognitiveDataset(self.train_triplets),
            shuffle=True,
            **kw
        )
        valid_loader = DataLoader(
            CognitiveDataset(self.valid_triplets),
            shuffle=False,
            **kw
        )
        test_loader = DataLoader(
            CognitiveDataset(self.test_triplets),
            shuffle=False,
            **kw
        )

        high_loader = DataLoader(
            CognitiveDataset(self.high_triplets),
            shuffle=False,
            **kw
        ) if self.high_triplets else None
        med_loader = DataLoader(
            CognitiveDataset(self.med_triplets),
            shuffle=False,
            **kw
        ) if self.med_triplets else None
        low_loader = DataLoader(
            CognitiveDataset(self.low_triplets),
            shuffle=False,
            **kw
        ) if self.low_triplets else None

        return train_loader, valid_loader, test_loader, high_loader, med_loader, low_loader
