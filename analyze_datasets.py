import os
import argparse
import numpy as np
import pandas as pd


RENAME_MAP = {
    "stu_id": "student_id",
    "exer_id": "item_id",
    "label": "correct",
    "cpt_seq": "concept_ids",
}


def load_split(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.rename(columns=RENAME_MAP)
    required_cols = ["student_id", "item_id", "correct", "concept_ids"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"{csv_path} 缺少列: {col}，当前列有: {list(df.columns)}"
            )
    return df


def parse_concepts(series: pd.Series):
    """
    把 concept_ids 列解析成 int 列表，返回：
    - all_concepts: 所有出现的concept id（flatten）
    - lens: 每条样本对应的 concept 数量
    """
    all_concepts = []
    lens = []
    for val in series:
        s = str(val)
        s = s.replace('"', "").replace("'", "").replace(",", ";")
        ids = [int(x) for x in s.split(";") if x.strip()]
        all_concepts.extend(ids)
        lens.append(len(ids))
    return np.array(all_concepts, dtype=np.int64), np.array(lens, dtype=np.int64)


def summarize_dataset(base_dir: str, name: str):
    base_path = os.path.join(base_dir, name)
    train_path = os.path.join(base_path, "train.csv")
    valid_path = os.path.join(base_path, "valid.csv")
    test_path = os.path.join(base_path, "test.csv")

    print("=" * 80)
    print(f"📊 Dataset: {name}")
    print(f"Base path: {base_path}")

    train_df = load_split(train_path)
    valid_df = load_split(valid_path)
    test_df = load_split(test_path)

    # 基本规模
    n_train = len(train_df)
    n_valid = len(valid_df)
    n_test = len(test_df)
    n_total = n_train + n_valid + n_test

    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    students = all_df["student_id"].astype(int).unique()
    items = all_df["item_id"].astype(int).unique()
    all_concepts_flat, lens_all = parse_concepts(all_df["concept_ids"])
    concepts = np.unique(all_concepts_flat)

    print(f"- Interactions: train={n_train}, valid={n_valid}, test={n_test}, total={n_total}")
    print(f"- #Students: {len(students)}")
    print(f"- #Items:    {len(items)}")
    print(f"- #Concepts: {len(concepts)}")

    # 每条样本的概念数量分布
    if len(lens_all) > 0:
        print("- Concepts per interaction:")
        print(f"  - mean: {lens_all.mean():.3f}")
        print(f"  - std:  {lens_all.std():.3f}")
        print(f"  - min:  {lens_all.min()}")
        print(f"  - 50%:  {np.percentile(lens_all, 50):.1f}")
        print(f"  - 90%:  {np.percentile(lens_all, 90):.1f}")
        print(f"  - 95%:  {np.percentile(lens_all, 95):.1f}")
        print(f"  - max:  {lens_all.max()}")

    # 每个学生的交互数量
    inter_per_stu = all_df.groupby("student_id")["item_id"].count().values
    print("- Interactions per student:")
    print(f"  - mean: {inter_per_stu.mean():.1f}")
    print(f"  - 50%:  {np.percentile(inter_per_stu, 50):.1f}")
    print(f"  - 90%:  {np.percentile(inter_per_stu, 90):.1f}")
    print(f"  - 95%:  {np.percentile(inter_per_stu, 95):.1f}")
    print(f"  - max:  {inter_per_stu.max()}")

    # 每个题目的交互数量（用于估计 item_emb 稀疏程度）
    inter_per_item = all_df.groupby("item_id")["student_id"].count().values
    print("- Interactions per item:")
    print(f"  - mean: {inter_per_item.mean():.1f}")
    print(f"  - 50%:  {np.percentile(inter_per_item, 50):.1f}")
    print(f"  - 90%:  {np.percentile(inter_per_item, 90):.1f}")
    print(f"  - 95%:  {np.percentile(inter_per_item, 95):.1f}")
    print(f"  - max:  {inter_per_item.max()}")

    print()  # blank line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据集根目录（包含 sample/assist_09/assist_17 等子目录）",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["sample", "assist_09", "assist_17"],
        help="要分析的数据集名称列表",
    )
    args = parser.parse_args()

    for name in args.datasets:
        try:
            summarize_dataset(args.data_dir, name)
        except Exception as e:
            print("=" * 80)
            print(f"❌ 处理数据集 {name} 时出错: {e}")


if __name__ == "__main__":
    main()
