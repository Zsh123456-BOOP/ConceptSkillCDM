import torch
import numpy as np

def build_interaction_graph(triplets, num_students, num_exercises):
    """
    将所有学生-题目的交互构建成一张统一的二分图。
    """
    exer_offset = num_students
    
    triplets_arr = np.array(triplets, dtype=np.int64)
    
    if len(triplets_arr) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    s_ids = triplets_arr[:, 0]
    e_ids = triplets_arr[:, 1] + exer_offset
    
    # 构建二分图的边（学生 -> 题目 以及 题目 -> 学生）
    src = np.concatenate([s_ids, e_ids])
    dst = np.concatenate([e_ids, s_ids])
    
    edge_index = torch.from_numpy(np.stack([src, dst])).long()
    
    return edge_index
