import torch
import numpy as np

def build_interaction_graph(triplets, num_students, num_exercises):
    """
    Builds a single, unified interaction graph from all student-exercise interactions.
    """
    exer_offset = num_students
    
    triplets_arr = np.array(triplets, dtype=np.int64)
    
    if len(triplets_arr) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    s_ids = triplets_arr[:, 0]
    e_ids = triplets_arr[:, 1] + exer_offset
    
    # Create edges for the bipartite graph (student -> exercise and exercise -> student)
    src = np.concatenate([s_ids, e_ids])
    dst = np.concatenate([e_ids, s_ids])
    
    edge_index = torch.from_numpy(np.stack([src, dst])).long()
    
    return edge_index