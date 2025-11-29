import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# 设置随机种子以便复现
torch.manual_seed(42)

class ConceptRelationDiscovery(nn.Module):
    """
    模块一：无监督概念关系发现 (Final Fix)
    """
    def __init__(self, num_concepts, input_dim, num_heads=2):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.W_Q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_K = nn.Linear(input_dim, input_dim, bias=False)
        
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, concept_emb):
        N, d = concept_emb.shape
        Q = self.W_Q(concept_emb).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.W_K(concept_emb).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Attention Scores [heads, N, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # [关键修改] 屏蔽对角线：防止模型只学到“自己连自己”
        # 创建一个对角线 Mask，设为负无穷大
        diag_mask = torch.eye(N, device=scores.device).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(self.num_heads, -1, -1)
        scores.masked_fill_(diag_mask, -1e9)
        
        # 使用 Softmax 生成概率图 (行归一化)
        A = torch.softmax(scores / self.temperature, dim=-1)
        return A

def loss_entropy(A):
    # 熵最小化，鼓励稀疏
    entropy = -torch.sum(A * torch.log(A + 1e-8), dim=-1)
    return torch.mean(entropy)

def loss_dag(A):
    # DAG 约束
    expm_A = torch.matrix_exp(A)
    trace = torch.trace(expm_A)
    return trace - A.shape[0]

def loss_symmetry(A):
    # 对称性约束
    return torch.mean((A - A.transpose(-1, -2)) ** 2)

# ==========================================
# 模拟测试流程
# ==========================================

print("=== 步骤 1: 初始化 (Diagonal Masking版) ===")
NUM_CONCEPTS = 10 
EMBED_DIM = 16
NUM_HEADS = 2

concept_embeddings = nn.Parameter(torch.randn(NUM_CONCEPTS, EMBED_DIM))
discovery_module = ConceptRelationDiscovery(NUM_CONCEPTS, EMBED_DIM, NUM_HEADS)

optimizer = optim.Adam(
    list(discovery_module.parameters()) + [concept_embeddings], 
    lr=0.02
)

print(f"Start Training...")
print("-" * 30)

for epoch in range(101):
    optimizer.zero_grad()
    
    adj_matrices = discovery_module(concept_embeddings)
    
    # --- Head 0: 前置关系 (DAG + Entropy) ---
    adj_prec = adj_matrices[0]
    l_entropy_0 = loss_entropy(adj_prec)
    l_dag = loss_dag(adj_prec)
    
    # --- Head 1: 相似关系 (Symmetry + Entropy) ---
    adj_sim = adj_matrices[1]
    l_entropy_1 = loss_entropy(adj_sim)
    l_sym = loss_symmetry(adj_sim)
    
    # 正则化 Embedding，防止数值过大
    l_reg = torch.norm(concept_embeddings) * 0.01
    
    # 总 Loss
    total_loss = (1.0 * l_entropy_0) + (1.0 * l_dag) + \
                 (1.0 * l_entropy_1) + (20.0 * l_sym) + l_reg # 加大 Sym 权重
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Total: {total_loss.item():.4f} | "
              f"DAG: {l_dag.item():.4f} | Sym: {l_sym.item():.4f} | "
              f"Ent(1): {l_entropy_1.item():.2f}")

print("-" * 30)
print("=== 步骤 3: 检查结果 ===")

with torch.no_grad():
    final_adjs = discovery_module(concept_embeddings)
    
    print("\n[Head 0: 前置关系 (应为 DAG)]")
    adj_0 = final_adjs[0]
    for i in range(5):
        max_val, max_idx = torch.max(adj_0[i], dim=0)
        print(f"概念 {i} -> 主要前置: 概念 {max_idx.item()} (概率 {max_val:.2f})")

    print("\n[Head 1: 相似关系 (应为对称且非对角)]")
    adj_1 = final_adjs[1]
    # 打印前几对互为最大相似的组合
    for i in range(5):
        max_val, j = torch.max(adj_1[i], dim=0)
        # 检查 j 的最大相似是否也是 i
        max_val_back, i_back = torch.max(adj_1[j], dim=0)
        
        is_sym = (i_back.item() == i)
        rel_str = "<==>" if is_sym else "-->"
        print(f"概念 {i} {rel_str} 概念 {j.item()} (概率 {max_val:.2f} vs {max_val_back:.2f}) {'[对称成功]' if is_sym else ''}")

print("-" * 30)