import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# 设置随机种子
torch.manual_seed(42)

# ==========================================
# 模块一：无监督概念关系发现 (保持之前验证成功的版本)
# ==========================================
class ConceptRelationDiscovery(nn.Module):
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
        
        # Attention Scores: [H, N, N]
        # score[h, i, j] 表示 i 和 j 的关系强度
        # 我们定义语义: A[i, j] 表示从 i 到 j 的权重 (i -> j)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 屏蔽对角线
        diag_mask = torch.eye(N, device=concept_emb.device).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(self.num_heads, -1, -1)
        scores.masked_fill_(diag_mask, -1e9)
        
        # 生成概率图
        # Softmax(dim=-1) 使得每一行归一化。
        # A[i, :] sum=1. 这意味着 A[i, j] 是 "i 分配给 j 的权重"。
        # 这符合 "i -> j" 的流向语义。
        A = torch.softmax(scores / self.temperature, dim=-1)
        return A

# ==========================================
# 模块二：学生能力解耦与传播 (新增)
# ==========================================
class StudentDisentanglement(nn.Module):
    def __init__(self, dim_student, num_concepts, dim_skill=4):
        super().__init__()
        self.dim_student = dim_student
        self.num_concepts = num_concepts
        self.dim_skill = dim_skill # 技巧向量维度通常较小

        # 1. 知识状态投影器: Student Emb -> Initial Mastery (N维，对应N个知识点)
        self.know_proj = nn.Sequential(
            nn.Linear(dim_student, dim_student),
            nn.ReLU(),
            nn.Linear(dim_student, num_concepts) # 输出维度 = 知识点数量
        )

        # 2. 应试技巧投影器: Student Emb -> Skill Vector
        self.skill_proj = nn.Sequential(
            nn.Linear(dim_student, dim_skill),
            nn.Tanh() # 技巧通常在一定范围内波动
        )
        
        # 3. 技巧转Guess/Slip: Skill Vector -> Guess & Slip scalars
        self.guess_slip_head = nn.Linear(dim_skill, 2) 

    def forward(self, student_emb, adj_matrix):
        """
        输入:
            student_emb: [Batch, dim_student]
            adj_matrix:  [N, N] (从模块一生成的图)
        """
        # --- 步骤 1: 解耦 ---
        # 初始知识状态 [Batch, N]
        z_knowledge_init = torch.sigmoid(self.know_proj(student_emb))
        
        # 技巧向量 [Batch, dim_skill]
        z_skill = self.skill_proj(student_emb)
        
        # --- 步骤 2: 图传播 (GNN) ---
        # 逻辑：利用学到的图 adj，将前置知识的掌握度传播给后续节点
        # 语义检查: 
        #   z_knowledge_init: [B, N]
        #   adj_matrix: [N, N], 其中 adj[i, j] 是 i -> j 的权重
        #   matmul(z, adj) -> [B, N]
        #   Result[b, j] = sum_i (z[b, i] * adj[i, j])
        #   即: 概念 j 的新状态 = 所有前置概念 i 的状态 * (i对j的影响权重)
        #   这是正确的 "前置 -> 后继" 传播逻辑。
        #   !!! 绝对不能使用转置 (adj.t())，否则就变成 "后继 -> 前置" 了 !!!
        propagation = torch.matmul(z_knowledge_init, adj_matrix)
        
        # 简单的残差连接
        z_knowledge_final = z_knowledge_init + propagation
        
        # --- 步骤 3: 计算猜测与失误 ---
        gs_params = torch.sigmoid(self.guess_slip_head(z_skill)) * 0.2 
        guess = gs_params[:, 0:1]
        slip = gs_params[:, 1:2]
        
        return z_knowledge_init, z_knowledge_final, z_skill, guess, slip

# ==========================================
# 模块三：简易预测与 Loss (用于闭环测试)
# ==========================================
class SimpleCDM(nn.Module):
    def __init__(self, num_concepts, dim_concept, num_students, dim_student):
        super().__init__()
        # Embedding 层
        self.concept_emb = nn.Embedding(num_concepts, dim_concept)
        self.student_emb = nn.Embedding(num_students, dim_student)
        
        # 模块一：图发现
        self.graph_module = ConceptRelationDiscovery(num_concepts, dim_concept, num_heads=2)
        
        # 模块二：解耦与传播
        self.student_module = StudentDisentanglement(dim_student, num_concepts)

    def forward(self, student_ids, concept_ids):
        """
        student_ids: [Batch]
        concept_ids: [Batch] (为了简化，假设每个样本是一人做一题，该题只包含一个知识点)
        """
        # 1. 获取 Embedding
        e_c = self.concept_emb.weight # [N, d_c]
        e_s = self.student_emb(student_ids) # [Batch, d_s]
        
        # 2. 模块一：生成图
        # 我们只取 Head 0 (前置关系) 用于知识传播
        adj_matrices = self.graph_module(e_c)
        adj_prec = adj_matrices[0] # [N, N]
        adj_sim = adj_matrices[1]  # [N, N] 用于 Loss 约束
        
        # 3. 模块二：解耦与传播
        # 注意：这里我们传递原始的 adj_prec，不做任何转置
        z_k_init, z_k_final, z_skill, guess, slip = self.student_module(e_s, adj_prec)
        
        # 4. 模块三：预测
        batch_mastery = z_k_final.gather(1, concept_ids.unsqueeze(1)) # [Batch, 1]
        
        pred = batch_mastery + guess - slip
        pred = torch.clamp(pred, 0, 1)
        
        return pred, adj_matrices, z_k_init, z_skill

# ==========================================
# 损失函数定义
# ==========================================
def loss_orthogonality(z_know, z_skill):
    if z_know.shape[1] > z_skill.shape[1]:
        z_know_reduced = z_know[:, :z_skill.shape[1]]
    else:
        z_know_reduced = z_know
        
    vn_know = F.normalize(z_know_reduced, dim=1)
    vn_skill = F.normalize(z_skill, dim=1)
    
    cosine = torch.sum(vn_know * vn_skill, dim=1)
    return torch.mean(cosine ** 2)

def loss_entropy(A):
    return -torch.sum(A * torch.log(A + 1e-8), dim=-1).mean()

def loss_dag(A):
    expm_A = torch.matrix_exp(A)
    return torch.trace(expm_A) - A.shape[0]

def loss_symmetry(A):
    return torch.mean((A - A.transpose(-1, -2)) ** 2)

# ==========================================
# 模拟训练流程
# ==========================================
print("=== 初始化集成测试 (Modules 1 + 2) ===")
NUM_STU = 50
NUM_CONCEPTS = 10
DIM_C = 16
DIM_S = 16

model = SimpleCDM(NUM_CONCEPTS, DIM_C, NUM_STU, DIM_S)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss() # 二分类交叉熵

# 生成假数据：随机生成一些做题记录
# 假设有 500 条做题记录
BATCH_SIZE = 500
dummy_s_ids = torch.randint(0, NUM_STU, (BATCH_SIZE,))
dummy_c_ids = torch.randint(0, NUM_CONCEPTS, (BATCH_SIZE,))
# 随机生成标签 (0或1)
dummy_labels = torch.randint(0, 2, (BATCH_SIZE,)).float().unsqueeze(1)

print(f"Data: {BATCH_SIZE} interactions, {NUM_STU} students, {NUM_CONCEPTS} concepts")
print("-" * 30)

for epoch in range(101):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    preds, adj_matrices, z_k_init, z_skill = model(dummy_s_ids, dummy_c_ids)
    
    # 1. 任务 Loss (BCE)
    l_task = criterion(preds, dummy_labels)
    
    # 2. 图结构 Loss (Module 1)
    adj_dag = adj_matrices[0]
    adj_sym = adj_matrices[1]
    l_dag = loss_dag(adj_dag)
    l_ent = loss_entropy(adj_dag) + loss_entropy(adj_sym)
    l_sym = loss_symmetry(adj_sym)
    
    # 3. 解耦 Loss (Module 2)
    l_orth = loss_orthogonality(z_k_init, z_skill)
    
    # 总 Loss
    total_loss = l_task + 1.0*l_dag + 10.0*l_sym + 1.0*l_ent + 1.0*l_orth
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Total: {total_loss.item():.4f} | Task: {l_task.item():.4f} | "
              f"DAG: {l_dag.item():.4f} | Orth: {l_orth.item():.4f}")

print("-" * 30)
print("=== 验证解耦效果 ===")
with torch.no_grad():
    # 随机取一个学生看看
    s_idx = torch.tensor([0])
    # 为了拿到 z_skill 和 z_know，我们得 hack 一下或者单独调用 student_module
    # 这里简单起见，利用 model.student_emb
    e_s = model.student_emb(s_idx)
    adj = model.graph_module(model.concept_emb.weight)[0]
    z_k, z_k_final, z_s, g, s = model.student_module(e_s, adj)
    
    print(f"学生 0 的技巧向量 (Skill): {z_s.numpy()[0]}")
    print(f"学生 0 的 Guess 参数: {g.item():.4f}, Slip 参数: {s.item():.4f}")
    print(f"知识向量 (前5维): {z_k.numpy()[0][:5]}")
    
    # 检查图结构是否依然存在 (受 Task Loss 驱动)
    print("\n[最终学习到的前置关系图 (局部)]")
    for i in range(5):
        max_val, max_idx = torch.max(adj[i], dim=0)
        print(f"Concept {i} -> Concept {max_idx.item()} (prob {max_val:.2f})")

print("-" * 30)