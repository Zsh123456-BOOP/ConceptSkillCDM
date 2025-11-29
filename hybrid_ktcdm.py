import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqGraphStudentEncoder(nn.Module):
    """
    模块二升级版：
    - 输入一条学生的作答序列 (concept_id 序列 + 正误序列)
    - 用 GRU 做序列 KT，得到每个时间步的隐藏状态 h_t
    - 把 h_t 映射到：
        - 知识状态 z_t (dim_cpt)
        - 技巧状态 s_t (skill_dim 很小)
    - 知识状态再过“概念图”传播（前置 + 相似），得到 graph-aware 知识状态 z_t_graph
    - 返回当前时间步 (或所有时间步) 的 (z_t_graph, s_t)，供 HybridKTCDM 做预测

    约定输入形状：
    - stu_ids: [B]                （可以只用来查一个静态 embedding, 也可以不用）
    - seq_cpt_ids: [B, T]         概念ID序列
    - seq_correct: [B, T]         正误标签序列 (0/1)
    - seq_mask: [B, T]            mask，1=有效, 0=padding
    - A_dag: [C, C]               Head0 的前置图邻接（已 softmax 正则化）
    - A_sym: [C, C]               Head1 的相似图邻接（已 softmax 正则化）

    返回：
    - z_t_graph: [B, T, dim_cpt]  图传播后的知识状态
    - s_t: [B, T, skill_dim]      技巧状态
    """

    def __init__(
        self,
        num_students: int,
        num_concepts: int,
        dim_stu: int,
        dim_cpt: int,
        rnn_hidden_dim: int = 128,
        skill_dim: int = 4,
        use_student_static: bool = True,
    ):
        super().__init__()
        self.num_students = num_students
        self.num_concepts = num_concepts
        self.dim_stu = dim_stu
        self.dim_cpt = dim_cpt
        self.rnn_hidden_dim = rnn_hidden_dim
        self.skill_dim = skill_dim
        self.use_student_static = use_student_static

        # 学生静态 embedding（可选，用作序列初始状态）
        if use_student_static:
            self.stu_emb = nn.Embedding(num_students, dim_stu)
        else:
            self.stu_emb = None

        # 概念 embedding（输入到 RNN 的 one-hot 替代）
        self.cpt_emb = nn.Embedding(num_concepts, dim_cpt)

        # 正误 embedding：0/1 → 各一个向量
        self.ans_emb = nn.Embedding(2, dim_cpt)

        # RNN：简单起见用单层 GRU，你可以改成 LSTM 或 Transformer
        # 输入是 [concept_emb || answer_emb]，维度 = 2 * dim_cpt
        self.rnn = nn.GRU(
            input_size=2 * dim_cpt,
            hidden_size=rnn_hidden_dim,
            batch_first=True,
        )

        # 把 RNN hidden 投影到“概念空间” → 知识状态
        self.proj_know = nn.Linear(rnn_hidden_dim, dim_cpt)

        # 把 RNN hidden 投影到“技巧空间” → 技巧状态
        self.proj_skill = nn.Linear(rnn_hidden_dim, skill_dim)

        # 图融合的权重，可以学： alpha * A_dag + (1-alpha) * A_sym
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, stu_ids, seq_cpt_ids, seq_correct, seq_mask, A_dag, A_sym):
        """
        seq_cpt_ids: [B, T]
        seq_correct: [B, T]  (0/1)
        seq_mask: [B, T]     (0/1)
        A_dag, A_sym: [C, C]
        """
        B, T = seq_cpt_ids.shape
        device = seq_cpt_ids.device
        C = self.num_concepts

        # 概念嵌入 + 正误嵌入
        cpt_embed = self.cpt_emb(seq_cpt_ids)        # [B, T, dim_cpt]
        ans_embed = self.ans_emb(seq_correct.long()) # [B, T, dim_cpt]

        rnn_input = torch.cat([cpt_embed, ans_embed], dim=-1)  # [B, T, 2*dim_cpt]

        # mask → pack_padded_sequence 可选，这里简单用 mask 在 loss 里处理
        # 如果你想更严谨，可以算出每个样本真实长度，用 pack 进 RNN
        if self.use_student_static and self.stu_emb is not None:
            # 用学生静态 embedding 映射到 RNN 初始 hidden
            stu_static = self.stu_emb(stu_ids)  # [B, dim_stu]
            h0 = torch.tanh(
                nn.functional.linear(
                    stu_static,  # [B, dim_stu]
                    torch.empty(self.rnn_hidden_dim, self.dim_stu, device=device).normal_(std=0.02),
                )
            ).unsqueeze(0)  # [1, B, H]
        else:
            h0 = torch.zeros(1, B, self.rnn_hidden_dim, device=device)

        # 直接跑 RNN
        rnn_out, _ = self.rnn(rnn_input, h0)  # [B, T, H]

        # 投影到知识 / 技巧空间
        z_t = self.proj_know(rnn_out)   # [B, T, dim_cpt]
        s_t = self.proj_skill(rnn_out)  # [B, T, skill_dim]

        # 图先验：对每个时间步的 z_t 做 A @ z_t^T
        # 先组合两个头
        alpha = torch.sigmoid(self.alpha)  # 限制在 (0,1)
        A_mix = alpha * A_dag + (1.0 - alpha) * A_sym  # [C, C]

        # 为了方便，可以先展平 batch/time 再做矩阵乘
        z_flat = z_t.reshape(B * T, self.dim_cpt)  # [B*T, C']
        # 假设 dim_cpt == num_concepts 时，可以直接 A_mix @ z
        # 如果 dim_cpt != C，你可以改成 A_mix 作用在“概念维度”上（需要做一个映射）
        if self.dim_cpt == C:
            z_flat_graph = torch.matmul(A_mix, z_flat.transpose(0, 1)).transpose(0, 1)  # [B*T, C]
        else:
            # 一般情况：先从“概念空间”映射回一个 C 维向量，再做图传播
            # 这里给一个简单实现，你可以按自己情况改：
            #   z_flat_c = W_c @ z_flat^T → [C, B*T]
            W_c = torch.empty(C, self.dim_cpt, device=device)
            nn.init.xavier_uniform_(W_c)
            z_flat_c = torch.matmul(W_c, z_flat.transpose(0, 1))        # [C, B*T]
            z_flat_c_graph = torch.matmul(A_mix, z_flat_c)              # [C, B*T]
            # 再映射回 dim_cpt
            W_back = torch.empty(self.dim_cpt, C, device=device)
            nn.init.xavier_uniform_(W_back)
            z_flat_graph = torch.matmul(W_back, z_flat_c_graph).transpose(0, 1)  # [B*T, dim_cpt]

        z_t_graph = z_flat_graph.reshape(B, T, self.dim_cpt)  # [B, T, dim_cpt]

        # mask 位置归零，防止对 loss 产生干扰
        mask = seq_mask.unsqueeze(-1)  # [B, T, 1]
        z_t_graph = z_t_graph * mask
        s_t = s_t * mask

        return z_t_graph, s_t

    @staticmethod
    def decouple_loss(z_t_graph, s_t, mask):
        """
        知识状态和技巧状态的解耦约束：
        在 mask 有效的位置上，让两者尽量正交（cosine 接近 0）

        z_t_graph: [B, T, Dk]
        s_t:       [B, T, Ds]
        mask:      [B, T]
        """
        # 归一化
        z_norm = F.normalize(z_t_graph, dim=-1)  # [B, T, Dk]
        s_norm = F.normalize(s_t, dim=-1)        # [B, T, Ds]

        # 为了算 cos 相似度，把 Ds 对齐到 Dk 或反之，这里简单做一个线性映射
        if z_norm.size(-1) != s_norm.size(-1):
            # 映射 s_norm → Dk 维
            B, T, _ = s_norm.shape
            Dk = z_norm.size(-1)
            W = torch.empty(Dk, s_norm.size(-1), device=s_norm.device)
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            s_mapped = torch.matmul(
                s_norm.reshape(-1, s_norm.size(-1)),  # [B*T, Ds]
                W.t()                                 # [Ds, Dk]
            ).reshape(B, T, Dk)
            s_norm = F.normalize(s_mapped, dim=-1)

        # 点积近似 cos
        cos = (z_norm * s_norm).sum(dim=-1)  # [B, T]
        loss = (cos ** 2) * mask  # mask 无效位置
        # 避免除以 0
        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom
