import torch
import torch.nn as nn
from typing import Tuple, Dict


def hsic_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Hilbert-Schmidt Independence Criterion (HSIC)
    用于约束两个随机向量的统计独立性。
    
    x: (B, dx)
    y: (B, dy)
    """
    B = x.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=x.device)

    def gaussian_kernel(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(u, v, p=2).pow(2)
        return torch.exp(-dist / (2 * sigma ** 2))

    K = gaussian_kernel(x, x)  # (B, B)
    L = gaussian_kernel(y, y)  # (B, B)

    H = torch.eye(B, device=x.device) - torch.ones((B, B), device=x.device) / B

    KH = torch.matmul(K, H)
    LH = torch.matmul(L, H)
    hsic_val = torch.trace(torch.matmul(KH, LH)) / ((B - 1) ** 2 + 1e-8)

    return hsic_val


def dag_constraint_loss(A: torch.Tensor) -> torch.Tensor:
    """
    NOTEARS 风格的 DAG 约束：
    h(A) = tr(exp(A)) - d
    A 要求非负（比如 ReLU 输出）。
    """
    d = A.size(0)
    expm_A = torch.matrix_exp(A)
    h_A = torch.trace(expm_A) - d
    return h_A


class CDMLoss(nn.Module):
    """
    综合损失：
    - BCE 预测损失
    - DAG 约束 (Structure)
    - 稀疏约束 (L1)
    - 解耦约束 (HSIC)
    """
    def __init__(
        self,
        lambda_dag: float = 0.05,
        lambda_sparse: float = 0.001,
        lambda_hsic: float = 0.1,
    ):
        super().__init__()
        self.lambda_dag = lambda_dag
        self.lambda_sparse = lambda_sparse
        self.lambda_hsic = lambda_hsic
        self.bce = nn.BCELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        adj_dag: torch.Tensor,
        h_know: torch.Tensor,
        z_skill: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Input:
            pred:     (B,) 预测答对概率
            target:   (B,) 真实标签（0/1）
            adj_dag:  (K, K) 前置关系图
            h_know:   (B, K) 知识状态（传播后）
            z_skill:  (B, D_skill) 技巧向量
        """
        # 1. 预测损失
        loss_pred = self.bce(pred, target)

        # 2. 图结构正则
        loss_dag = dag_constraint_loss(adj_dag)
        loss_sparse = torch.norm(adj_dag, p=1)

        # 3. 解耦独立性约束
        loss_disentangle = hsic_loss(h_know, z_skill)

        total_loss = (
            loss_pred
            + self.lambda_dag * loss_dag
            + self.lambda_sparse * loss_sparse
            + self.lambda_hsic * loss_disentangle
        )

        # 🔧 这里的 key 命名适配当前 Trainer（pred / dag / sparse / hsic）
        stats = {
            "pred": loss_pred.item(),
            "dag": loss_dag.item(),
            "sparse": loss_sparse.item(),
            "hsic": loss_disentangle.item(),
        }

        return total_loss, stats
