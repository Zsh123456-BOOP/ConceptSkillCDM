import torch
import torch.nn as nn


class ConceptGNN(nn.Module):
    def __init__(self, num_layers: int, aggr: str = "gcn"):
        super().__init__()
        self.num_layers = num_layers
        self.aggr = aggr

    def forward(self, A_norm: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_norm: [num_concepts, num_concepts] normalized adjacency
            S: [batch_size, num_concepts] initial knowledge states
        Returns:
            S_prop: [batch_size, num_concepts]
        """
        S_l = S
        for _ in range(self.num_layers):
            if self.aggr == "gcn":
                S_l = torch.matmul(S_l, A_norm)
            else:
                S_l = torch.matmul(S_l, A_norm)
        return S_l


__all__ = ["ConceptGNN"]
