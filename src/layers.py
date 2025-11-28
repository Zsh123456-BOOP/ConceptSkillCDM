import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LightGCN(nn.Module):
    def __init__(self, dim, layers, dropout=0.2):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.convs = nn.ModuleList([GCNConv(dim, dim, cached=False) for _ in range(layers)])

    def forward(self, x, edge_index):
        x_all = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            # 斯巴达数据去ReLU，提升线性传播效率
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_all.append(x)
        return torch.stack(x_all, dim=0).mean(dim=0)