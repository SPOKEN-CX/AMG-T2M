"""Length regression on frozen, word-aligned CLIP features and dependency graphs."""
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from .LengthEst import GraphResBlock, StylizationBlock, NormalFFN


class CachedMotionLengthEstimator(nn.Module):
    def __init__(self, latent_dim=512, layers=4, heads=4, dropout=0.3):
        super().__init__()
        self.graph_res_blocks = nn.ModuleList(
            GraphResBlock(latent_dim, latent_dim, heads) for _ in range(layers))
        self.stylization_blocks = nn.ModuleList(
            StylizationBlock(latent_dim, latent_dim, dropout) for _ in range(layers))
        self.cap_len_emb = nn.Linear(1, latent_dim)
        nn.init.xavier_normal_(self.cap_len_emb.weight)
        nn.init.zeros_(self.cap_len_emb.bias)
        self.normffn = NormalFFN(latent_dim, 1, dropout)

    def forward(self, graph):
        # Only text-derived word counts are supplied; motion length is never an input.
        graph = graph.clone()
        emb = self.cap_len_emb(graph.word_count.reshape(-1, 1).float())
        for block, style in zip(self.graph_res_blocks, self.stylization_blocks):
            graph.x = block(graph)
            graph.x = style(graph.x, emb[graph.batch])
        # Pool real nodes only. Padding must not change a sentence's prediction.
        per_node = self.normffn.ResLinear(graph.x) + self.normffn.output(graph.x)
        return torch.sigmoid(global_mean_pool(per_node, graph.batch)).squeeze(-1)
