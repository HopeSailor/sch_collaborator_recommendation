from __future__ import annotations

import torch
from torch import nn

from .hypergcn import HyperGCNEncoder
from .scoring import TranslationalScorer


class SCHModel(nn.Module):
    def __init__(self, node_count: int, dimension: int, adjacency: torch.Tensor):
        super().__init__()
        self.encoder = HyperGCNEncoder(node_count, dimension, adjacency)
        self.scorer = TranslationalScorer(dimension)

    def encode(self) -> torch.Tensor:
        return self.encoder()

    def score(
        self, embeddings: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.scorer(embeddings, source, target)
