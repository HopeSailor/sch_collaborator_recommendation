from __future__ import annotations

import torch
from torch import nn


class TranslationalScorer(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.coauthor_relation = nn.Parameter(torch.zeros(dimension))

    def forward(
        self, embeddings: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        translated = embeddings[source] + self.coauthor_relation
        return -torch.linalg.vector_norm(translated - embeddings[target], dim=1)
