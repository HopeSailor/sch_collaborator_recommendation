from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn


def sparse_adjacency(
    node_count: int, hyperedges: list[list[int]], device: torch.device
) -> torch.Tensor:
    weights: dict[tuple[int, int], float] = defaultdict(float)
    for hyperedge in hyperedges:
        nodes = sorted(set(hyperedge))
        if len(nodes) < 2:
            continue
        if len(nodes) == 2:
            pairs = [(nodes[0], nodes[1])]
        else:
            left, right = nodes[0], nodes[-1]
            pairs = [(left, right)]
            pairs.extend(
                pair
                for node in nodes[1:-1]
                for pair in ((node, left), (node, right))
            )
        weight = 1.0 / len(pairs)
        for left, right in pairs:
            weights[left, right] += weight
            weights[right, left] += weight

    rows = [left for left, _ in weights] + list(range(node_count))
    cols = [right for _, right in weights] + list(range(node_count))
    values = list(weights.values()) + [1.0] * node_count
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    adjacency = torch.sparse_coo_tensor(
        indices, values_tensor, (node_count, node_count), device=device
    ).coalesce()

    degree = torch.zeros(node_count, dtype=torch.float32, device=device)
    degree.scatter_add_(0, adjacency.indices()[0], adjacency.values())
    inverse_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    row, col = adjacency.indices()
    normalized = adjacency.values() * inverse_sqrt[row] * inverse_sqrt[col]
    return torch.sparse_coo_tensor(
        adjacency.indices(), normalized, adjacency.shape, device=device
    ).coalesce()


class HyperGCNEncoder(nn.Module):
    def __init__(self, node_count: int, dimension: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.input_projection = nn.Parameter(torch.empty(node_count, dimension))
        self.output_projection = nn.Linear(dimension, dimension, bias=False)
        nn.init.xavier_uniform_(self.input_projection)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(self) -> torch.Tensor:
        hidden = torch.relu(torch.sparse.mm(self.adjacency, self.input_projection))
        hidden = torch.sparse.mm(self.adjacency, hidden)
        return torch.relu(self.output_projection(hidden))
