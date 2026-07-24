"""
Virtual node onsite: K learnable virtual nodes attend to all atoms,
producing K "aspect" vectors per graph. Bound vs unbound aspects are
compared via a delta MLP, yielding a graph-level onsite signal.

This adds a third scale of comparison alongside node onsite (per-atom)
and edge onsite (per-interaction).
"""

import math

import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_sum


class VirtualNodeOnsite(nn.Module):
    """K virtual nodes attend to graph, compare bound vs unbound."""

    def __init__(self, node_scalar_dim: int, num_virtual_nodes: int = 4,
                 virtual_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.num_virtual_nodes = num_virtual_nodes
        self.virtual_dim = virtual_dim

        # Learnable queries: (K, d) — shared between bound/unbound
        self.queries = nn.Parameter(torch.randn(num_virtual_nodes, virtual_dim) * 0.02)
        self.key_proj = nn.Linear(node_scalar_dim, virtual_dim)
        self.val_proj = nn.Linear(node_scalar_dim, virtual_dim)

        # Attention dropout (applied after softmax)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Comparison MLP: delta -> transformed delta
        if dropout > 0:
            self.compare = nn.Sequential(
                nn.Linear(virtual_dim, virtual_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(virtual_dim, virtual_dim),
            )
        else:
            self.compare = nn.Sequential(
                nn.Linear(virtual_dim, virtual_dim),
                nn.SiLU(),
                nn.Linear(virtual_dim, virtual_dim),
            )

    @property
    def output_dim(self) -> int:
        return self.num_virtual_nodes * self.virtual_dim

    def forward(self, h_bound_scalars: torch.Tensor, h_unbound_scalars: torch.Tensor,
                batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        v_bound = self._attend(h_bound_scalars, batch, batch_size)      # (B, K, d)
        v_unbound = self._attend(h_unbound_scalars, batch, batch_size)  # (B, K, d)
        delta = v_bound - v_unbound                                      # (B, K, d)
        compared = self.compare(delta)                                   # (B, K, d)
        return compared.view(batch_size, -1)                             # (B, K*d)

    def _attend(self, node_scalars: torch.Tensor, batch: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        keys = self.key_proj(node_scalars)                              # (N, d)
        values = self.val_proj(node_scalars)                            # (N, d)
        scale = math.sqrt(self.virtual_dim)
        scores = keys @ self.queries.T / scale                          # (N, K)
        attn = scatter_softmax(scores, batch, dim=0)                    # (N, K)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)
        weighted = values.unsqueeze(1) * attn.unsqueeze(2)              # (N, K, d)
        return scatter_sum(weighted, batch, dim=0, dim_size=batch_size)  # (B, K, d)
