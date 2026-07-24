"""
Readout fusion modules: combine multiple pooling streams before the prediction head.

All modules have the same interface:
    forward(streams: List[Tensor]) -> Tensor
where each stream is (B, D_i) and the output is (B, fusion_dim).

Available:
    GatedReadoutFusion      — each stream gates the sum of all others
    FiLMReadoutFusion       — first stream is "main", rest are "modulators" (FiLM)
    SelfAttentionReadoutFusion — streams as tokens, single-layer self-attention
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedReadoutFusion(nn.Module):
    """Each stream produces a sigmoid gate for the sum of all other streams.

    output[i] = sigma(W_gate[i] @ stream[i]) * sum_{j!=i} W_proj[j] @ stream[j]
    Final output = concat(output[0], ..., output[N-1]).
    """

    def __init__(self, stream_dims: List[int], fusion_dim: int):
        super().__init__()
        n = len(stream_dims)
        self.n_streams = n
        self.projs = nn.ModuleList([nn.Linear(d, fusion_dim) for d in stream_dims])
        self.gates = nn.ModuleList([nn.Linear(d, fusion_dim) for d in stream_dims])
        self.output_dim = n * fusion_dim

    def forward(self, streams: List[torch.Tensor]) -> torch.Tensor:
        projected = [self.projs[i](streams[i]) for i in range(self.n_streams)]
        parts = []
        for i in range(self.n_streams):
            gate = torch.sigmoid(self.gates[i](streams[i]))
            others = sum(projected[j] for j in range(self.n_streams) if j != i)
            parts.append(gate * others)
        return torch.cat(parts, dim=-1)


class FiLMReadoutFusion(nn.Module):
    """Feature-wise Linear Modulation.

    First stream is "main", all remaining streams are concatenated as "modulator".
    output = gamma(modulator) * main + beta(modulator)
    """

    def __init__(self, stream_dims: List[int], fusion_dim: int):
        super().__init__()
        assert len(stream_dims) >= 2, "FiLM requires at least 2 streams"
        main_dim = stream_dims[0]
        mod_dim = sum(stream_dims[1:])
        self.main_proj = nn.Linear(main_dim, fusion_dim)
        self.gamma = nn.Linear(mod_dim, fusion_dim)
        self.beta = nn.Linear(mod_dim, fusion_dim)
        self.output_dim = fusion_dim

    def forward(self, streams: List[torch.Tensor]) -> torch.Tensor:
        main = self.main_proj(streams[0])
        modulator = torch.cat(streams[1:], dim=-1)
        return self.gamma(modulator) * main + self.beta(modulator)


class SelfAttentionReadoutFusion(nn.Module):
    """Streams as tokens, single-layer multi-head self-attention, then mean pool.

    Each stream is projected to a shared dim, forming N tokens.
    One layer of multi-head self-attention over these tokens.
    Output = mean of attended tokens.
    """

    def __init__(self, stream_dims: List[int], fusion_dim: int, num_heads: int = 4):
        super().__init__()
        self.n_streams = len(stream_dims)
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        assert fusion_dim % num_heads == 0

        self.input_projs = nn.ModuleList(
            [nn.Linear(d, fusion_dim) for d in stream_dims]
        )
        self.q_proj = nn.Linear(fusion_dim, fusion_dim)
        self.k_proj = nn.Linear(fusion_dim, fusion_dim)
        self.v_proj = nn.Linear(fusion_dim, fusion_dim)
        self.out_proj = nn.Linear(fusion_dim, fusion_dim)
        self.output_dim = fusion_dim

    def forward(self, streams: List[torch.Tensor]) -> torch.Tensor:
        B = streams[0].size(0)
        N = self.n_streams
        d_head = self.fusion_dim // self.num_heads

        # Project each stream to shared dim → (B, N, D)
        tokens = torch.stack(
            [self.input_projs[i](streams[i]) for i in range(N)], dim=1
        )

        Q = self.q_proj(tokens).view(B, N, self.num_heads, d_head).transpose(1, 2)
        K = self.k_proj(tokens).view(B, N, self.num_heads, d_head).transpose(1, 2)
        V = self.v_proj(tokens).view(B, N, self.num_heads, d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(d_head)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, self.fusion_dim)

        out = self.out_proj(out)
        return out.mean(dim=1)  # (B, fusion_dim)


_FUSION_REGISTRY = {
    "gated": GatedReadoutFusion,
    "film": FiLMReadoutFusion,
    "self_attn": SelfAttentionReadoutFusion,
}


def build_readout_fusion(mode: str, stream_dims: List[int],
                         fusion_dim: int) -> nn.Module:
    """Factory for readout fusion modules."""
    cls = _FUSION_REGISTRY.get(mode)
    if cls is None:
        raise ValueError(
            f"Unknown readout fusion mode '{mode}'. "
            f"Available: {list(_FUSION_REGISTRY)}"
        )
    return cls(stream_dims, fusion_dim)
