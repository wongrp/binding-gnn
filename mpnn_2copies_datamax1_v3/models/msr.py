"""Multi-Shell Radial readout (MSR).

Bilinear bound-vs-unbound readout using K parallel learned signed radial
filters, with no spherical-harmonic angular structure.

Motivated by the layer_behavior finding (paper2/layer_behavior.tex) that
the trained MPNN+PR global-SH readout uses the SH machinery as a rank-1
separable shell counter: the angular pair-coherence factor
P_l(cos gamma_jk) is empirically unused. MSR drops the SH machinery and
exposes K independent signed radial channels directly.

Per-filter forward (matched to GlobalSHPooling conventions):
    1. Ligand centroid per graph.
    2. Pocket-atom distances d_j to the centroid.
    3. RBF expansion of d_j -> radial MLP -> R_k(d_j) for k = 1..K.
    4. Per-atom channel weights w_b(h_j^b), w_ub(h_j^ub).
    5. Filtered feature sums per (graph, filter, channel):
         f_b[B, k, c]  = sum_j R_k(d_j) * w_b[j, c]
         f_ub[B, k, c] = sum_j R_k(d_j) * w_ub[j, c]
    6. Channel-wise diagonal bilinear: P[B, k, c] = f_b[B, k, c] * f_ub[B, k, c].
    7. Flatten and project -> out_dim.

Equivalent to global SH at l=0..L_max (after the rank-1 collapse) when
K = L_max + 1 and the angular factor is replaced by 1. K can be larger
than L_max + 1 to give more shell-channel capacity at fixed parameter
budget.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum
from e3nn.math import soft_one_hot_linspace


class MultiShellRadial(nn.Module):
    """Bilinear shell-counter readout: K parallel signed radial filters.

    Parameters
    ----------
    n_scalar_in : int
        Number of input scalar channels per atom.
    n_channels : int
        Channels per filter head (default 16, matching GlobalSHPooling).
    num_filters : int
        Number of parallel signed radial filters (K).
    out_dim : int
        Output dim after final projection.
    separate_proj : bool
        Use separate weight projections for bound and unbound copies.
    num_rbf : int
        Number of RBF basis functions for the radial parameterization.
    r_max : float
        Maximum distance for radial cutoff (Å).
    """

    def __init__(
        self,
        n_scalar_in: int,
        n_channels: int = 16,
        num_filters: int = 8,
        out_dim: int = 64,
        separate_proj: bool = True,
        num_rbf: int = 20,
        r_max: float = 20.0,
    ):
        super().__init__()
        self.n_scalar_in = n_scalar_in
        self.n_channels = n_channels
        self.num_filters = num_filters
        self.num_rbf = num_rbf
        self.r_max = r_max

        # Learned signed radial filters: 2-layer MLP on RBF -> K outputs.
        # Same parameterization as GlobalSHPooling.radial.
        self.radial = nn.Sequential(
            nn.Linear(num_rbf, 64),
            nn.SiLU(),
            nn.Linear(64, num_filters, bias=False),
        )
        nn.init.uniform_(self.radial[-1].weight, -1e-3, 1e-3)

        # Per-atom channel projections
        self.weight_proj_b = nn.Linear(n_scalar_in, n_channels)
        if separate_proj:
            self.weight_proj_ub = nn.Linear(n_scalar_in, n_channels)
        else:
            self.weight_proj_ub = None

        # Final projection: K * n_channels -> out_dim
        self.out_proj = nn.Linear(num_filters * n_channels, out_dim)
        self.output_dim = out_dim

    def forward(
        self,
        h_bound: torch.Tensor,
        h_unbound: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        ligand_mask: torch.Tensor,
        batch_size: int,
        **kwargs,  # accept and ignore extra kwargs (e.g., esm_residue_features)
    ) -> torch.Tensor:
        """Compute MSR readout features.

        Args
        ----
        h_bound, h_unbound : [N, n_scalar_in]
        pos                : [N, 3]
        batch              : [N] graph index per atom
        ligand_mask        : [N] True = ligand atom
        batch_size         : int

        Returns
        -------
        readout : [batch_size, out_dim]
        """
        pocket_mask = ~ligand_mask

        # 1. Ligand centroid per graph
        lig_pos = pos[ligand_mask]
        lig_batch = batch[ligand_mask]
        centroid = scatter_mean(lig_pos, lig_batch, dim=0, dim_size=batch_size)

        # 2. Pocket-atom distances to ligand centroid
        pocket_pos = pos[pocket_mask]
        pocket_batch = batch[pocket_mask]
        rel = pocket_pos - centroid[pocket_batch]
        dist = rel.norm(dim=-1)  # [N_pocket]

        # 3. RBF expansion + radial MLP -> K filters per atom
        bin_width = self.r_max / max(1, self.num_rbf - 1)
        rbf = soft_one_hot_linspace(
            dist, 0.0, self.r_max + bin_width,
            self.num_rbf, basis="smooth_finite", cutoff=True,
        )
        rbf = rbf / rbf.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        R_k = self.radial(rbf)  # [N_pocket, K]

        # 4. Per-atom channel weights (operate on pocket atoms only)
        h_b_pocket = h_bound[pocket_mask]
        h_ub_pocket = h_unbound[pocket_mask]
        w_b = self.weight_proj_b(h_b_pocket)  # [N_pocket, C]
        w_ub = (
            self.weight_proj_ub(h_ub_pocket)
            if self.weight_proj_ub is not None
            else self.weight_proj_b(h_ub_pocket)
        )

        # 5. Filtered feature sums per (graph, filter, channel)
        # weighted[j, k, c] = R_k[j] * w[j, c]
        weighted_b = R_k.unsqueeze(-1) * w_b.unsqueeze(1)   # [N_pocket, K, C]
        weighted_ub = R_k.unsqueeze(-1) * w_ub.unsqueeze(1)  # [N_pocket, K, C]
        f_b = scatter_sum(weighted_b, pocket_batch, dim=0, dim_size=batch_size)   # [B, K, C]
        f_ub = scatter_sum(weighted_ub, pocket_batch, dim=0, dim_size=batch_size)  # [B, K, C]

        # 6. Channel-wise diagonal bilinear (no full nc x nc matrix; cheap)
        P = f_b * f_ub  # [B, K, C]

        # 7. Flatten and project
        return self.out_proj(P.flatten(start_dim=1))  # [B, out_dim]
