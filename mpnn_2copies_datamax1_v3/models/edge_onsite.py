"""
Edge-level onsite interaction modules.

Compares bound vs unbound edge features at inter-molecular edges.
Edge features are constructed from endpoint node scalars using a
symmetric encoder (invariant to edge direction).

Architecture mirrors node-level onsite.py:
  EdgeEncoder         (h_i, h_j) → edge hidden vector
  EdgeOnsiteDelta     e_bound - e_unbound → MLP → scalars
  EdgeOnsiteMLP       cat(e_bound, e_unbound) → MLP → scalars
  CompositeEdgeOnsite runs multiple modules, concatenates outputs

Factory:
  build_edge_onsite(node_scalar_dim, edge_hidden_dim, edge_onsite_dim, mode)
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from e3nn.o3 import Irreps, SphericalHarmonics, TensorProduct
from e3nn.math import soft_one_hot_linspace


class EdgeEncoder(nn.Module):
    """Symmetric edge feature encoder from endpoint node scalars.

    Uses f(h_i + h_j, |h_i - h_j|) so that e_ij = e_ji.
    """

    def __init__(self, node_scalar_dim: int, edge_hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_scalar_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        sym = h_i + h_j
        asym = (h_i - h_j).abs()
        return self.mlp(torch.cat([sym, asym], dim=-1))


class EdgeOnsiteDelta(nn.Module):
    """Process e_bound - e_unbound through MLP."""

    def __init__(self, edge_hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_hidden_dim, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.output_dim = out_dim

    def forward(self, e_bound: torch.Tensor, e_unbound: torch.Tensor,
                **kw) -> torch.Tensor:
        return self.mlp(e_bound - e_unbound)


class EdgeOnsiteMLP(nn.Module):
    """Concatenate bound and unbound edge features, process through MLP."""

    def __init__(self, edge_hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * edge_hidden_dim, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.output_dim = out_dim

    def forward(self, e_bound: torch.Tensor, e_unbound: torch.Tensor,
                **kw) -> torch.Tensor:
        return self.mlp(torch.cat([e_bound, e_unbound], dim=-1))


class CompositeEdgeOnsite(nn.Module):
    """Run multiple edge onsite modules and concatenate outputs."""

    def __init__(self, modules: list):
        super().__init__()
        self.mods = nn.ModuleList(modules)
        self.output_dim = sum(m.output_dim for m in modules)

    def forward(self, e_bound: torch.Tensor, e_unbound: torch.Tensor,
                **kwargs) -> torch.Tensor:
        return torch.cat([m(e_bound, e_unbound, **kwargs) for m in self.mods], dim=-1)


class EdgeOnsiteBilinear(nn.Module):
    """Hadamard product: (W_b @ e_b) * (W_u @ e_u) → project."""

    def __init__(self, edge_hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj_b = nn.Linear(edge_hidden_dim, out_dim)
        self.proj_u = nn.Linear(edge_hidden_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.output_dim = out_dim

    def forward(self, e_bound: torch.Tensor, e_unbound: torch.Tensor,
                **kw) -> torch.Tensor:
        return self.out_proj(self.proj_b(e_bound) * self.proj_u(e_unbound))


class EdgeOnsiteReembedCG(nn.Module):
    """Re-embed scalar edge states into l>0 via edge direction SH, CG couple.

    Each edge already has a direction (rel_vec from endpoint positions).
    Projects scalar edge states into irreps via h * R(r) * Y(r_hat),
    then CG-couples bound vs unbound and contracts to scalars.
    """

    def __init__(self, edge_hidden_dim: int, out_dim: int,
                 reembed_lmax: int = 1, cg_mode: str = "uuu",
                 r_max: float = 5.0, num_radial_basis: int = 14):
        super().__init__()
        self.output_dim = out_dim
        self.reembed_lmax = reembed_lmax
        self.r_max = r_max
        self.num_radial_basis = num_radial_basis

        # SH for re-embedding
        self.irreps_sh = Irreps.spherical_harmonics(reembed_lmax)
        self.sh = SphericalHarmonics(
            self.irreps_sh, normalize=True, normalization="component"
        )

        # Project edge scalars to smaller dim
        self.proj_dim = min(edge_hidden_dim, 64)
        self.scalar_proj = nn.Linear(edge_hidden_dim, self.proj_dim)

        # Radial MLP: RBF → one weight per l
        self.radial = nn.Sequential(
            nn.Linear(num_radial_basis, 64),
            nn.SiLU(),
            nn.Linear(64, reembed_lmax + 1, bias=False),
        )
        nn.init.uniform_(self.radial[-1].weight, -1e-3, 1e-3)

        # Re-embedded irreps: proj_dim copies of each l
        reembed_parts = [(self.proj_dim, ir) for _, ir in self.irreps_sh]
        self.reembed_irreps = Irreps(reembed_parts)

        # CG product
        from onsite import _build_self_instructions
        instructions = _build_self_instructions(self.reembed_irreps, mode=cg_mode)
        assert instructions, (
            f"No valid {cg_mode} CG paths for {self.reembed_irreps}."
        )
        self.cg = TensorProduct(
            self.reembed_irreps, self.reembed_irreps,
            self.reembed_irreps, instructions
        )

        # Invariant contraction sectors + output projection
        from onsite import _build_sectors
        self._sectors = _build_sectors(self.reembed_irreps)
        total_mul = sum(m for m, _ in self.reembed_irreps)
        self.out_proj = nn.Linear(total_mul, out_dim)

    def forward(self, e_bound: torch.Tensor, e_unbound: torch.Tensor,
                rel_vec: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        e_bound, e_unbound : (E, D) edge state scalars
        rel_vec : (E, 3) edge direction vectors (dst_pos - src_pos)
        """
        assert rel_vec is not None, "EdgeOnsiteReembedCG requires rel_vec"

        dist = rel_vec.norm(dim=-1, keepdim=False)  # (E,)

        # SH of edge direction
        sh_values = self.sh(rel_vec)  # (E, (lmax+1)^2)

        # Radial basis → weights per l
        bin_width = self.r_max / max(1, self.num_radial_basis - 1)
        rbf = soft_one_hot_linspace(
            dist, 0.0, self.r_max + bin_width,
            self.num_radial_basis, basis="smooth_finite", cutoff=True
        )
        rbf = rbf / rbf.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        radial_w = self.radial(rbf)  # (E, lmax+1)

        def _reembed_edge(e_scalar):
            h_proj = self.scalar_proj(e_scalar)  # (E, proj_dim)
            offset_sh = 0
            parts = []
            for l_idx, (_, ir) in enumerate(self.irreps_sh):
                d = 2 * ir.l + 1
                Y_l = sh_values[:, offset_sh:offset_sh + d]  # (E, d)
                R_l = radial_w[:, l_idx]  # (E,)
                # (E, proj_dim, d): each scalar channel modulated by Y_l * R_l
                feat = h_proj.unsqueeze(-1) * (R_l.unsqueeze(-1) * Y_l).unsqueeze(-2)
                parts.append(feat.reshape(feat.size(0), -1))  # (E, proj_dim * d)
                offset_sh += d
            return torch.cat(parts, dim=-1)  # (E, proj_dim * (lmax+1)^2)

        rb = _reembed_edge(e_bound)
        ru = _reembed_edge(e_unbound)

        # CG couple
        coupled = self.cg(rb, ru)

        # Invariant contraction against unbound reference
        from onsite import _invariant_contract
        dots = _invariant_contract(coupled, ru, self._sectors)

        return self.out_proj(dots)


_EDGE_ONSITE_REGISTRY = {
    "mlp": EdgeOnsiteMLP,
    "delta": EdgeOnsiteDelta,
    "bilinear": EdgeOnsiteBilinear,
    "reembed_cg": EdgeOnsiteReembedCG,
}


def _build_edge_comparison(edge_hidden_dim: int, out_dim: int, mode: str, **kwargs):
    """Build the comparison module (delta, mlp, bilinear, reembed_cg, or composite)."""
    parts = [m.strip() for m in mode.split(",")]
    if len(parts) == 1:
        cls = _EDGE_ONSITE_REGISTRY.get(parts[0])
        if cls is None:
            raise ValueError(
                f"Unknown edge onsite mode '{parts[0]}'. "
                f"Available: {list(_EDGE_ONSITE_REGISTRY)}"
            )
        if parts[0] == "reembed_cg":
            return cls(edge_hidden_dim, out_dim, **kwargs)
        return cls(edge_hidden_dim, out_dim)
    modules = []
    for m in parts:
        cls = _EDGE_ONSITE_REGISTRY.get(m)
        if cls is None:
            raise ValueError(
                f"Unknown edge onsite mode '{m}'. "
                f"Available: {list(_EDGE_ONSITE_REGISTRY)}"
            )
        if m == "reembed_cg":
            modules.append(cls(edge_hidden_dim, out_dim, **kwargs))
        else:
            modules.append(cls(edge_hidden_dim, out_dim))
    return CompositeEdgeOnsite(modules)


class EdgeOnsite(nn.Module):
    """Full edge onsite pipeline: encode endpoint nodes → compare → pool.

    Parameters
    ----------
    node_scalar_dim : int
        Dimension of l=0 scalar node features (cfg.scalars).
    edge_hidden_dim : int
        Intermediate edge representation dimension.
    edge_onsite_dim : int
        Per-module output dimension (total may differ for composite).
    mode : str
        Comparison mode: "mlp", "delta", or comma-separated composite.
    """

    def __init__(self, node_scalar_dim: int, edge_hidden_dim: int,
                 edge_onsite_dim: int, mode: str = "mlp", **kwargs):
        super().__init__()
        self.encoder = EdgeEncoder(node_scalar_dim, edge_hidden_dim)
        self.comparison = _build_edge_comparison(edge_hidden_dim, edge_onsite_dim,
                                                 mode, **kwargs)
        self.output_dim = self.comparison.output_dim
        self._needs_rel_vec = "reembed_cg" in mode

    def forward(self, h_bound_scalars: torch.Tensor, h_unbound_scalars: torch.Tensor,
                inter_src: torch.Tensor, inter_dst: torch.Tensor,
                edge_batch: torch.Tensor, batch_size: int,
                pos: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        h_bound_scalars : (N_total, D) bound node l=0 scalars in N-node space
        h_unbound_scalars : (N_total, D) unbound node l=0 scalars
        inter_src, inter_dst : (E_inter,) unique inter-molecular edge indices
        edge_batch : (E_inter,) graph assignment per edge
        batch_size : int
        pos : (N_total, 3) optional node positions (needed for reembed_cg)

        Returns
        -------
        (batch_size, output_dim) pooled edge onsite features
        """
        # Encode edges from both copies using shared encoder
        e_bound = self.encoder(h_bound_scalars[inter_src],
                               h_bound_scalars[inter_dst])
        e_unbound = self.encoder(h_unbound_scalars[inter_src],
                                 h_unbound_scalars[inter_dst])

        # Compare bound vs unbound
        kwargs = {}
        if self._needs_rel_vec:
            assert pos is not None, "reembed_cg edge comparison requires pos"
            kwargs["rel_vec"] = pos[inter_dst] - pos[inter_src]
        edge_feats = self.comparison(e_bound, e_unbound, **kwargs)

        # Pool per graph
        return scatter_sum(edge_feats, edge_batch, dim=0, dim_size=batch_size)


def build_edge_onsite(node_scalar_dim: int, edge_hidden_dim: int,
                      edge_onsite_dim: int, mode: str, **kwargs) -> EdgeOnsite:
    """Factory function for edge onsite module."""
    return EdgeOnsite(node_scalar_dim, edge_hidden_dim, edge_onsite_dim, mode,
                      **kwargs)


# ---------------------------------------------------------------------------
# Persistent edge state: edges updated every MP layer from endpoint nodes
# ---------------------------------------------------------------------------

class EdgeUpdateMLP(nn.Module):
    """Residual MLP that updates edge state from current state + endpoint info.

    Input: cat(edge_state, sym_endpoints, asym_endpoints) → 3 * edge_state_dim
    Output: edge_state + residual (same dim as edge_state)
    """

    def __init__(self, edge_state_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * edge_state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_state_dim),
        )

    def forward(self, edge_state: torch.Tensor,
                h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        """Update edge state with residual from endpoint scalars.

        Parameters
        ----------
        edge_state : (E, D) current edge state
        h_src, h_dst : (E, D) endpoint scalar features (already bottlenecked)
        """
        sym = h_src + h_dst
        asym = (h_src - h_dst).abs()
        return edge_state + self.mlp(torch.cat([edge_state, sym, asym], dim=-1))


class PersistentEdgeOnsite(nn.Module):
    """Persistent edge states updated every MP layer.

    Lifecycle:
      init:    edge_state = EdgeEncoder(node_scalars[src], node_scalars[dst])
      per-layer: edge_state += EdgeUpdateMLP(edge_state, bottleneck(src), bottleneck(dst))
      final:   compare(e_bound, e_unbound) → pool → graph features

    Parameters
    ----------
    node_scalar_dim : int
        Number of l=0 scalar channels in node features (cfg.n_scalar_channels).
    edge_state_dim : int
        Persistent edge hidden dimension.
    edge_hidden_dim : int
        Hidden dim for the initial EdgeEncoder.
    edge_onsite_dim : int
        Per-module output dim for comparison.
    mode : str
        Comparison mode: "mlp", "delta", "delta,mlp".
    edge_feedback : bool
        If True, provides scatter-back projection from edges to nodes.
    """

    def __init__(self, node_scalar_dim: int, edge_state_dim: int,
                 edge_hidden_dim: int, edge_onsite_dim: int,
                 mode: str = "delta,mlp", edge_feedback: bool = False,
                 **kwargs):
        super().__init__()

        # Bottleneck: project node scalars → edge_state_dim for update inputs
        self.bottleneck = nn.Linear(node_scalar_dim, edge_state_dim)

        # Initial encoder: node scalars → edge_state_dim
        self.encoder = EdgeEncoder(node_scalar_dim, edge_state_dim)

        # Per-layer update (weight-tied across layers)
        update_hidden = min(2 * edge_state_dim, 128)
        self.update_mlp = EdgeUpdateMLP(edge_state_dim, update_hidden)

        # Final comparison (reuse existing modules)
        self.comparison = _build_edge_comparison(edge_state_dim, edge_onsite_dim,
                                                 mode, **kwargs)
        self.output_dim = self.comparison.output_dim

        # Optional feedback: project edge state back to node scalar space
        self.edge_feedback = edge_feedback
        if edge_feedback:
            self.feedback_proj = nn.Linear(edge_state_dim, node_scalar_dim)
        else:
            self.feedback_proj = None

    def init_edge_states(self, node_scalars: torch.Tensor,
                         inter_src: torch.Tensor,
                         inter_dst: torch.Tensor) -> torch.Tensor:
        """Initialize edge states from endpoint node scalars.

        Parameters
        ----------
        node_scalars : (N, D) l=0 scalar features in N-node space
        inter_src, inter_dst : (E_inter,) inter-molecular edge indices

        Returns
        -------
        (E_inter, edge_state_dim) initial edge states
        """
        return self.encoder(node_scalars[inter_src], node_scalars[inter_dst])

    def update_edge_states(self, edge_state: torch.Tensor,
                           node_scalars: torch.Tensor,
                           inter_src: torch.Tensor,
                           inter_dst: torch.Tensor) -> torch.Tensor:
        """Update edge states from current node scalars (one MP layer).

        Parameters
        ----------
        edge_state : (E_inter, edge_state_dim)
        node_scalars : (N, D) current l=0 scalars in N-node space
        inter_src, inter_dst : (E_inter,) edge indices

        Returns
        -------
        (E_inter, edge_state_dim) updated edge states
        """
        h_src = self.bottleneck(node_scalars[inter_src])
        h_dst = self.bottleneck(node_scalars[inter_dst])
        return self.update_mlp(edge_state, h_src, h_dst)

    def scatter_feedback(self, edge_state: torch.Tensor,
                         inter_src: torch.Tensor,
                         inter_dst: torch.Tensor,
                         num_nodes: int) -> torch.Tensor:
        """Scatter edge state back to nodes (both endpoints).

        Returns (N, node_scalar_dim) feedback to add to node scalars.
        """
        proj = self.feedback_proj(edge_state)
        fb = torch.zeros(num_nodes, proj.size(-1), device=proj.device)
        fb.scatter_add_(0, inter_src.unsqueeze(-1).expand_as(proj), proj)
        fb.scatter_add_(0, inter_dst.unsqueeze(-1).expand_as(proj), proj)
        return fb

    def compare_and_pool(self, e_bound: torch.Tensor, e_unbound: torch.Tensor,
                         edge_batch: torch.Tensor, batch_size: int,
                         rel_vec: torch.Tensor = None) -> torch.Tensor:
        """Compare bound vs unbound edge states, pool per graph.

        Returns (batch_size, output_dim).
        """
        kwargs = {}
        if rel_vec is not None:
            kwargs["rel_vec"] = rel_vec
        edge_feats = self.comparison(e_bound, e_unbound, **kwargs)
        return scatter_sum(edge_feats, edge_batch, dim=0, dim_size=batch_size)


def build_persistent_edge_onsite(node_scalar_dim: int, edge_state_dim: int,
                                 edge_hidden_dim: int, edge_onsite_dim: int,
                                 mode: str, edge_feedback: bool = False,
                                 **kwargs) -> PersistentEdgeOnsite:
    """Factory for persistent edge onsite module."""
    return PersistentEdgeOnsite(
        node_scalar_dim=node_scalar_dim,
        edge_state_dim=edge_state_dim,
        edge_hidden_dim=edge_hidden_dim,
        edge_onsite_dim=edge_onsite_dim,
        mode=mode,
        edge_feedback=edge_feedback,
        **kwargs,
    )
