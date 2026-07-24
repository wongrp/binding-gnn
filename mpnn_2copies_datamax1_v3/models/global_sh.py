"""
Global SH Pooling: angular descriptors for readout.

Encodes the angular distribution of pocket atoms relative to the ligand
centroid using spherical harmonics, then computes a cross power spectrum
between bound and unbound representations.

Modes:
  - "cross"  (default): cross power spectrum P_{cc'} = Σ_m F^b_{c,m} F^ub_{c',m}
  - "cross_l_norm": cross power spectrum + squared norms of cross-l CG outputs
    for (l1≠l2) → L>0, giving invariants that the power spectrum misses.

Head modes (global_sh_head):
  - "none"  (default): cross power spectrum only, projected by nn.Linear.
  - "self_cg" / "gate" / "gate_cg": replace power spectrum with full CG product
    + equivariant head (see EquivariantHead).
  - "augment_self_cg" / "augment_gate_cg": keep the full cross power spectrum,
    add L>0 CG features via an equivariant head, then project the concatenation.
    Fixes the bottleneck where replace-mode heads discard cross-channel information.

Paper Section 6.2, Eq. (global_sh).  2-copy mode only.
"""

import torch
import torch.nn as nn
from e3nn.o3 import SphericalHarmonics, wigner_3j, Irreps, Linear, FullyConnectedTensorProduct
from e3nn.nn import Gate
from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter_mean, scatter_sum


def _precompute_cross_l_cg(lmax):
    """Precompute CG coefficients for (l1, l2) → L with l1 ≠ l2, L > 0.

    Returns a list of (l1, l2, L, cg_matrix) tuples where cg_matrix has
    shape [(2*l1+1), (2*l2+1), (2*L+1)].
    """
    paths = []
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            if l1 == l2:
                continue  # same-l already captured by power spectrum
            for L in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                if L == 0:
                    continue  # L=0 with l1=l2 is power spectrum; l1≠l2→L=0 is forbidden
                cg = wigner_3j(l1, l2, L)  # [2l1+1, 2l2+1, 2L+1]
                paths.append((l1, l2, L, cg))
    return paths


class EquivariantHead(nn.Module):
    """Small equivariant network: per-graph irreps -> scalars.

    Modes:
      "gate"    -- Gate nonlinearity (SiLU on scalars, sigmoid gates on l>0),
                   then extract l=0 scalars.
      "self_cg" -- Self-TP (h x h) targeting scalar output only (learned
                   cross-channel contraction), then linear.
      "gate_cg" -- Gate first, then self-TP -> scalars.
    """

    def __init__(self, irreps_in, out_dim, mode="gate", n_tp_out=None):
        super().__init__()
        self.mode = mode
        self.irreps_in = Irreps(irreps_in)

        n_scalar_in = sum(m for m, ir in self.irreps_in if ir.l == 0)

        if mode in ("gate", "gate_cg"):
            scalars = Irreps([(m, ir) for m, ir in self.irreps_in if ir.l == 0])
            vectors = Irreps([(m, ir) for m, ir in self.irreps_in if ir.l > 0])
            n_gates = sum(m for m, _ in vectors)
            gates_ir = Irreps(f"{n_gates}x0e") if n_gates else Irreps("")

            self.gate = Gate(
                scalars, [nn.SiLU()] * len(scalars),
                gates_ir, [nn.Sigmoid()] * len(gates_ir),
                vectors,
            )
            self.pre_gate = Linear(self.irreps_in, self.gate.irreps_in)
            gate_out_irreps = self.gate.irreps_out
            self._n_scalar_gate = sum(m for m, ir in gate_out_irreps if ir.l == 0)

        if mode == "gate":
            self.out_proj = nn.Linear(self._n_scalar_gate, out_dim)
        elif mode == "self_cg":
            n_out = n_tp_out if n_tp_out is not None else n_scalar_in
            tp_out = Irreps(f"{n_out}x0e")
            self.self_tp = FullyConnectedTensorProduct(
                self.irreps_in, self.irreps_in, tp_out
            )
            self.out_proj = nn.Sequential(nn.SiLU(), nn.Linear(n_out, out_dim))
        elif mode == "gate_cg":
            n_out = n_tp_out if n_tp_out is not None else self._n_scalar_gate
            tp_out = Irreps(f"{n_out}x0e")
            self.self_tp = FullyConnectedTensorProduct(
                gate_out_irreps, gate_out_irreps, tp_out
            )
            self.out_proj = nn.Sequential(
                nn.SiLU(), nn.Linear(n_out, out_dim)
            )
        else:
            raise ValueError(f"Unknown EquivariantHead mode: {mode}")

    def forward(self, x):
        if self.mode == "gate":
            x = self.pre_gate(x)
            x = self.gate(x)
            return self.out_proj(x[:, :self._n_scalar_gate])
        elif self.mode == "self_cg":
            x = self.self_tp(x, x)
            return self.out_proj(x)
        elif self.mode == "gate_cg":
            x = self.pre_gate(x)
            x = self.gate(x)
            x = self.self_tp(x, x)
            return self.out_proj(x)


class GlobalSHPooling(nn.Module):
    """Global angular descriptors comparing bound vs unbound pocket shells.

    Parameters
    ----------
    n_scalar_in : int
        Number of scalar channels per node (l=0 from hidden irreps).
    n_channels : int
        Number of learned weight channels (C).
    lmax : int
        Maximum spherical harmonic degree.
    out_dim : int
        Output dimension after projecting the power spectrum.
    num_rbf : int
        Number of radial basis functions.
    r_max : float
        Maximum radial distance for RBF (should cover pocket diameter / 2).
    separate_proj : bool
        If True, use separate weight projections for bound and unbound.
    mode : str
        "cross" for standard cross power spectrum.
        "cross_l_norm" for cross power spectrum + cross-l CG norm invariants.
    """

    def __init__(self, n_scalar_in: int, n_channels: int = 16, lmax: int = 2,
                 out_dim: int = 64, num_rbf: int = 20, r_max: float = 20.0,
                 separate_proj: bool = False, mode: str = "cross",
                 head: str = "none", n_heads: int = 1,
                 esm_proj_dim: int = 0, esm_input_dim: int = 1280,
                 radial_form: str = "learned"):
        super().__init__()
        self.n_channels = n_channels
        self.lmax = lmax
        self.out_dim = out_dim
        self.num_rbf = num_rbf
        self.r_max = r_max
        self.separate_proj = separate_proj
        self.mode = mode
        self.head = head
        self.n_heads = n_heads
        self.esm_proj_dim = esm_proj_dim
        self.radial_form = radial_form

        # ESM projection (per-residue evolutionary features → compact dim)
        self.esm_proj = None
        if esm_proj_dim > 0:
            self.esm_proj = nn.Sequential(
                nn.Linear(esm_input_dim, esm_proj_dim),
                nn.SiLU(),
            )

        # Spherical harmonics
        self.sh = SphericalHarmonics(
            list(range(lmax + 1)), normalize=True, normalization="component"
        )

        if radial_form == "learned":
            # Default: learned MLP from RBF → one weight per l (shared across heads).
            self.radial = nn.Sequential(
                nn.Linear(num_rbf, 64),
                nn.SiLU(),
                nn.Linear(64, lmax + 1, bias=False),
            )
            nn.init.uniform_(self.radial[-1].weight, -1e-3, 1e-3)
        elif radial_form == "multipole":
            # Physically motivated: R_l(r) = (r^2 + r_c^2)^(-(l+1)/2) * cos_cutoff(r),
            # the multipole tail with a soft core at r_c (avoids r=0 singularity)
            # and a cosine cutoff at r_max (matches the learned RBF's cutoff=True).
            # No learned parameters.
            self.radial = None
            self._multipole_clip = 0.5  # Angstrom; below typical pocket-atom distances
        else:
            raise ValueError(
                f"Unknown radial_form: {radial_form}. "
                f"Expected one of: learned, multipole."
            )

        # Per-node channel weights from scalar features (one projection per head)
        # When ESM is active, input dim grows by esm_proj_dim
        weight_in_dim = n_scalar_in + esm_proj_dim
        self.weight_projs = nn.ModuleList([
            nn.Linear(weight_in_dim, n_channels) for _ in range(n_heads)
        ])
        if separate_proj:
            self.weight_projs_unbound = nn.ModuleList([
                nn.Linear(weight_in_dim, n_channels) for _ in range(n_heads)
            ])
        else:
            self.weight_projs_unbound = self.weight_projs
        # Backward compat: single-head code paths use these
        self.weight_proj = self.weight_projs[0]
        self.weight_proj_unbound = self.weight_projs_unbound[0]

        if n_heads > 1 and head != "none":
            raise ValueError(
                f"n_heads > 1 only supported with head='none', got head='{head}'"
            )

        if head.startswith("augment_"):
            # Augment mode: cross power spectrum + L>0 CG features
            sub_mode = head[len("augment_"):]
            if sub_mode == "gate":
                raise ValueError(
                    "augment_gate is not supported: L>0-only input has no L=0 "
                    "scalars for Gate output. Use augment_self_cg or augment_gate_cg."
                )

            # A. Cross power spectrum path (same as head="none")
            cross_dim = n_channels * n_channels * (lmax + 1)
            self._cross_l_paths = []
            cross_l_dim = 0
            if mode == "cross_l_norm":
                self._cross_l_paths = _precompute_cross_l_cg(lmax)
                cross_l_dim = n_channels * n_channels * len(self._cross_l_paths)
                for i, (l1, l2, L, cg) in enumerate(self._cross_l_paths):
                    self.register_buffer(f"_cg_{i}", cg)

            # B. CG product for L>0 only (L=0 is redundant with power spectrum)
            self._augment_cg_paths = []
            for l1 in range(lmax + 1):
                for l2 in range(lmax + 1):
                    for L in range(abs(l1 - l2), l1 + l2 + 1):
                        if L == 0:
                            continue
                        cg = wigner_3j(l1, l2, L)
                        self._augment_cg_paths.append((l1, l2, L))
                        self.register_buffer(
                            f"_augment_cg_{len(self._augment_cg_paths) - 1}", cg
                        )

            # L>0 output irreps
            cg_irreps_l_gt_0 = Irreps("+".join(
                f"{n_channels}x{L}{'e' if L % 2 == 0 else 'o'}"
                for L in range(1, 2 * lmax + 1)
            ))
            self.eq_head = EquivariantHead(
                cg_irreps_l_gt_0, out_dim, mode=sub_mode, n_tp_out=n_channels
            )

            # C. Final projection: power spectrum + equivariant head scalars
            inner_dim = cross_dim + cross_l_dim + out_dim
            self.out_proj = nn.Linear(inner_dim, out_dim)

        elif head != "none":
            # Full CG product mode (replace): precompute ALL CG coefficients
            self._full_cg_paths = []
            for l1 in range(lmax + 1):
                for l2 in range(lmax + 1):
                    for L in range(abs(l1 - l2), l1 + l2 + 1):
                        cg = wigner_3j(l1, l2, L)
                        self._full_cg_paths.append((l1, l2, L))
                        self.register_buffer(
                            f"_full_cg_{len(self._full_cg_paths) - 1}", cg
                        )

            # Output irreps from CG product: C x (0e + 1o + 2e + ... + (2*lmax)e/o)
            cg_irreps = Irreps("+".join(
                f"{n_channels}x{L}{'e' if L % 2 == 0 else 'o'}"
                for L in range(2 * lmax + 1)
            ))
            self.eq_head = EquivariantHead(cg_irreps, out_dim, mode=head)
        else:
            # Original behavior: cross power spectrum + optional cross-l norm
            cross_dim = n_channels * n_channels * (lmax + 1)

            # Cross-l norm invariants
            self._cross_l_paths = []
            cross_l_dim = 0
            if mode == "cross_l_norm":
                self._cross_l_paths = _precompute_cross_l_cg(lmax)
                cross_l_dim = n_channels * n_channels * len(self._cross_l_paths)
                for i, (l1, l2, L, cg) in enumerate(self._cross_l_paths):
                    self.register_buffer(f"_cg_{i}", cg)

            inner_dim = (cross_dim + cross_l_dim) * n_heads
            self.out_proj = nn.Linear(inner_dim, out_dim)

        self.output_dim = out_dim

    def forward(self, h_bound_scalars, h_unbound_scalars, pos, batch, ligand_mask, batch_size,
                esm_residue_features=None):
        """
        Parameters
        ----------
        h_bound_scalars : [N, n_scalar_in]
        h_unbound_scalars : [N, n_scalar_in]
        pos : [N, 3]
        batch : [N]
        ligand_mask : [N]
        batch_size : int
        esm_residue_features : [N_residues, esm_input_dim] or None
            Per-residue ESM embeddings, already mapped to atom/residue indices.
            Only used when esm_proj_dim > 0. Passed as None otherwise.
        """
        pocket_mask = ~ligand_mask

        # 1. Ligand centroid per graph
        lig_pos = pos[ligand_mask]
        lig_batch = batch[ligand_mask]
        centroid = scatter_mean(lig_pos, lig_batch, dim=0, dim_size=batch_size)

        # 2. Pocket atom vectors from centroid
        pocket_pos = pos[pocket_mask]
        pocket_batch = batch[pocket_mask]
        rel = pocket_pos - centroid[pocket_batch]
        dist = rel.norm(dim=-1)

        # 3. Spherical harmonics Y_l^m(r̂)
        sh_values = self.sh(rel)  # [N_pocket, (lmax+1)²]

        # 4. Radial weights R_l(r)
        if self.radial_form == "learned":
            bin_width = self.r_max / max(1, self.num_rbf - 1)
            rbf = soft_one_hot_linspace(
                dist, 0.0, self.r_max + bin_width,
                self.num_rbf, basis="smooth_finite", cutoff=True
            )
            rbf = rbf / rbf.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            radial_w = self.radial(rbf)  # [N_pocket, lmax+1]
        else:
            # Multipole: R_l(r) = (r^2 + r_c^2)^(-(l+1)/2) * cos_cutoff(r)
            r_clip = self._multipole_clip
            r_safe = torch.sqrt(dist * dist + r_clip * r_clip)  # [N_pocket]
            l_orders = torch.arange(self.lmax + 1, dtype=dist.dtype, device=dist.device)
            exponents = -(l_orders + 1)  # [-1, -2, ..., -(lmax+1)]
            multipole = r_safe.unsqueeze(-1) ** exponents.unsqueeze(0)  # [N_pocket, lmax+1]
            inside = dist < self.r_max
            cutoff = torch.zeros_like(dist)
            cutoff[inside] = 0.5 * (
                1.0 + torch.cos(torch.pi * dist[inside] / self.r_max)
            )
            radial_w = multipole * cutoff.unsqueeze(-1)

        # 5. Precompute radially-weighted SH per l (shared across heads)
        weighted_Y_parts = []
        sh_offset = 0
        for l_idx in range(self.lmax + 1):
            d = 2 * l_idx + 1
            Y_l = sh_values[:, sh_offset:sh_offset + d]
            R_l = radial_w[:, l_idx:l_idx + 1]
            weighted_Y_parts.append(R_l * Y_l)  # [N_pocket, d]
            sh_offset += d

        # 6. Build per-graph SH expansion per head
        h_b_pocket = h_bound_scalars[pocket_mask]
        h_u_pocket = h_unbound_scalars[pocket_mask]

        # Concatenate projected ESM features to weight projection input
        if self.esm_proj is not None and esm_residue_features is not None:
            esm_pocket = esm_residue_features[pocket_mask]
            esm_projected = self.esm_proj(esm_pocket)
            h_b_pocket = torch.cat([h_b_pocket, esm_projected], dim=-1)
            h_u_pocket = torch.cat([h_u_pocket, esm_projected], dim=-1)

        all_F_bound = []   # list of n_heads lists
        all_F_unbound = []
        for head_idx in range(self.n_heads):
            w_bound = self.weight_projs[head_idx](h_b_pocket)
            w_unbound = self.weight_projs_unbound[head_idx](h_u_pocket)

            F_bound_parts = []
            F_unbound_parts = []
            for l_idx in range(self.lmax + 1):
                d = 2 * l_idx + 1
                weighted_Y = weighted_Y_parts[l_idx]

                msg_b = w_bound.unsqueeze(-1) * weighted_Y.unsqueeze(-2)
                msg_u = w_unbound.unsqueeze(-1) * weighted_Y.unsqueeze(-2)

                agg_b = scatter_sum(msg_b.reshape(msg_b.size(0), -1),
                                    pocket_batch, dim=0, dim_size=batch_size)
                agg_u = scatter_sum(msg_u.reshape(msg_u.size(0), -1),
                                    pocket_batch, dim=0, dim_size=batch_size)

                F_bound_parts.append(agg_b.view(batch_size, self.n_channels, d))
                F_unbound_parts.append(agg_u.view(batch_size, self.n_channels, d))

            all_F_bound.append(F_bound_parts)
            all_F_unbound.append(F_unbound_parts)

        # For single-head backward compat (CG/augment paths use these)
        F_bound_parts = all_F_bound[0]
        F_unbound_parts = all_F_unbound[0]

        if self.head != "none" and not self.head.startswith("augment_"):
            # 7. Full CG product between bound and unbound representations
            device = F_bound_parts[0].device
            output_by_L = {}
            for L in range(2 * self.lmax + 1):
                output_by_L[L] = torch.zeros(
                    batch_size, self.n_channels, 2 * L + 1, device=device
                )

            for i, (l1, l2, L) in enumerate(self._full_cg_paths):
                cg = getattr(self, f"_full_cg_{i}")  # [2l1+1, 2l2+1, 2L+1]
                Fb = F_bound_parts[l1]   # [B, C, 2l1+1]
                Fu = F_unbound_parts[l2]  # [B, C, 2l2+1]

                # T^L_{c,M} = Σ_{m1,m2} CG^{LM}_{l1m1,l2m2} F^b_{c,m1} F^ub_{c,m2}
                Fb_cg = torch.einsum('bcm, mnL -> bcnL', Fb, cg)
                T = torch.einsum('bcnL, bcn -> bcL', Fb_cg, Fu)
                output_by_L[L] = output_by_L[L] + T

            # Assemble into irreps tensor [B, C * Σ_L (2L+1)]
            cg_features = torch.cat(
                [output_by_L[L].reshape(batch_size, -1)
                 for L in range(2 * self.lmax + 1)],
                dim=-1,
            )

            # 8. Equivariant head -> scalars
            return self.eq_head(cg_features)

        # 7. Cross power spectrum per head: P^(l)_{cc'} = Σ_m F^b_{c,m} · F^ub_{c',m}
        power_parts = []
        for head_idx in range(self.n_heads):
            Fb_parts = all_F_bound[head_idx]
            Fu_parts = all_F_unbound[head_idx]

            for l_idx in range(self.lmax + 1):
                Fb = Fb_parts[l_idx]
                Fu = Fu_parts[l_idx]
                P = torch.bmm(Fb, Fu.transpose(1, 2))  # [B, C, C]
                power_parts.append(P.reshape(batch_size, -1))

            # 8. Cross-l CG norm invariants (if enabled)
            if self.mode == "cross_l_norm" and self._cross_l_paths:
                for i, (l1, l2, L, _) in enumerate(self._cross_l_paths):
                    cg = getattr(self, f"_cg_{i}")
                    Fb = Fb_parts[l1]
                    Fu = Fu_parts[l2]
                    Fb_cg = torch.einsum('bcm, mnL -> bcnL', Fb, cg)
                    T = torch.einsum('bcnL, bdn -> bcdL', Fb_cg, Fu)
                    T_norm_sq = (T * T).sum(dim=-1)
                    power_parts.append(T_norm_sq.reshape(batch_size, -1))

        power = torch.cat(power_parts, dim=-1)

        # 9. Augment with L>0 CG features if applicable
        if self.head.startswith("augment_"):
            device = F_bound_parts[0].device
            max_L = 2 * self.lmax
            output_by_L = {}
            for L in range(1, max_L + 1):
                output_by_L[L] = torch.zeros(
                    batch_size, self.n_channels, 2 * L + 1, device=device
                )

            for i, (l1, l2, L) in enumerate(self._augment_cg_paths):
                cg = getattr(self, f"_augment_cg_{i}")
                Fb = F_bound_parts[l1]
                Fu = F_unbound_parts[l2]
                Fb_cg = torch.einsum('bcm, mnL -> bcnL', Fb, cg)
                T = torch.einsum('bcnL, bcn -> bcL', Fb_cg, Fu)
                output_by_L[L] = output_by_L[L] + T

            cg_features = torch.cat(
                [output_by_L[L].reshape(batch_size, -1)
                 for L in range(1, max_L + 1)],
                dim=-1,
            )
            eq_scalars = self.eq_head(cg_features)
            power = torch.cat([power, eq_scalars], dim=-1)

        # 10. Project to output dim
        return self.out_proj(power)
