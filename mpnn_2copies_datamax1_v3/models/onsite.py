"""
Onsite interaction modules for combining bound and unbound representations.

Architecture
------------
Equivariant cores   (h_bound, h_unbound) → full irreps   [.output_irreps]
Scalar modules      (h_bound, h_unbound) → scalars        [.output_dim]
WithProjection      wraps any core → scalars  (dot-product against ref)
CompositeOnsite     runs multiple modules, concatenates outputs

All cores output irreps_in by convention, enabling composition:

    core1 = build_onsite_core(irreps, "linear_combo")
    core2 = build_onsite_core(irreps, "cg")
    # compatible because both output irreps_in

Factory
-------
build_onsite_core(irreps_in, mode)         → equivariant core
build_onsite(irreps_in, out_dim, mode)     → scalar-output module (for model)
                                             mode can be comma-separated for composite
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps, Linear, TensorProduct, SphericalHarmonics
from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter, scatter_softmax
from torch_geometric.utils import to_dense_batch


# ───────────────────────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────────────────────

def _build_self_instructions(irreps, mode='uvu'):
    """CG-valid TensorProduct instructions for self-coupling (h ⊗ h).

    Both inputs and the output share the same irreps.

    Parameters
    ----------
    mode : {"uvu", "uuu"}
        uvu: input-1 and output share a multiplicity index.
             O(n_c²) weights per path.  Works for any irreps.
        uuu: all three indices match (elementwise in multiplicity space).
             O(n_c) weights per path.  Requires mul_in1 == mul_in2 == mul_out
             for each included path — only paths between equal-multiplicity
             sectors are valid.
    """
    instructions = []
    for i1, (m1, ir1) in enumerate(irreps):
        for i2, (m2, ir2) in enumerate(irreps):
            for io, (mo, iro) in enumerate(irreps):
                if iro not in ir1 * ir2:
                    continue
                if mode == 'uvu' and m1 == mo:
                    instructions.append((i1, i2, io, 'uvu', True))
                elif mode == 'uuu' and m1 == m2 == mo:
                    instructions.append((i1, i2, io, 'uuu', True))
    return instructions


def _invariant_contract(features, ref, sectors):
    """Per-(mul, l) dot product: features^(l) · ref^(l) → [N, total_mul].

    l=0 channels: signed element-wise product (preserves sign).
    l>0 channels: Σ_m f_m * r_m  per multiplicity (alignment score, signed).
    """
    dots = []
    for offset, mul, d in sectors:
        f = features[:, offset:offset + mul * d].view(-1, mul, d)
        r = ref[:, offset:offset + mul * d].view(-1, mul, d)
        dots.append((f * r).sum(dim=-1))          # [N, mul]
    return torch.cat(dots, dim=-1)


def _build_sectors(irreps):
    """Return list of (offset, mul, 2l+1) tuples for iterating over irreps."""
    sectors = []
    offset = 0
    for mul, ir in irreps:
        d = 2 * ir.l + 1
        sectors.append((offset, mul, d))
        offset += mul * d
    return sectors


def _polynomial_envelope(dist, cutoff):
    """Smooth polynomial cutoff: 1 at r=0, 0 at r=cutoff."""
    x = (dist / cutoff).clamp(max=1.0)
    return (1.0 - x * x) ** 3


# ═══════════════════════════════════════════════════════════════════════════
#  Equivariant Cores — all output irreps_in
# ═══════════════════════════════════════════════════════════════════════════

class DeltaCore(nn.Module):
    """h_bound − h_unbound.  Parameter-free.  output_irreps = irreps_in."""

    def __init__(self, irreps_in: Irreps):
        super().__init__()
        self.output_irreps = irreps_in

    def forward(self, h_bound, h_unbound, **kw):
        return h_bound - h_unbound


class LinearComboCore(nn.Module):
    """Learned  W_b @ h_bound + W_u @ h_unbound   (two e3nn Linears).

    Each Linear mixes multiplicities within a given l sector independently,
    which is the correct equivariant generalisation of per-channel α, β.
    Higher-l outputs are preserved for downstream use.
    """

    def __init__(self, irreps_in: Irreps):
        super().__init__()
        self.proj_bound = Linear(irreps_in, irreps_in)
        self.proj_unbound = Linear(irreps_in, irreps_in)
        self.output_irreps = irreps_in

    def forward(self, h_bound, h_unbound, **kw):
        return self.proj_bound(h_bound) + self.proj_unbound(h_unbound)


class CGProductCore(nn.Module):
    """CG tensor product  h_bound ⊗ h_unbound.

    Connection mode is controlled by ``cg_mode``:

    - ``"uvu"``: input-1 and output share a multiplicity index.
      O(n_c²) weights per CG path.  Works for any irreps.
    - ``"uuu"``: all three multiplicity indices match (elementwise).
      O(n_c) weights per path.  Only paths between equal-multiplicity
      sectors are included — use with true-diagonal irreps (Nx0e + Nx1o + …).

    Output has the same irreps as input (irreps_in).
    """

    def __init__(self, irreps_in: Irreps, cg_mode: str = "uvu"):
        super().__init__()
        instructions = _build_self_instructions(irreps_in, mode=cg_mode)
        assert instructions, (
            f"No valid {cg_mode} CG paths for {irreps_in}. "
            f"For uuu, all coupled sectors must have equal multiplicity."
        )
        self.tp = TensorProduct(irreps_in, irreps_in, irreps_in, instructions)
        self.output_irreps = irreps_in

    def forward(self, h_bound, h_unbound, **kw):
        return self.tp(h_bound, h_unbound)


class GatingCore(nn.Module):
    """Bound l=0 scalars gate unbound l>0 features (and reverse).

    One gate value per (multiplicity, l) channel, broadcast across its
    2l+1 m-components — this preserves equivariance.

    Raw output includes a scalar delta for completeness, then an e3nn
    Linear projects back to irreps_in for composability.

    lmax > 0 ::

        fwd = σ(W_fwd @ bound_l0) ⊙ unbound_{l>0}    (per-mul gate)
        rev = σ(W_rev @ unbound_l0) ⊙ bound_{l>0}
        raw = cat(scalar_delta, fwd, rev)
        out = Linear(raw_irreps → irreps_in)

    lmax = 0  (bidirectional GLU) ::

        raw = cat(σ(W_fwd @ h_b) ⊙ h_u,  σ(W_rev @ h_u) ⊙ h_b)
        out = Linear(raw → irreps_in)
    """

    def __init__(self, irreps_in: Irreps):
        super().__init__()
        self.irreps_in = irreps_in

        n_scalar = sum(m for m, ir in irreps_in if ir.l == 0)
        nonscalar_sectors = [(m, ir) for m, ir in irreps_in if ir.l > 0]
        n_nonscalar_mul = sum(m for m, ir in nonscalar_sectors)
        self.n_scalar = n_scalar
        self.has_nonscalar = n_nonscalar_mul > 0

        if self.has_nonscalar:
            self.gate_fwd = nn.Linear(n_scalar, n_nonscalar_mul)
            self.gate_rev = nn.Linear(n_scalar, n_nonscalar_mul)

            # one gate per (mul, l), broadcast across 2l+1 components
            repeats = []
            for m, ir in nonscalar_sectors:
                repeats.extend([2 * ir.l + 1] * m)
            self.register_buffer('_repeats', torch.tensor(repeats))

            scalar_irreps = Irreps([(m, ir) for m, ir in irreps_in if ir.l == 0])
            doubled_ns = Irreps([(2 * m, ir) for m, ir in nonscalar_sectors])
            raw_irreps = (scalar_irreps + doubled_ns).simplify()
        else:
            # scalar-only: bidirectional GLU
            self.gate_fwd = nn.Linear(n_scalar, n_scalar)
            self.gate_rev = nn.Linear(n_scalar, n_scalar)
            raw_irreps = Irreps(f"{2 * n_scalar}x0e")

        self.out_proj = Linear(raw_irreps, irreps_in)
        self.output_irreps = irreps_in

    def forward(self, h_bound, h_unbound, **kw):
        sb = h_bound[:, :self.n_scalar]
        su = h_unbound[:, :self.n_scalar]

        if self.has_nonscalar:
            nb = h_bound[:, self.n_scalar:]
            nu = h_unbound[:, self.n_scalar:]

            g_fwd = torch.sigmoid(self.gate_fwd(sb))
            g_fwd = g_fwd.repeat_interleave(self._repeats, dim=-1)
            gated_fwd = g_fwd * nu

            g_rev = torch.sigmoid(self.gate_rev(su))
            g_rev = g_rev.repeat_interleave(self._repeats, dim=-1)
            gated_rev = g_rev * nb

            raw = torch.cat([sb - su, gated_fwd, gated_rev], dim=-1)
        else:
            gated_fwd = torch.sigmoid(self.gate_fwd(sb)) * su
            gated_rev = torch.sigmoid(self.gate_rev(su)) * sb
            raw = torch.cat([gated_fwd, gated_rev], dim=-1)

        return self.out_proj(raw)


# ═══════════════════════════════════════════════════════════════════════════
#  Scalar Modules — output [N, output_dim]
# ═══════════════════════════════════════════════════════════════════════════

class MLPOnsite(nn.Module):
    """Concat l=0 scalars from bound + unbound → 2-layer MLP → scalars.

    Only uses l=0 channels.  For lmax > 0, consider equivariant cores
    (delta, linear_combo, cg, gate) wrapped with WithProjection instead.
    """

    def __init__(self, irreps_in: Irreps, out_dim: int, **kw):
        super().__init__()
        n_scalar = sum(m for m, ir in irreps_in if ir.l == 0)
        has_nonscalar = any(ir.l > 0 for _, ir in irreps_in)
        if has_nonscalar:
            warnings.warn(
                f"MLPOnsite only uses l=0 channels ({n_scalar} scalars) from "
                f"irreps_in={irreps_in}. Higher-l features are discarded. "
                "Consider an equivariant core (cg, gate) + WithProjection for lmax > 0.",
                stacklevel=2,
            )
        self.n_scalar = n_scalar
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_scalar, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.output_dim = out_dim

    def forward(self, h_bound, h_unbound, **kw):
        sb = h_bound[:, :self.n_scalar]
        su = h_unbound[:, :self.n_scalar]
        return self.mlp(torch.cat([sb, su], dim=-1))


class InvariantProjection(nn.Module):
    """Per-(mul, l) dot product  h_bound^(l) · h_unbound^(l) → scalar.

    The core contraction is **parameter-free**; a learned Linear maps the
    resulting alignment scores to the desired output dimension.

    For lmax = 0 this coincides with element-wise product + linear — the
    same operation as CGProductCore but without learned coupling weights.
    The distinction matters at lmax > 0: InvariantProjection is a
    parameter-free (l,l)→0 contraction (dot product per channel), while
    CGProductCore learns a bilinear weight tensor via uvu CG coupling.
    Use InvariantProjection when you want a cheap alignment readout;
    use CGProductCore when the coupling itself should be learned.
    """

    def __init__(self, irreps_in: Irreps, out_dim: int, **kw):
        super().__init__()
        self.output_dim = out_dim
        self._sectors = _build_sectors(irreps_in)
        total_mul = sum(m for m, _ in irreps_in)
        self.out_proj = nn.Linear(total_mul, out_dim)

    def forward(self, h_bound, h_unbound, **kw):
        dots = _invariant_contract(h_bound, h_unbound, self._sectors)
        return self.out_proj(dots)


class DirectCGOnsite(nn.Module):
    """CG product h_bound ⊗ h_unbound → Linear projection (no dot-product contraction).

    Same CG computation as WithProjection(CGProductCore(...)), but skips
    the dot-product contraction against the reference.  For lmax=0:

        WithProjection:  Linear(W * h_b * h_u * h_u)  = degree (1,2)
        DirectCG:        Linear(W * h_b * h_u)         = degree (1,1)
    """

    def __init__(self, irreps_in: Irreps, out_dim: int, cg_mode: str = "uvu", **kw):
        super().__init__()
        self.core = CGProductCore(irreps_in, cg_mode=cg_mode)
        total_dim = sum(m * (2 * ir.l + 1) for m, ir in irreps_in)
        self.out_proj = nn.Linear(total_dim, out_dim)
        self.output_dim = out_dim

    def forward(self, h_bound, h_unbound, **kw):
        features = self.core(h_bound, h_unbound, **kw)
        return self.out_proj(features)


class SymmetricDeltaOnsite(nn.Module):
    """Δ paired readout with symmetric scalar reduction.

    Per (mul, ℓ) output before the final Linear:

        ℓ = 0:  h_b^(0) - h_ub^(0)                    (pure linear subtraction)
        ℓ > 0:  (h_b - h_ub) · (h_b + h_ub) per ch.    (= ||h_b||² - ||h_ub||²)

    Antisymmetric under (bound, unbound) swap. No privileged copy: the
    ℓ > 0 contraction is against the symmetric reference h_b + h_ub rather
    than against h_unbound alone (as in WithProjection(DeltaCore)).

    Followed by a Linear projection to out_dim.
    """

    def __init__(self, irreps_in: Irreps, out_dim: int, **kw):
        super().__init__()
        self.output_dim = out_dim
        self._sectors = _build_sectors(irreps_in)
        total_mul = sum(m for m, _ in irreps_in)
        self.out_proj = nn.Linear(total_mul, out_dim)

    def forward(self, h_bound, h_unbound, **kw):
        delta = h_bound - h_unbound
        symref = h_bound + h_unbound
        parts = []
        for offset, mul, d in self._sectors:
            d_delta = delta[:, offset:offset + mul * d].view(-1, mul, d)
            if d == 1:  # ℓ = 0: pass through pure subtraction
                parts.append(d_delta.squeeze(-1))
            else:        # ℓ > 0: signed magnitude difference per channel
                d_sym = symref[:, offset:offset + mul * d].view(-1, mul, d)
                parts.append((d_delta * d_sym).sum(dim=-1))
        out = torch.cat(parts, dim=-1)
        return self.out_proj(out)


class DeltaNorm2Onsite(nn.Module):
    """Δ paired readout reduced via squared norm: ||h_b − h_ub||² per (channel, ℓ).

    Per (mul, ℓ) output before the final Linear:

        ||(h_b − h_ub)^(c,ℓ)||² = Σ_m ((h_b − h_ub)^(c,ℓ)_m)²

    Symmetric under (bound, unbound) swap (the difference flips sign and the
    square is invariant). Preserves orientation magnitude at ℓ > 0 (unlike
    SymmetricDeltaOnsite which collapses to ||h_b||² − ||h_ub||²). At ℓ = 0
    the output is the squared scalar difference (sign info is lost — unlike
    SymmetricDeltaOnsite which keeps the signed difference at ℓ = 0).

    Followed by a Linear projection to out_dim.
    """

    def __init__(self, irreps_in: Irreps, out_dim: int, **kw):
        super().__init__()
        self.output_dim = out_dim
        self._sectors = _build_sectors(irreps_in)
        total_mul = sum(m for m, _ in irreps_in)
        self.out_proj = nn.Linear(total_mul, out_dim)

    def forward(self, h_bound, h_unbound, **kw):
        delta = h_bound - h_unbound
        parts = []
        for offset, mul, d in self._sectors:
            d_delta = delta[:, offset:offset + mul * d].view(-1, mul, d)
            parts.append((d_delta * d_delta).sum(dim=-1))  # per-channel squared norm
        out = torch.cat(parts, dim=-1)
        return self.out_proj(out)


class OverlapOnsite(nn.Module):
    """Pure SOAP-style cross-overlap, zero learned readout weights.

    For each ℓ, sum across channels and m:

        s^(ℓ) = Σ_c Σ_m h_b^(c,ℓ)_m · h_ub^(c,ℓ)_m

    Output: L_max+1 scalars per node (one per ℓ in irreps_in). Manifestly
    symmetric under (bound, unbound) swap. No privileged copy, no learned
    weights in the readout — the inductive bias is exactly the SOAP kernel.
    """

    def __init__(self, irreps_in: Irreps, out_dim: int = None, **kw):
        super().__init__()
        self._sectors = _build_sectors(irreps_in)
        # Group sectors by ℓ — one scalar per distinct ℓ
        ells = sorted({d for _, _, d in self._sectors})  # d = 2ℓ+1
        self._ell_dims = ells
        self.output_dim = len(ells)
        if out_dim is not None and out_dim != self.output_dim:
            warnings.warn(
                f"OverlapOnsite: out_dim={out_dim} ignored; using {self.output_dim} "
                f"(one scalar per ℓ, zero learned weights).",
                stacklevel=2,
            )

    def forward(self, h_bound, h_unbound, **kw):
        # Group sectors by ℓ (i.e. by d = 2ℓ+1).
        n_nodes = h_bound.shape[0]
        by_ell = {d: torch.zeros(n_nodes, device=h_bound.device, dtype=h_bound.dtype)
                  for d in self._ell_dims}
        for offset, mul, d in self._sectors:
            slab_b = h_bound[:, offset:offset + mul * d].view(-1, mul, d)
            slab_u = h_unbound[:, offset:offset + mul * d].view(-1, mul, d)
            by_ell[d] = by_ell[d] + (slab_b * slab_u).sum(dim=(-1, -2))
        # Concatenate in fixed (increasing ℓ) order.
        return torch.stack([by_ell[d] for d in self._ell_dims], dim=-1)


class PerturbationAttention(nn.Module):
    """Multi-head self-attention on  δh = h_bound^(0) − h_unbound^(0)
    across atoms within each graph.

    Uses ``to_dense_batch`` for efficient batched attention with masking.
    Requires ``batch`` and ``batch_size`` kwargs in forward().

    Only uses l=0 channels.  For lmax > 0, consider equivariant cores
    (delta, linear_combo, cg, gate) wrapped with WithProjection instead.

    Note: Attention is O(N^2) in the number of atoms per graph.  This is
    acceptable for typical 5A-cutoff graphs (20-80 atoms) but may become
    a bottleneck for larger cutoffs or full-protein inputs.
    """

    def __init__(self, irreps_in: Irreps, out_dim: int, num_heads: int = 4, **kw):
        super().__init__()
        n_scalar = sum(m for m, ir in irreps_in if ir.l == 0)
        has_nonscalar = any(ir.l > 0 for _, ir in irreps_in)
        if has_nonscalar:
            warnings.warn(
                f"PerturbationAttention only uses l=0 channels ({n_scalar} scalars) "
                f"from irreps_in={irreps_in}. Higher-l features are discarded. "
                "Consider an equivariant core (cg, gate) + WithProjection for lmax > 0.",
                stacklevel=2,
            )
        self.n_scalar = n_scalar
        self.num_heads = num_heads
        attn_dim = (out_dim // num_heads) * num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.delta_proj = nn.Linear(n_scalar, attn_dim)
        self.q_proj = nn.Linear(attn_dim, attn_dim)
        self.k_proj = nn.Linear(attn_dim, attn_dim)
        self.v_proj = nn.Linear(attn_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, out_dim)
        self.output_dim = out_dim

    def forward(self, h_bound, h_unbound, batch=None, batch_size=None, **kw):
        delta = h_bound[:, :self.n_scalar] - h_unbound[:, :self.n_scalar]
        delta = self.delta_proj(delta)

        x_dense, mask = to_dense_batch(delta, batch, batch_size=batch_size)
        B, max_N, D = x_dense.shape
        H = self.num_heads

        Q = self.q_proj(x_dense).view(B, max_N, H, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_dense).view(B, max_N, H, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_dense).view(B, max_N, H, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        key_mask = mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(~key_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.masked_fill(~key_mask, 0.0)

        out = (attn @ V).transpose(1, 2).reshape(B, max_N, -1)
        out = self.out_proj(out)
        return out[mask]


# ═══════════════════════════════════════════════════════════════════════════
#  ReembedCGOnsite — scalar MP → l>0 re-embedding → CG onsite
# ═══════════════════════════════════════════════════════════════════════════

class ReembedCGOnsite(nn.Module):
    """Re-embed scalar features into l>0 via one equivariant aggregation,
    then CG-couple bound and unbound, contract to scalars.

    Implements Eq. 7 from the paper::

        h_i^(l) = Σ_j R_l(r_ij) Y_l^m(r̂_ij) h_j^(T,0)

    This is mathematically one layer of equivariant message passing applied
    *after* scalar MP completes.  The resulting l>0 features are CG-coupled
    between bound and unbound copies and contracted to scalars via dot product.

    Requires kwargs in forward(): ``pos``, ``bound_edge_index``,
    ``unbound_edge_index``.
    """

    def __init__(self, irreps_in: Irreps, out_dim: int,
                 reembed_lmax: int = 1, envelope_cutoff: float = 5.0,
                 r_max: float = 5.0,
                 num_radial_basis: int = 14, radial_hidden: int = 64,
                 cg_mode: str = "uvu",
                 reembed_attention: bool = False,
                 contraction_refs: str = "unbound",
                 reembed_iterations: int = 1,
                 dual_cg: bool = False,
                 degree2_invariants: bool = False,
                 reembed_proj_dim: int = 0,
                 out_proj_bias: bool = True,
                 **kw):
        super().__init__()
        self.output_dim = out_dim
        self.n_scalar = sum(m for m, ir in irreps_in if ir.l == 0)
        self.reembed_lmax = reembed_lmax
        self.envelope_cutoff = envelope_cutoff
        self.r_max = r_max
        self.num_radial_basis = num_radial_basis

        has_nonscalar = any(ir.l > 0 for _, ir in irreps_in)
        if has_nonscalar:
            warnings.warn(
                f"ReembedCGOnsite only uses l=0 channels ({self.n_scalar} scalars) "
                f"from irreps_in={irreps_in} as the base for re-embedding.",
                stacklevel=2,
            )

        # Spherical harmonics for re-embedding
        self.irreps_sh = Irreps.spherical_harmonics(reembed_lmax)
        self.sh = SphericalHarmonics(
            self.irreps_sh, normalize=True, normalization="component"
        )

        # Project scalars to a smaller dim before re-embedding
        self.proj_dim = reembed_proj_dim if reembed_proj_dim > 0 else min(self.n_scalar, 64)
        self.scalar_proj = nn.Linear(self.n_scalar, self.proj_dim)

        # Direction 5: Attention-weighted re-embedding
        self.reembed_attention = reembed_attention
        if reembed_attention:
            # proj_dim (src) + proj_dim (dst) + 1 (distance) → 1 (logit)
            self.attn_mlp = nn.Sequential(
                nn.Linear(self.proj_dim * 2 + 1, self.proj_dim),
                nn.SiLU(),
                nn.Linear(self.proj_dim, 1),
            )

        # Re-embedded irreps: proj_dim copies of each l
        reembed_parts = []
        for _, ir in self.irreps_sh:
            reembed_parts.append((self.proj_dim, ir))
        self.reembed_irreps = Irreps(reembed_parts)

        # Radial MLP: RBF → one weight per l
        self.radial = nn.Sequential(
            nn.Linear(num_radial_basis, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, reembed_lmax + 1, bias=False),
        )
        nn.init.uniform_(self.radial[-1].weight, -1e-3, 1e-3)

        # CG product between bound and unbound re-embedded features
        instructions = _build_self_instructions(self.reembed_irreps, mode=cg_mode)
        assert instructions, (
            f"No valid {cg_mode} CG paths for {self.reembed_irreps}. "
            f"For uuu, all sectors must have equal multiplicity."
        )
        self.cg = TensorProduct(
            self.reembed_irreps, self.reembed_irreps,
            self.reembed_irreps, instructions
        )

        # Direction 2: Multi-reference contraction
        self.contraction_refs = contraction_refs
        n_refs = 3 if contraction_refs == "multi" else 1
        self._sectors = _build_sectors(self.reembed_irreps)
        total_mul = sum(m for m, _ in self.reembed_irreps)

        # T-series: Dual CG — add self-couplings rb⊗rb, ru⊗ru alongside rb⊗ru
        self.dual_cg = dual_cg
        n_cg_terms = 3 if dual_cg else 1
        total_dots_dim = total_mul * n_refs * n_cg_terms

        # Degree-2 invariants: pass l=0 of CG output directly to readout,
        # bypassing contraction. The l=0 channels of CG(rb, ru) are already
        # rotationally invariant (degree-2, with learned CG weights). Contraction
        # multiplies them by a reference, making them degree-3 — this loses the
        # pure alignment signal. Passing them through uncontracted preserves it.
        self.degree2_invariants = degree2_invariants
        self._n_l0 = sum(m for m, ir in self.reembed_irreps if ir.l == 0)
        if degree2_invariants:
            total_dots_dim += self._n_l0 * n_cg_terms  # l=0 of each CG term

        # Direction 3: Iterative re-embedding
        self.reembed_iterations = reembed_iterations
        if reembed_iterations > 1:
            self.refine_projs = nn.ModuleList([
                nn.Linear(total_dots_dim, self.proj_dim)
                for _ in range(reembed_iterations - 1)
            ])
            self.out_proj = nn.Linear(total_dots_dim * reembed_iterations, out_dim, bias=out_proj_bias)
        else:
            self.out_proj = nn.Linear(total_dots_dim, out_dim, bias=out_proj_bias)

    def _reembed(self, h_scalar, pos, edge_index, num_nodes):
        """One-shot equivariant aggregation: scalars → l>0 irreps.

        If reembed_attention is enabled, learns per-edge attention scores
        applied before scatter aggregation (shared across all l channels).
        """
        src, dst = edge_index
        rel = pos[dst] - pos[src]
        dist = rel.norm(dim=-1)

        # Spherical harmonics
        sh_values = self.sh(rel)  # [E, (lmax+1)^2]

        # Polynomial envelope (skip if envelope_cutoff=0)
        envelope = _polynomial_envelope(dist, self.envelope_cutoff) if self.envelope_cutoff else 1.0  # [E]

        # Radial basis → radial weights per l
        bin_width = self.r_max / max(1, self.num_radial_basis - 1)
        rbf = soft_one_hot_linspace(
            dist, 0.0, self.r_max + bin_width,
            self.num_radial_basis, basis="smooth_finite", cutoff=True
        )
        rbf = rbf / rbf.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        radial_w = self.radial(rbf)  # [E, lmax+1]

        # Project source scalars
        h_proj = self.scalar_proj(h_scalar)  # [N, proj_dim]

        # Direction 5: Attention weights (shared across all l)
        if self.reembed_attention:
            attn_input = torch.cat([h_proj[src], h_proj[dst], dist.unsqueeze(-1)], dim=-1)
            attn_logits = self.attn_mlp(attn_input).squeeze(-1)  # [E]
            attn_w = scatter_softmax(attn_logits, dst, dim=0, dim_size=num_nodes)  # [E]
        else:
            attn_w = None

        # Build re-embedded features per l
        offset_sh = 0
        parts = []
        for l_idx, (_, ir) in enumerate(self.irreps_sh):
            d = 2 * ir.l + 1
            Y_l = sh_values[:, offset_sh:offset_sh + d]  # [E, d]
            R_l = radial_w[:, l_idx]                       # [E]

            # msg[e, c, m] = h_proj[src[e], c] * R_l[e] * envelope[e] * Y_l[e, m]
            weight = (R_l * envelope).unsqueeze(-1) * Y_l         # [E, d]
            if attn_w is not None:
                weight = weight * attn_w.unsqueeze(-1)             # [E, d]
            msg = h_proj[src].unsqueeze(-1) * weight.unsqueeze(-2) # [E, proj_dim, d]
            msg_flat = msg.reshape(msg.size(0), -1)                # [E, proj_dim * d]

            agg = scatter(msg_flat, dst, dim=0, dim_size=num_nodes, reduce='sum')
            parts.append(agg)
            offset_sh += d

        return torch.cat(parts, dim=-1)  # [N, proj_dim * (lmax+1)^2]

    def _contract(self, coupled, rb, ru):
        """Invariant contraction, respecting contraction_refs setting."""
        if self.contraction_refs == "multi":
            dots_ru = _invariant_contract(coupled, ru, self._sectors)
            dots_rb = _invariant_contract(coupled, rb, self._sectors)
            dots_delta = _invariant_contract(coupled, rb - ru, self._sectors)
            return torch.cat([dots_ru, dots_rb, dots_delta], dim=-1)
        else:
            return _invariant_contract(coupled, ru, self._sectors)

    def forward(self, h_bound, h_unbound, pos=None, pos_unbound=None,
                bound_edge_index=None, unbound_edge_index=None, **kw):
        assert pos is not None, "ReembedCGOnsite requires 'pos' kwarg"
        assert bound_edge_index is not None, (
            "ReembedCGOnsite requires 'bound_edge_index' kwarg"
        )
        assert unbound_edge_index is not None, (
            "ReembedCGOnsite requires 'unbound_edge_index' kwarg"
        )

        num_nodes = h_bound.size(0)
        sb = h_bound[:, :self.n_scalar]
        su = h_unbound[:, :self.n_scalar]

        all_dots = []
        for it in range(self.reembed_iterations):
            # Re-embed both copies into l>0 irreps
            pos_ub = pos_unbound if pos_unbound is not None else pos
            rb = self._reembed(sb, pos, bound_edge_index, num_nodes)
            ru = self._reembed(su, pos_ub, unbound_edge_index, num_nodes)

            # CG couple (cross-coupling, plus self-couplings if dual_cg)
            coupled = self.cg(rb, ru)
            dots = self._contract(coupled, rb, ru)
            # Pass l=0 of CG output uncontracted (degree-2 invariants)
            if self.degree2_invariants:
                dots = torch.cat([dots, coupled[:, :self._n_l0]], dim=-1)
            if self.dual_cg:
                coupled_bb = self.cg(rb, rb)
                coupled_uu = self.cg(ru, ru)
                dots_bb = self._contract(coupled_bb, rb, ru)
                dots_uu = self._contract(coupled_uu, rb, ru)
                dots = torch.cat([dots, dots_bb, dots_uu], dim=-1)
                if self.degree2_invariants:
                    dots = torch.cat([dots, coupled_bb[:, :self._n_l0],
                                           coupled_uu[:, :self._n_l0]], dim=-1)
            all_dots.append(dots)

            # Refine bound scalars for next iteration (unbound stays fixed)
            if it < self.reembed_iterations - 1:
                delta = self.refine_projs[it](dots)  # [N, proj_dim]
                sb = sb.clone()
                sb[:, :self.proj_dim] = sb[:, :self.proj_dim] + delta

        if self.reembed_iterations > 1:
            return self.out_proj(torch.cat(all_dots, dim=-1))
        else:
            return self.out_proj(dots)


# ═══════════════════════════════════════════════════════════════════════════
#  WithProjection — core → scalars via dot-product contraction
# ═══════════════════════════════════════════════════════════════════════════

class WithProjection(nn.Module):
    """Wrap an equivariant core to produce scalars.

    Projects via invariant dot-product contraction against a reference:

        l = 0:  features · ref           (signed element-wise product)
        l > 0:  Σ_m f^(l)_m · ref^(l)_m  (alignment score, signed)

    This is strictly more informative than taking norms, which discard
    the orientation of the perturbation.

    If the core's output_irreps differ from ref_irreps, a Linear projects
    the reference to match.

    Parameters
    ----------
    ref_source : {"unbound", "bound"}
        Which representation to use as the dot-product reference.
        Default "unbound" contracts against the isolated-molecule features.
    """

    def __init__(self, core: nn.Module, out_dim: int, ref_irreps: Irreps = None,
                 ref_source: str = "unbound"):
        super().__init__()
        assert ref_source in ("unbound", "bound"), f"ref_source must be 'unbound' or 'bound', got '{ref_source}'"
        self.core = core
        self.output_dim = out_dim
        self.ref_source = ref_source

        out_irreps = core.output_irreps
        if ref_irreps is not None and ref_irreps != out_irreps:
            self.ref_proj = Linear(ref_irreps, out_irreps)
        else:
            self.ref_proj = None

        self._sectors = _build_sectors(out_irreps)
        total_mul = sum(m for m, _ in out_irreps)
        self.out_proj = nn.Linear(total_mul, out_dim)

    def forward(self, h_bound, h_unbound, **kw):
        features = self.core(h_bound, h_unbound, **kw)
        raw_ref = h_unbound if self.ref_source == "unbound" else h_bound
        ref = raw_ref if self.ref_proj is None else self.ref_proj(raw_ref)
        dots = _invariant_contract(features, ref, self._sectors)
        return self.out_proj(dots)


# ═══════════════════════════════════════════════════════════════════════════
#  Composite — combine multiple onsite modules
# ═══════════════════════════════════════════════════════════════════════════

class CompositeOnsite(nn.Module):
    """Run multiple onsite modules and concatenate their scalar outputs.

    Config example::

        onsite_mode: "delta,mlp"   # two modules, each outputting onsite_dim scalars
        onsite_dim: 64             # per-module output dim

    Total output_dim = sum of individual output_dims.
    """

    def __init__(self, modules: list):
        super().__init__()
        self.mods = nn.ModuleList(modules)
        self.output_dim = sum(m.output_dim for m in modules)

    def forward(self, h_bound, h_unbound, **kw):
        return torch.cat([m(h_bound, h_unbound, **kw) for m in self.mods], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
#  ResidueOnsite — pool atoms→residues, compare bound vs unbound
# ═══════════════════════════════════════════════════════════════════════════

class ResidueOnsite(nn.Module):
    """Compare bound vs unbound scalar features pooled at residue granularity.

    Atoms are pooled into residues (via ``residue_id``), then bound and
    unbound residue representations are compared and aggregated to a
    graph-level vector.

    The ligand is treated as one "residue" (residue_id=0).  Pocket atoms
    share residue IDs from the PDB structure.

    Parameters
    ----------
    n_channels_in : int
        Number of input scalar channels per atom.
    out_dim : int
        Output dimension per graph.
    agg : {"mean", "sum"}
        Atom → residue aggregation.
    compare : {"delta", "delta_mlp", "concat_mlp", "reembed_cg"}
        How to compare bound vs unbound residue features.
        "reembed_cg" lifts residue scalars into l>0 irreps using centroid
        positions, then compares via CG product.
    reembed_lmax : int
        Max angular order for reembed_cg mode.
    reembed_proj_dim : int
        Projection dim for reembed_cg (0 = default min(n_channels_in, 64)).
    """

    def __init__(self, n_channels_in: int, out_dim: int = 64,
                 agg: str = "mean", compare: str = "delta_mlp",
                 reembed_lmax: int = 1, reembed_proj_dim: int = 0):
        super().__init__()
        self.agg = agg
        self.compare = compare
        self._output_dim = out_dim

        if compare == "delta":
            self.proj = nn.Linear(n_channels_in, out_dim)
        elif compare == "delta_mlp":
            self.mlp = nn.Sequential(
                nn.Linear(n_channels_in, out_dim * 2),
                nn.SiLU(),
                nn.Linear(out_dim * 2, out_dim),
            )
        elif compare == "concat_mlp":
            self.mlp = nn.Sequential(
                nn.Linear(2 * n_channels_in, out_dim * 2),
                nn.SiLU(),
                nn.Linear(out_dim * 2, out_dim),
            )
        elif compare == "reembed_cg":
            # Build a ReembedCGOnsite for residue-level features
            from e3nn.o3 import Irreps
            residue_irreps = Irreps(f"{n_channels_in}x0e")
            self._reembed = ReembedCGOnsite(
                irreps_in=residue_irreps,
                out_dim=out_dim,
                reembed_lmax=reembed_lmax,
                cg_mode="uuu",
                r_max=15.0,  # residue centroids can be farther apart
                envelope_cutoff=15.0,
                num_radial_basis=14,
                reembed_proj_dim=reembed_proj_dim,
            )
            self._output_dim = self._reembed.output_dim
        else:
            raise ValueError(f"Unknown compare mode '{compare}'")

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, h_bound_scalars, h_unbound_scalars, residue_id,
                batch_original, batch_size, pos=None):
        """
        Parameters
        ----------
        h_bound_scalars : [N_total, C]  scalar features from bound copy
        h_unbound_scalars : [N_total, C]  scalar features from unbound copy
        residue_id : [N_total]  per-atom residue assignment (0=ligand, 1..R=pocket residues)
        batch_original : [N_total]  graph index per original node
        batch_size : int  number of graphs in batch
        pos : [N_total, 3]  atom positions (needed for reembed_cg mode)
        """
        device = residue_id.device

        # 1. Make residue_id unique across batch
        max_res_per_graph = scatter(residue_id, batch_original, dim=0,
                                    dim_size=batch_size, reduce='max')  # [batch_size]
        cumsum = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        cumsum[1:] = torch.cumsum(max_res_per_graph + 1, dim=0)
        global_residue_id = residue_id + cumsum[batch_original]
        num_residues = int(cumsum[-1].item())

        # 2. Pool atoms → residues
        h_b_res = scatter(h_bound_scalars, global_residue_id, dim=0,
                          dim_size=num_residues, reduce=self.agg)
        h_u_res = scatter(h_unbound_scalars, global_residue_id, dim=0,
                          dim_size=num_residues, reduce=self.agg)

        # 3. Compare bound vs unbound at residue level
        if self.compare == "delta":
            compared = self.proj(h_b_res - h_u_res)
        elif self.compare == "delta_mlp":
            compared = self.mlp(h_b_res - h_u_res)
        elif self.compare == "concat_mlp":
            compared = self.mlp(torch.cat([h_b_res, h_u_res], dim=-1))
        elif self.compare == "reembed_cg":
            # Pool positions to residue centroids
            pos_res = scatter(pos, global_residue_id, dim=0,
                              dim_size=num_residues, reduce='mean')
            # Build residue-level edges (all pairs within cutoff)
            from torch_geometric.nn import radius_graph
            batch_residue = torch.zeros(num_residues, dtype=torch.long, device=device)
            batch_residue.scatter_(0, global_residue_id, batch_original)
            edge_index = radius_graph(pos_res, r=self._reembed.r_max,
                                      batch=batch_residue, max_num_neighbors=64)
            # Run ReembedCGOnsite on residue-level data
            return self._reembed(
                h_b_res, h_u_res,
                pos=pos_res,
                bound_edge_index=edge_index,
                unbound_edge_index=edge_index,
            )

        # 4. Build batch index for residues
        batch_residue = torch.zeros(num_residues, dtype=torch.long, device=device)
        batch_residue[global_residue_id] = batch_original

        # 5. Pool residues → graph
        return scatter(compared, batch_residue, dim=0,
                       dim_size=batch_size, reduce='sum')


# ═══════════════════════════════════════════════════════════════════════════
#  Factories
# ═══════════════════════════════════════════════════════════════════════════

_CORE_REGISTRY = {
    "delta":        lambda ir, **kw: DeltaCore(ir),
    "linear_combo": lambda ir, **kw: LinearComboCore(ir),
    "cg":           lambda ir, **kw: CGProductCore(ir, cg_mode=kw.get('cg_mode', 'uvu')),
    "gate":         lambda ir, **kw: GatingCore(ir),
}

_SCALAR_REGISTRY = {
    "mlp":         MLPOnsite,
    "inv_proj":    InvariantProjection,
    "attention":   PerturbationAttention,
    "reembed_cg":  ReembedCGOnsite,
    "direct_cg":   DirectCGOnsite,
    "delta_sym":   SymmetricDeltaOnsite,
    "delta_norm2": DeltaNorm2Onsite,
    "overlap":     OverlapOnsite,
}


def build_onsite_core(irreps_in: Irreps, mode: str, **kwargs) -> nn.Module:
    """Return an equivariant core (output = irreps_in).

    Available: delta, linear_combo, cg, gate.
    """
    if mode not in _CORE_REGISTRY:
        raise ValueError(
            f"Unknown core mode '{mode}'. Choose from: {list(_CORE_REGISTRY)}"
        )
    return _CORE_REGISTRY[mode](irreps_in, **kwargs)


def _build_single_onsite(irreps_in: Irreps, out_dim: int, mode: str, **kwargs) -> nn.Module:
    """Build a single onsite module (not composite)."""
    if mode == "linear":
        mode = "delta"

    if mode in _CORE_REGISTRY:
        core = build_onsite_core(irreps_in, mode, **kwargs)
        ref_source = kwargs.get("ref_source", "unbound")
        return WithProjection(core, out_dim, ref_irreps=irreps_in, ref_source=ref_source)
    elif mode in _SCALAR_REGISTRY:
        return _SCALAR_REGISTRY[mode](irreps_in, out_dim, **kwargs)
    else:
        all_modes = list(_CORE_REGISTRY) + list(_SCALAR_REGISTRY) + ["linear"]
        raise ValueError(
            f"Unknown onsite mode '{mode}'. Choose from: {all_modes}"
        )


def build_onsite(irreps_in: Irreps, out_dim: int, mode: str, **kwargs) -> nn.Module:
    """Return a scalar-output onsite module for direct use in a model.

    Core modes (delta, linear_combo, cg, gate) are wrapped with
    ``WithProjection`` (dot-product contraction against h_unbound).

    Scalar modes (mlp, inv_proj, attention) are returned directly.

    Composite modes: comma-separated list (e.g. ``"delta,mlp"``) builds
    each sub-module with ``out_dim`` and concatenates outputs via
    ``CompositeOnsite``.  Total output_dim = N * out_dim.

    Legacy alias: ``"linear"`` → DeltaCore + WithProjection.

    All returned modules accept
    ``forward(h_bound, h_unbound, batch=None, batch_size=None, **kw)``
    and have an ``.output_dim`` attribute.
    """
    parts = [m.strip() for m in mode.split(",")]
    if len(parts) == 1:
        return _build_single_onsite(irreps_in, out_dim, parts[0], **kwargs)
    else:
        modules = [_build_single_onsite(irreps_in, out_dim, m, **kwargs) for m in parts]
        return CompositeOnsite(modules)
