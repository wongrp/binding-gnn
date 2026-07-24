"""
SE(3)-equivariant TPConv layer for binding affinity prediction.

Nonlinearity options:
1. Self-CG product (h ⊗ h): equivariant bilinear nonlinearity (diagonal or full)
2. Gate activation: SiLU on scalars, sigmoid gates on vectors
3. Node-conditioned radial weights: done at model level, not here

Flow: TP messages → aggregate → self-CG → BatchNorm → gate → residual
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import Irreps, SphericalHarmonics, FullyConnectedTensorProduct, TensorProduct
from e3nn.nn import Gate, BatchNorm
from torch_scatter import scatter
from typing import Literal


def polynomial_envelope(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Smooth polynomial cutoff: 1 at r=0, 0 at r=cutoff, vanishing derivatives at boundary.

    Should typically be disabled (envelope_cutoff=0) when edges are precomputed:
    the radial MLP already learns distance-dependent weights from RBF features,
    and this envelope severely attenuates 3-5Å interactions (VdW, pi-stacking,
    hydrophobic contacts) that are critical for binding affinity prediction.
    """
    x = (dist / cutoff).clamp(max=1.0)
    return (1.0 - x * x) ** 3


def make_radial_mlp(edge_dim: int, hidden: int, num_layers: int, out_dim: int) -> nn.Sequential:
    """Build radial MLP that produces tensor product weights from edge features.

    Final layer initialized near-zero so residual dominates early training.
    """
    layers = [nn.Linear(edge_dim, hidden), nn.SiLU()]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
    final = nn.Linear(hidden, out_dim, bias=False)
    nn.init.uniform_(final.weight, -1e-3, 1e-3)
    layers.append(final)
    return nn.Sequential(*layers)


def _build_diagonal_self_tp(irreps: Irreps) -> TensorProduct:
    """Build diagonal self-TP: only same-ℓ irreps couple (0⊗0, 1⊗1, 2⊗2).

    Uses "uvu" when multiplicities match (e.g. 40x0e ⊗ 40x0e → 40x0e: 40² weights).
    Falls back to "uvw" when they don't (e.g. 12x1o ⊗ 12x1o → 40x0e: 12*12*40 weights).
    """
    instructions = []
    for i, (mul_i, ir_i) in enumerate(irreps):
        for j, (mul_j, ir_j) in enumerate(irreps):
            if ir_i.l != ir_j.l:
                continue
            for k, (mul_k, ir_k) in enumerate(irreps):
                if ir_k in ir_i * ir_j:
                    mode = "uvu" if mul_i == mul_k else "uvw"
                    instructions.append((i, j, k, mode, True))
    return TensorProduct(irreps, irreps, irreps, instructions=instructions, shared_weights=True)


def _build_elementwise_self_tp(irreps: Irreps) -> TensorProduct:
    """Build elementwise self-TP: uuu connection mode (all three indices match).

    O(n_c) weights per path — only paths where mul_in1 == mul_in2 == mul_out
    are included. Much cheaper than diagonal (uvu, O(n_c²)) or full (uvw, O(n_c³)).

    For non-uniform multiplicities (e.g. 128x0e + 14x1o), only sectors with
    matching multiplicity couple. Parity rules further restrict paths.
    """
    instructions = []
    for i, (mul_i, ir_i) in enumerate(irreps):
        for j, (mul_j, ir_j) in enumerate(irreps):
            for k, (mul_k, ir_k) in enumerate(irreps):
                if ir_k in ir_i * ir_j and mul_i == mul_j == mul_k:
                    instructions.append((i, j, k, "uuu", True))
    assert instructions, (
        f"No valid uuu self-CG paths for {irreps}. "
        f"All coupled sectors must have equal multiplicity."
    )
    return TensorProduct(irreps, irreps, irreps, instructions=instructions, shared_weights=True)


def _build_message_tp(irreps_in: Irreps, irreps_sh: Irreps, irreps_out: Irreps,
                      tp_mode: str, sparsity: float = 0.25) -> TensorProduct:
    """Build message tensor product with the specified connection mode.

    "full": FullyConnectedTP (uvw everywhere). Most params.
    "diagonal": all valid CG paths, but "uvu" when mul_in == mul_out. Fewer params.
    "sparse": random subset of paths (uvw). Cheapest.
    """
    if tp_mode == "full":
        return FullyConnectedTensorProduct(
            irreps_in, irreps_sh, irreps_out, shared_weights=False)

    if tp_mode == "diagonal":
        instructions = []
        for i_in, (mul_in, ir_in) in enumerate(irreps_in):
            for i_sh, (mul_sh, ir_sh) in enumerate(irreps_sh):
                for i_out, (mul_out, ir_out) in enumerate(irreps_out):
                    if ir_out in ir_in * ir_sh:
                        mode = "uvu" if mul_in == mul_out else "uvw"
                        instructions.append((i_in, i_sh, i_out, mode, True))
    elif tp_mode == "sparse":
        import random
        instructions = []
        for i_in, (mul_in, ir_in) in enumerate(irreps_in):
            for i_sh, (mul_sh, ir_sh) in enumerate(irreps_sh):
                for i_out, (mul_out, ir_out) in enumerate(irreps_out):
                    if ir_out in ir_in * ir_sh:
                        if random.random() < sparsity:
                            instructions.append((i_in, i_sh, i_out, "uvw", True))
        # Fallback: at least keep scalar → scalar
        if not instructions:
            for i_in, (mul_in, ir_in) in enumerate(irreps_in):
                if ir_in.l == 0:
                    for i_sh, (mul_sh, ir_sh) in enumerate(irreps_sh):
                        for i_out, (mul_out, ir_out) in enumerate(irreps_out):
                            if ir_out.l == 0 and ir_out in ir_in * ir_sh:
                                instructions.append((i_in, i_sh, i_out, "uvw", True))
    else:
        raise ValueError(f"Unknown tp_mode: {tp_mode}")

    return TensorProduct(irreps_in, irreps_sh, irreps_out,
                         instructions=instructions, shared_weights=False)


class TPConv(nn.Module):
    """
    SE(3)-equivariant message passing layer using tensor products.

    Flow: TP messages → aggregate → self-CG → BatchNorm → gate → residual

    Args:
        irreps_in: Input irreps
        irreps_out: Output irreps
        edge_dim: Dimension of edge features (RBF basis)
        lmax: Maximum spherical harmonic degree
        hidden: Hidden dimension in radial MLP
        num_radial_layers: Depth of radial MLP
        aggr: Aggregation method ("mean" or "sum")
        use_norm: Whether to apply equivariant BatchNorm
        self_cg: Self-CG product mode: "none", "elementwise" (uuu), "diagonal" (uvu), or "full" (uvw)
        tp_mode: Message TP mode: "full", "diagonal", or "sparse"
        sparsity: For sparse mode, fraction of paths to include
    """

    def __init__(
        self,
        irreps_in: str,
        irreps_out: str,
        edge_dim: int,
        lmax: int = 2,
        hidden: int = 64,
        num_radial_layers: int = 1,
        aggr: str = "mean",
        use_norm: bool = True,
        self_cg: str = "none",
        tp_mode: Literal["full", "diagonal", "sparse"] = "full",
        sparsity: float = 0.25,
        envelope_cutoff: float = None,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.aggr = aggr
        self.tp_mode = tp_mode
        self.envelope_cutoff = envelope_cutoff

        # Spherical harmonics for encoding relative positions
        self.irreps_sh = Irreps.spherical_harmonics(lmax)
        self.sh = SphericalHarmonics(self.irreps_sh, normalize=True, normalization="component")

        # Message tensor product: input ⊗ spherical_harmonics → output
        self.tp = _build_message_tp(
            self.irreps_in, self.irreps_sh, self.irreps_out, tp_mode, sparsity)

        # Radial network produces TP weights from edge features
        self.radial = make_radial_mlp(edge_dim, hidden, num_radial_layers, self.tp.weight_numel)

        # Residual connection (linear projection if irreps change)
        self.skip = (
            None if self.irreps_in == self.irreps_out
            else o3.Linear(self.irreps_in, self.irreps_out)
        )

        # Self-CG product: equivariant bilinear nonlinearity (h ⊗ h → h)
        if self_cg == "full":
            self.self_cg = FullyConnectedTensorProduct(
                self.irreps_out, self.irreps_out, self.irreps_out, shared_weights=True
            )
        elif self_cg == "diagonal":
            self.self_cg = _build_diagonal_self_tp(self.irreps_out)
        elif self_cg == "elementwise":
            self.self_cg = _build_elementwise_self_tp(self.irreps_out)
        else:
            self.self_cg = None

        # Equivariant BatchNorm
        self.norm = BatchNorm(self.irreps_out) if use_norm else None

        # Gate activation: SiLU on scalars, sigmoid gates multiply vectors
        scalars_out = o3.Irreps([(m, ir) for m, ir in self.irreps_out if ir.l == 0])
        vectors_out = o3.Irreps([(m, ir) for m, ir in self.irreps_out if ir.l > 0])
        n_gates = sum(m for m, _ in vectors_out)
        gates_ir = o3.Irreps(f"{n_gates}x0e") if n_gates else o3.Irreps("")

        self.gate = Gate(
            scalars_out, [nn.SiLU()] * len(scalars_out),
            gates_ir, [nn.Sigmoid()] * len(gates_ir),
            vectors_out,
        )
        self.pre_gate = o3.Linear(self.irreps_out, self.gate.irreps_in)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        rel = pos[dst] - pos[src]

        # Message passing
        tp_weights = self.radial(edge_attr)
        if self.envelope_cutoff:
            tp_weights = tp_weights * polynomial_envelope(rel.norm(dim=-1), self.envelope_cutoff).unsqueeze(-1)
        messages = self.tp(x[src], self.sh(rel), tp_weights)
        agg = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce=self.aggr)

        # Self-CG: equivariant nonlinearity
        if self.self_cg is not None:
            agg = agg + self.self_cg(agg, agg)

        # BatchNorm
        if self.norm is not None:
            agg = self.norm(agg)

        # Gate
        agg = self.pre_gate(agg)
        agg = self.gate(agg)

        # Residual
        if self.skip is not None:
            return self.skip(x) + agg
        return x + agg

    def extra_repr(self):
        return f"tp_mode={self.tp_mode}, weight_numel={self.tp.weight_numel}"
