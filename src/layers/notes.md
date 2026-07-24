# TPConv Design Notes

## Linear can't create l>0 from scalars
`o3.Linear(Nx0e, Mx0e + Kx1o + ...)` zeros out all l>0 channels.
Project to scalars only; first TPConv lifts via `h ⊗ Y(r̂)`.

## Self-CG product as nonlinearity
Without self-CG, the only nonlinearity on node features is the Gate.
Everything between aggregation and gate (linear, norm, linear) composes
into one linear map — stacking them adds no expressiveness.

`h ⊗ h` is bilinear, hence nonlinear, and equivariant:
- Diagonal: same-ℓ only (0⊗0, 1⊗1, 2⊗2). Cheap.
- Full: all pairs. More expressive, more params.

Flag: `self_cg="none" | "diagonal" | "full"`.

Applied as residual: `agg + self_cg(agg, agg)`. Gradient of the
quadratic term scales as O(||agg||) — mitigated by BatchNorm after.

## No self-loops
`Y(r̂)` undefined at r=0. Residual + self-CG handle self-interaction.

## Radial MLP final layer
- `bias=False` so `MLP(0) = 0` — messages vanish exactly at cutoff.
- Small init `[-1e-3, 1e-3]` so residual dominates early.
- This removes the need for a separate cutoff envelope in TPConv:
  if RBF→0 at cutoff and MLP(0)=0, there's no discontinuity.

## Gate constructor
- One activation per irrep group: `[nn.SiLU()] * len(scalars_out)`.
- `pre_gate` targets `self.gate.irreps_in` (single source of truth).

## Diagonal self-TP: connection modes matter
Output channels are unrestricted (1o⊗1o → 0e, 2e is valid CG).
The problem was "uvw" on 0e⊗0e→0e: 40³ = 64K weights.
Fix: "uvu" when `mul_in == mul_out`, "uvw" fallback otherwise.

For `40x0e + 12x1o + 6x2e`:
- 0e⊗0e→0e: uvu, 40² = 1,600
- 1o⊗1o→0e: uvw, 12·12·40 = 5,760
- 1o⊗1o→2e: uvw, 12·12·6 = 864
- 2e⊗2e→0e: uvw, 6·6·40 = 1,440
- 2e⊗2e→2e: uvu, 6² = 36
- Total: 9,700 (vs 88K full, 9.1× reduction)

## FullyConnectedTP param explosion
`FCTP(128x0e, 1x0e, 128x0e)` = 128² = 16K weights per edge.
Radial MLP must output all of them. For scalar-only models,
SchNet-style `h_j * W(r)` needs only 128 weights.

## Node-conditioned radial weights (model-level, not here)
Concat scalar node features with RBF:
```
edge_attr = cat([RBF(r_ij), h_i_scalars, h_j_scalars])
```
Makes TP weights node-dependent. Done at model level since
it requires knowing which channels are scalars.
