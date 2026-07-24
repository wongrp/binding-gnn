# Q-series: Equivariant Angular Resolution Ablation

## Theme
The paper defines 5 angular resolution configurations (A-E). The existing
9-model ablation ladder covers only A (scalar everything) and D (scalar MP,
equivariant on-site via reembed). Config E is degenerate (higher-l SH with
scalar TP = Config A). Q-series fills the remaining gap: **Config B**
(equivariant MP + equivariant on-site) and **Config C** (equivariant MP +
scalar on-site).

Motivation: the apo forward-pass comparison showed all 9 scalar models are
insensitive to coordinate perturbations (hybrid |delta| ~0.001 pK) but
brittle to topology changes (~0.5 pK median). Equivariant models propagate
directional information through l>0 channels -- they *might* show more
coordinate sensitivity. This is the control experiment needed before
concluding "coordinates don't matter."

## Design
Each Q config is a minimal edit of a scalar ladder model: change ONLY
irreps_hidden (384x0e -> 128x0e + 14x1o), lmax (0 -> 1), scalars (192 -> 128),
and onsite_mode (reembed_cg -> cg for Config B, or mlp for Config C).
All other settings (edges, persist, feedback, virtual, training) are identical
to the scaffold.

## Scaffolds (5 mechanism levels x 2 angular configs = 8 configs)
- b2 (baseline, no on-site): q1 only (B=C degenerate)
- c7 (node + edge on-site): q2 (B), q3 (C)
- d2 (persist + feedback): q4 (B), q5 (C)
- d3 (persist, edge-only): q6 only (B=C degenerate)
- l4 (persist + fb + virtual): q7 (B), q8 (C)

## Prior equivariant attempts
- b1 (Config B, cg uvu): RMSE 1.42 -- bad training settings (no LR decay, wd=1e-6)
- d12 (Config B, reembed_cg + edge): RMSE 1.36 -- only 3 layers
- c22 (planned Config C): never completed (no checkpoint)
Q-series fixes these: standard training, 5 layers, proper wd.

## Key questions
1. Does equivariant MP increase hybrid-mode |delta| (coordinate sensitivity)?
2. Does Config B (CG on-site) vs Config C (MLP on-site) matter when MP is equivariant?
3. Does equivariant MP help or hurt with persist/feedback/virtual mechanisms?
