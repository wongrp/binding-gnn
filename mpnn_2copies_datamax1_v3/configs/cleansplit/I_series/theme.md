# I-series: Alternative Fusion Methods

## Theme
Two levels of fusion:
1. **Comparison-level** (i1, i2): How bound vs unbound edge states are compared
2. **Readout-level** (i3-i5): How multiple readout streams are combined before the prediction head

## Comparison-level (edge onsite alternatives)
- **i1 reembed_cg**: Re-embed scalar edge states into l>0 via SH of edge direction, then CG-couple bound vs unbound. Geometric edge comparison.
- **i2 bilinear**: Hadamard product of projected bound/unbound edge states. Multiplicative interaction captures correlations that delta misses.

## Readout-level (readout fusion)
All three replace simple concatenation of node/edge/virtual readout streams:
- **i3 gated**: Each stream produces a sigmoid gate for the sum of all other streams. Learns which signals to attend to.
- **i4 film**: FiLM (Feature-wise Linear Modulation). Node pool is "main signal"; edge+virtual produce affine transform (γ, β). Asymmetric.
- **i5 self_attn**: Each stream is a token in single-layer multi-head self-attention. Fully symmetric. Mean-pool attended tokens.

## Parent configs
- i1, i2: based on d2 (persist+feedback, best val-selected)
- i3-i5: based on e8 (d2 + virtual K=8 d=32, best oracle)

## Key question
The val-test gap is the bottleneck. These fusion alternatives may help if the gap comes from how streams interact, not just from individual stream quality.
