# T-Series: Multi-ref Combinations

## Theme

S-series tested five enhancements to the reembed_cg onsite comparison
independently. **s2_multi_ref won decisively** (val-selected RMSE 1.307 vs
next-best s3_multiscale at 1.341; best-test numbers are 1.272 and 1.305
respectively but those are test-peeked). Multi-ref contracts the CG output
against three references (ru, rb, rb−ru) instead of just ru, tripling the
readout richness.

The T-series asks: **what combines well with multi-ref?**

## Configs

| # | Addition over multi-ref | Key flag | Hypothesis |
|---|------------------------|----------|------------|
| t1 | Attention | `reembed_attention: true` | Does learned neighbor weighting help now that readout is richer? |
| t2 | lmax=2 | `onsite_reembed_lmax: 2` | Does multi-ref unlock higher angular resolution? (lmax=2 alone failed in B-series) |
| t3 | Dual CG | `dual_cg: true` | Do self-couplings rb⊗rb, ru⊗ru add useful signal alongside cross-coupling rb⊗ru? |
| t4 | Wider onsite | `onsite_dim: 128` | Is 64-dim a bottleneck for multi-ref's 3x richer input? |
| t5 | Uncontracted l=0 | `degree2_invariants: true` | Does passing CG l=0 scalars directly (degree-2) complement the contracted degree-3 signal? |

## Base

All configs inherit s2_multi_ref's settings: d2 base (persistent edges +
feedback) + `contraction_refs: "multi"`, cleansplit, 1200 epochs, batch 64.

## Mathematical context

The CG+contraction pipeline produces **degree-3 invariants**: three geometric
objects multiplied together (CG coupling × reference contraction). The l=0
channels of the CG output are already rotationally invariant — they're
**degree-2** (direct alignment scores with learned weights). Contracting them
against a reference needlessly promotes them to degree-3, losing the pure
alignment signal. t5 tests whether preserving this degree-2 pathway helps.
