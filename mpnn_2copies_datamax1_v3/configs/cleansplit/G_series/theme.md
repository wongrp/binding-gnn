# G-series: Capacity Rebalancing

## Theme
Shift capacity from the backbone (node hidden dims) into the comparison
machinery (edges, radial MLP, onsite, head). D-series showed that d31's
tiny 192x0e backbone with full comparison modules (772K params) gets the
best oracle RMSE (1.2855), while d2's 384x0e backbone is best val-selected
(1.3058). The backbone may be over-parameterized relative to the modules
that actually compare bound vs unbound states.

## Key D-series lessons
- Persistent edges + feedback is the foundation (d2 = best val-selected)
- Bigger edge_state_dim helps oracle (d18: 128 vs d2: 64)
- Shrinking backbone works (d31: 192x0e, 772K → best oracle 1.2855)
- Val-test gap is the bottleneck, not raw capacity

## Directions
1. **Bigger edges on smaller backbones**: d31-style narrow backbone + d18-style
   edge_state_dim=128. More capacity where the comparison happens.
2. **Richer radial MLP**: radial_hidden=128 (vs 64). The radial MLP controls
   how distance information modulates messages — currently the tightest
   bottleneck in D-series configs.
3. **Deeper narrow**: More layers on narrow backbones. Depth compensates for
   width; larger receptive field without parameter explosion.
4. **Larger comparison modules**: onsite_dim=128, head_hidden=256. Give the
   comparison and readout stages more room.
5. **Combine E/F winners**: Best virtual_pool_streams from E, best LR schedule
   from F, applied to the best G backbone.

## Dependencies
- Wait for E-series (virtual node configs) and F-series (LR schedules) results
- Best LR schedule from F should be applied to all G configs
- Best virtual config from E should be included in g5+ configs
