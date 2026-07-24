# Z Series: Readout Ablations

Parent: u44_dual_scale_sh

Tests whether the raw node pool (512 dims = 80% of head input) is necessary,
whether unbound pooling adds information, and whether node features can be
compressed before pooling.

| Config | node_readout | pool_unbound | pool_delta | node_pool_proj_dim | pool_streams | Description |
|--------|-------------|-------------|-----------|-------------------|-------------|-------------|
| z1 | false | — | — | — | — | No node pool at all. Edge/onsite/SH only. |
| z2 | true | true | false | 0 | lig,pkt | Also pool unbound. 4 streams. |
| z3 | true | false | false | 64 | lig,pkt | Compress bound pool to 64. |
| z4 | true | true | false | 64 | lig,pkt | Both copies pooled, both compressed. |
| z5 | true | false | false | 32 | lig,pkt | Even more compressed (32). |
| z6 | false | — | — | — | lig,pkt | No node pool. (Same as z1 for now.) |
| z7 | true | false | false | 0 | pocket | Pool pocket only. Ligand from edges/SH. |
| z8 | true | false | true | 0 | lig,pkt | Pool (bound - unbound) difference. |
