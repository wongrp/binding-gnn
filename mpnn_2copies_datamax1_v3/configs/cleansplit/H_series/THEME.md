# H-series: Edge Readout

**Theme**: Test direct edge state readout — pooling raw persistent edge states
into the prediction head — and edge-only readout (no node pooling at all).

**New model flags**:
- `edge_readout: true` — scatter_sum bound edge states into head
- `node_readout: false` — skip node pooling entirely

**Motivation**: The current readout is dominated by node pooling (sum over all
atoms, most far from the interface). Edge states encode inter-molecular contact
geometry directly — binding affinity IS about interface contacts. Two questions:
1. Does adding edge readout help on top of existing node readout? (h1-h3)
2. Can edge-only readout replace node readout entirely? (h4-h8)

**Configs**:

Additive (edge_readout on top of full model):
- h1: D2 + edge_readout (1.55M)
- h2: D15 (D2 + virtual K=4) + edge_readout (1.64M)
- h3: D18 (D2 + edge_dim=128) + edge_readout (1.72M)

Edge-only (node_readout=false, onsite_mode=none):
- h4: 384x0e 5L, edge_onsite + edge_readout (1.45M)
- h5: 384x0e 5L, pure edge_readout only, head_in=64 (1.26M)
- h6: 256x0e 5L, edge_onsite + edge_readout (836K)
- h7: 256x0e 3L, edge_onsite + edge_readout (593K)
- h8: 384x0e 5L, edge_onsite + edge_readout + virtual node attention (1.54M)

Controls: D2 (1.291), D15 (1.292), D18 (1.288) from D-series.
