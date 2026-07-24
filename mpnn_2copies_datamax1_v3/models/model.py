"""
Unified equivariant MPNN for binding affinity prediction.

Handles both 1-copy and 2-copy (onsite interaction) modes via config:
- 1-copy: standard MPNN, pool all nodes together
- 2-copy: bound copy sees intra+inter edges, unbound sees intra only;
  onsite modules compare bound vs unbound features (see onsite.py)

Pooling streams (pool_streams config):
- ["all"]:                     pool all nodes together (default)
- ["ligand", "pocket"]:        separate ligand/pocket pooling, concat
- ["all", "ligand", "pocket"]: all three streams concatenated

Previously split across mpnn_datamax1_v2 and mpnn_2copies_datamax1_v2.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps, Linear
from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter, scatter_mean, scatter_sum
from torch_geometric.utils import to_dense_batch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layers.tpconv import TPConv
from onsite import build_onsite, ResidueOnsite
from edge_onsite import build_edge_onsite, build_persistent_edge_onsite
from virtual_onsite import VirtualNodeOnsite
from readout_fusion import build_readout_fusion
from global_sh import GlobalSHPooling
from msr import MultiShellRadial


class GINLayer(nn.Module):
    """GIN (Graph Isomorphism Network) layer — topology-only message passing.

    h_i = MLP((1 + eps) * h_i + sum_j h_j)

    No distance information, no edge features. Geometry enters only
    through the post-MP comparison mechanisms (onsite, SH).
    """

    def __init__(self, in_dim, out_dim, eps=0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, pos, edge_index, edge_attr=None):
        # pos and edge_attr are ignored — topology only
        src, dst = edge_index
        agg = scatter_sum(x[src], dst, dim=0, dim_size=x.size(0))
        out = self.mlp((1 + self.eps) * x + agg)
        return out + self.residual(x)


@dataclass
class ModelConfig:
    # Architecture
    irreps_in: str = "40x0e"
    irreps_hidden: str = "64x0e + 16x1o + 8x2e"
    num_layers: int = 5
    scalars: int = 64
    lmax: int = 2

    # Backbone type: "tpconv" (default, e3nn tensor product) or "gin" (topology-only)
    backbone: str = "tpconv"

    # Envelope cutoff for polynomial attenuation on TP weights.
    # 0 = disabled. Should be 0 when edges are precomputed: the envelope's
    # purpose (smoothing the hard cutoff boundary) is unnecessary with fixed
    # edges, and it severely attenuates physically important 3-5Å interactions.
    # The radial MLP can learn any distance weighting from the RBF features.
    envelope_cutoff: float = 0.0

    # r_max: controls RBF bin spacing (soft_one_hot_linspace range).
    # Should match the data preprocessing cutoff (typically 5.0Å).
    r_max: float = 5.0
    num_basis: int = 14
    use_precomputed_edges: bool = True
    precomputed_edge_dim: int = 21

    # Knob 2: Cross-object edge type embeddings
    cross_object_embed_dim: int = 0

    # Knob 5: Destination features in message
    use_dst_features: bool = False

    # Knob 6: Input atom type embeddings
    num_atom_types: int = 17
    input_atom_embed_dim: int = 0

    # Amino acid type embedding (pocket atoms only, ligand = padding index 0)
    aa_embed_dim: int = 0            # 0 = disabled, >0 = learned embedding dim
    num_aa_types: int = 22           # 20 standard AA + UNK + padding(0)

    # Residue-level input: separate projection for residue nodes (different feature space)
    # When >0, residue/pocket nodes use x_physchem (this dim) instead of x (atom features)
    # and get their own input projection to the same hidden dim. Ligand atoms use the standard path.
    residue_input_dim: int = 0       # 0 = disabled (all nodes share input_proj), 27 = physicochemical

    # Radial MLP
    radial_hidden: int = 128
    radial_layers: int = 2

    # TPConv options
    tp_mode: str = "full"
    tp_sparsity: float = 0.25
    self_cg: str = "none"
    use_norm: bool = True

    # Node-conditioned radial: concat current scalar hidden state per layer
    node_conditioned_radial: bool = True

    # Virtual node in MP: per-layer global aggregation + broadcast (scalar only)
    # Separate virtual nodes for bound and unbound copies (no cross-contamination)
    virtual_node_mp: bool = False

    # Context injection: broadcast a fixed embedding to all nodes before MP
    # Used for ATOMICA graph-level embeddings as global context
    context_embed_dim: int = 0       # input dim of context embedding (0 = disabled)
    context_proj_dim: int = 0        # project to this dim before adding (0 = match n_scalar_channels)

    # Per-residue injection: add per-atom embeddings from a protein language model
    # Each pocket atom gets its parent residue's embedding via data.residue_id
    residue_lm_dim: int = 0          # input dim of per-residue LM embedding (0 = disabled)

    # Asymmetric angular resolution: zero l>0 channels on pocket atoms after each MP layer.
    # Only effective when lmax > 0 and use_2copies = True.
    pocket_scalar_only: bool = False

    # Aggregation
    aggr: str = "sum"
    pool_mode: str = "sum"
    node_readout: bool = True           # pool nodes into readout (disable for edge-only)
    use_2copies: bool = True             # model expects 2-copy (bound+unbound) input
    pool_unbound: bool = False          # also pool unbound node features (doubles node pool streams)
    pool_delta: bool = False            # pool (bound - unbound) instead of raw bound
    node_pool_proj_dim: int = 0         # project node scalars before pooling (0 = no projection, use full scalars)
    pool_attention: bool = False         # gated attention pooling: learn per-node importance weights
    onsite_self_attn: bool = False       # self-attention over per-atom onsite features before pooling
    onsite_self_attn_heads: int = 4      # number of attention heads

    # Head
    head_hidden: int = 128
    head_layers: int = 2
    dropout: float = 0.0
    output_clamp: float = 0.0        # clamp predictions to [-c, c] (0 = disabled). 16.0 recommended.
    head_norm: str = "none"           # "none", "batch" (BatchNorm1d), or "layer" (LayerNorm) before head MLP

    # ESM readout fusion (precomputed per-complex embeddings)
    esm_readout_dim: int = 0        # project ESM pool to this dim before head (0 = disabled)
    esm_input_dim: int = 1280       # ESM-2 embedding dimension

    # Feature layout for GIGN
    num_scalar_features: int = 23

    # Feature masking for CORDIAL-like ablation
    # "none" = all features, "no_atom_types" = zero atom type one-hot,
    # "topology_only" = keep only degree + valence, "geometry_only" = all zeros
    feature_mask: str = "none"
    # Which nodes to apply the mask to: "both", "ligand", "pocket"
    feature_mask_target: str = "both"

    # Onsite interactions (2-copy mode)
    # Core modes (equivariant, wrapped with WithProjection):
    #   "linear" (alias: delta), "linear_combo", "cg", "gate"
    # Scalar modes (direct scalar output):
    #   "mlp", "inv_proj", "attention"
    # Composite: comma-separated, e.g. "delta,mlp"
    # "none" disables onsite
    onsite_mode: str = "none"
    onsite_dim: int = 64           # per-module output dim (total may differ for composite)

    # Re-embedding lmax for "reembed_cg" onsite mode (Config D in paper)
    # Only used when onsite_mode contains "reembed_cg"
    onsite_reembed_lmax: int = 0

    # CG connection mode for onsite: "uvu" (O(n_c²) weights) or "uuu" (O(n_c), true diagonal)
    # Applies to "cg" core and "reembed_cg" module
    onsite_cg_mode: str = "uvu"

    # S-series: Advancing geometric re-embedding (reembed_cg enhancements)
    # Direction 5: Learn per-edge attention scores in re-embedding aggregation
    reembed_attention: bool = False
    # Direction 2: Contract CG output against multiple references ("unbound" or "multi")
    contraction_refs: str = "unbound"
    # Direction 3: Iterative reembed→CG→contract rounds (1 = current behavior)
    reembed_iterations: int = 1
    # T-series: Dual CG — add self-couplings rb⊗rb, ru⊗ru alongside rb⊗ru
    dual_cg: bool = False
    # Degree-2 invariants: concatenate raw dot products (rb·ru, rb·rb, ru·ru)
    # alongside CG-based degree-3 invariants. Direct alignment without coupling.
    degree2_invariants: bool = False
    # Re-embedding projection dimension. 0 = default (min(n_scalar, 64)).
    onsite_reembed_proj_dim: int = 0
    # Whether out_proj has a bias term. Setting false removes the constant
    # per-atom bias floor (~0.68 in u44_s42). Tests whether the bias is
    # decorative (absorbed by head) or carries information.
    onsite_out_proj_bias: bool = True
    # Direction 1: Apply reembed_cg at every MP layer, learn per-layer weights
    multiscale_reembed: bool = False
    # Restrict atom on-site contribution to interface atoms (atoms incident to
    # at least one inter-molecular edge). Used to control the sum-domain axis
    # in the node-vs-edge ablation comparison (zc40-style).
    onsite_interface_mask: bool = False

    # Multi-Shell Radial readout (MSR): SH-free bilinear shell counter with K
    # parallel signed radial filters. Default off (fully backwards compatible).
    # See models/msr.py for details.
    msr_pooling: bool = False
    msr_num_filters: int = 8
    msr_n_channels: int = 16
    msr_out_dim: int = 64
    msr_num_rbf: int = 20
    msr_r_max: float = 20.0
    msr_separate_proj: bool = True

    # Jumping Knowledge: concatenate scalar features from ALL layers before projection
    # Captures multi-scale information (layer 1 ≈ 5Å local, layer 5 ≈ 25Å global)
    jk_readout: bool = False

    # Pooling streams: which node subsets to pool separately
    # ["all"] (default), ["ligand", "pocket"], ["all", "ligand", "pocket"]
    pool_streams: List[str] = field(default_factory=lambda: ["all"])

    # Edge onsite: compare bound vs unbound edge features at inter-molecular edges
    # "none" disables (default), "mlp", "delta", "delta,mlp"
    edge_onsite_mode: str = "none"
    edge_onsite_dim: int = 64      # per-module output dim
    edge_hidden_dim: int = 128     # intermediate edge representation

    # Persistent edge states: edges updated every MP layer from endpoint nodes
    persistent_edge_state: bool = False   # enable persistent edge states
    edge_state_dim: int = 64             # persistent edge hidden dim
    edge_feedback: bool = False          # scatter edge state back to nodes
    edge_readout: bool = False           # pool raw edge states into readout
    edge_readout_mode: str = "bound"     # "bound" (pool e_bound), "delta" (pool e_bound - e_unbound)

    # Virtual node onsite (graph-level comparison via K learned attention heads)
    virtual_onsite_mode: str = "none"     # "none" or "attention"
    num_virtual_nodes: int = 4            # K
    virtual_node_dim: int = 64            # d per virtual node
    virtual_dropout: float = 0.0          # attention + MLP dropout in virtual onsite
    # Which atom subsets each VirtualNodeOnsite attends to:
    # ["all"] (default), ["ligand", "pocket"], ["all", "ligand", "pocket"]
    virtual_pool_streams: List[str] = field(default_factory=lambda: ["all"])

    # Virtual edge onsite: K virtual nodes attend to persistent edge states
    virtual_edge_onsite_mode: str = "none"  # "none" or "attention"

    # Edge-level CG re-embedding (for edge_onsite_mode containing "reembed_cg")
    edge_onsite_reembed_lmax: int = 1
    edge_onsite_cg_mode: str = "uuu"

    # Global SH pooling: angular descriptors for readout (2-copy only)
    global_sh_pooling: bool = False
    global_sh_channels: int = 16
    global_sh_lmax: int = 2
    global_sh_out_dim: int = 64
    global_sh_separate_proj: bool = False   # separate weight projections for bound/unbound
    global_sh_mode: str = "cross"           # "cross" or "cross_l_norm"
    global_sh_head: str = "none"            # "none", "gate", "self_cg", "gate_cg"
    global_sh_n_heads: int = 1              # number of independent weight projections
    global_sh_atomica: bool = False          # condition SH weights on ATOMICA interface embedding
    global_sh_radial_form: str = "learned"  # "learned" (default MLP) or "multipole" (r^-(l+1) * cos_cutoff, no params)

    # Residue-level global SH pooling: coarser angular descriptors from residue centroids
    residue_global_sh_pooling: bool = False
    residue_global_sh_channels: int = 16
    residue_global_sh_lmax: int = 2
    residue_global_sh_out_dim: int = 64
    residue_global_sh_separate_proj: bool = False
    residue_global_sh_head: str = "none"
    residue_global_sh_pre_mlp: bool = False  # MLP on pooled residue features before SH projection

    # Edge global SH: angular descriptors of contact geometry (inter-molecular edge directions)
    edge_global_sh: bool = False             # SH expansion over edge direction vectors
    edge_global_sh_channels: int = 16
    edge_global_sh_lmax: int = 2
    edge_global_sh_out_dim: int = 64
    edge_global_sh_residue_pool: bool = False  # pool edges to residue pairs before SH

    # Residue pooling: how atoms aggregate to residues for residue-level SH
    residue_pool_attn: bool = False          # attention-weighted atom→residue pooling
    residue_cross_attn: bool = False         # cross-residue self-attention after pooling
    residue_cross_attn_heads: int = 4

    # Residue-level onsite: pool atoms→residues, compare bound vs unbound
    # "none" disables (default), "delta", "delta_mlp", "concat_mlp", "reembed_cg"
    residue_onsite_mode: str = "none"
    residue_onsite_dim: int = 64
    residue_onsite_reembed_lmax: int = 1  # for reembed_cg mode
    residue_agg: str = "mean"           # "mean" or "sum"

    # Readout fusion: combine multiple pooling streams before head
    # "none" = concat (default), "gated", "film", "self_attn"
    readout_fusion: str = "none"
    readout_fusion_dim: int = 128


class EquivariantMPNN(nn.Module):
    """Unified equivariant MPNN with optional onsite interactions and stream pooling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.use_precomputed_edges = cfg.use_precomputed_edges

        self.irreps_hidden = Irreps(cfg.irreps_hidden)

        # Knob 6: Input atom type embeddings
        if cfg.input_atom_embed_dim > 0:
            self.atom_type_embed = nn.Embedding(cfg.num_atom_types, cfg.input_atom_embed_dim)
            input_dim = cfg.input_atom_embed_dim + cfg.num_scalar_features
        else:
            self.atom_type_embed = None
            input_dim = cfg.num_atom_types + cfg.num_scalar_features

        # Amino acid type embedding (pocket atoms only)
        if cfg.aa_embed_dim > 0:
            self.aa_embed = nn.Embedding(cfg.num_aa_types, cfg.aa_embed_dim, padding_idx=0)
            input_dim += cfg.aa_embed_dim
        else:
            self.aa_embed = None

        self.irreps_in = Irreps(f"{input_dim}x0e")

        # Input projection: scalars -> scalar subset of hidden
        self.irreps_scalar = Irreps([(m, ir) for m, ir in self.irreps_hidden if ir.l == 0])
        self.input_proj = Linear(self.irreps_in, self.irreps_scalar)

        # Separate input projection for residue nodes (different feature space)
        self.residue_input_dim = cfg.residue_input_dim
        if cfg.residue_input_dim > 0:
            residue_irreps_in = Irreps(f"{cfg.residue_input_dim}x0e")
            self.residue_input_proj = Linear(residue_irreps_in, self.irreps_scalar)
        else:
            self.residue_input_proj = None

        # Edge dimension
        if cfg.use_precomputed_edges:
            self.edge_dim = cfg.radial_hidden
            self.edge_proj = nn.Sequential(
                nn.Linear(cfg.precomputed_edge_dim, cfg.radial_hidden),
                nn.SiLU(),
                nn.Linear(cfg.radial_hidden, self.edge_dim),
            )
        else:
            self.edge_dim = cfg.num_basis
            self.edge_proj = None

        # Knob 2: Cross-object edge type embeddings
        if cfg.cross_object_embed_dim > 0:
            self.edge_dim += cfg.cross_object_embed_dim
            self.cross_object_embed = nn.Embedding(3, cfg.cross_object_embed_dim)
        else:
            self.cross_object_embed = None

        # Knob 5: Destination features
        if cfg.use_dst_features:
            self.edge_dim += input_dim
        self.use_dst_features = cfg.use_dst_features

        # Node-conditioned radial
        self.node_conditioned_radial = cfg.node_conditioned_radial
        self.n_scalar_channels = sum(m for m, ir in self.irreps_hidden if ir.l == 0)
        # Asymmetric angular: l>0 channels start right after scalars (contiguous)
        self.pocket_scalar_only = cfg.pocket_scalar_only
        self._l_gt0_start = self.n_scalar_channels
        if cfg.node_conditioned_radial:
            self.edge_dim += 2 * self.n_scalar_channels

        # Context injection: project external embedding to scalar channels
        self.context_proj = None
        if cfg.context_embed_dim > 0:
            self.context_proj = nn.Linear(cfg.context_embed_dim, self.n_scalar_channels)

        # Per-residue LM injection: project per-residue embedding to scalar channels
        self.residue_lm_proj = None
        if cfg.residue_lm_dim > 0:
            self.residue_lm_proj = nn.Linear(cfg.residue_lm_dim, self.n_scalar_channels)

        # Message passing layers
        self.layers = nn.ModuleList()
        if cfg.backbone == "gin":
            for i in range(cfg.num_layers):
                in_dim = self.n_scalar_channels if i == 0 else self.n_scalar_channels
                # GIN input projection for first layer (raw features → hidden)
                if i == 0:
                    in_dim = sum(m for m, ir in self.irreps_scalar if ir.l == 0)
                self.layers.append(GINLayer(in_dim, self.n_scalar_channels))
        else:
            for i in range(cfg.num_layers):
                irreps_in_layer = self.irreps_scalar if i == 0 else self.irreps_hidden
                self.layers.append(
                    TPConv(
                        irreps_in=irreps_in_layer,
                        irreps_out=self.irreps_hidden,
                        edge_dim=self.edge_dim,
                        lmax=cfg.lmax,
                        hidden=cfg.radial_hidden,
                        num_radial_layers=cfg.radial_layers,
                        aggr=cfg.aggr,
                        use_norm=cfg.use_norm,
                        self_cg=cfg.self_cg,
                        tp_mode=cfg.tp_mode,
                        sparsity=cfg.tp_sparsity,
                        envelope_cutoff=cfg.envelope_cutoff,
                    )
                )

        # Virtual node in MP: per-layer global aggregate + broadcast (scalar channels only)
        # Uses a bottleneck MLP: scalars → scalars//4 → scalars to keep params reasonable
        self.vn_mlps = None
        if cfg.virtual_node_mp:
            sc = self.n_scalar_channels
            bottleneck = max(sc // 4, 64)
            self.vn_mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(sc, bottleneck), nn.SiLU(), nn.Linear(bottleneck, sc))
                for _ in range(cfg.num_layers)
            ])

        # Project to scalars for pooling
        self.jk_readout = cfg.jk_readout
        if cfg.jk_readout:
            jk_input_dim = cfg.num_layers * self.n_scalar_channels
            self.node_proj = nn.Linear(jk_input_dim, cfg.scalars)
        else:
            self.node_proj = Linear(self.irreps_hidden, Irreps(f"{cfg.scalars}x0e"))

        # Optional projection of node scalars before pooling
        self.node_pool_proj = None
        self._node_pool_dim = cfg.scalars  # dim per node after projection
        if cfg.node_pool_proj_dim > 0:
            self.node_pool_proj = nn.Sequential(
                nn.Linear(cfg.scalars, cfg.node_pool_proj_dim),
                nn.SiLU(),
            )
            self._node_pool_dim = cfg.node_pool_proj_dim

        # Gated attention pooling: learn per-node importance weight
        self.pool_attn = None
        if cfg.pool_attention:
            attn_input_dim = self._node_pool_dim
            self.pool_attn = nn.Sequential(
                nn.Linear(attn_input_dim, attn_input_dim),
                nn.SiLU(),
                nn.Linear(attn_input_dim, 1),
            )

        # Onsite interaction (see onsite.py for all modes)
        if cfg.onsite_mode != "none":
            onsite_kwargs = dict(cg_mode=cfg.onsite_cg_mode)
            if "reembed_cg" in cfg.onsite_mode:
                onsite_kwargs.update(
                    reembed_lmax=cfg.onsite_reembed_lmax,
                    envelope_cutoff=cfg.envelope_cutoff,
                    r_max=cfg.r_max,
                    num_radial_basis=cfg.num_basis,
                    reembed_attention=cfg.reembed_attention,
                    contraction_refs=cfg.contraction_refs,
                    reembed_iterations=cfg.reembed_iterations,
                    dual_cg=cfg.dual_cg,
                    degree2_invariants=cfg.degree2_invariants,
                    reembed_proj_dim=cfg.onsite_reembed_proj_dim,
                    out_proj_bias=cfg.onsite_out_proj_bias,
                )
            self.onsite = build_onsite(
                self.irreps_hidden, cfg.onsite_dim, cfg.onsite_mode, **onsite_kwargs
            )
            onsite_output_dim = self.onsite.output_dim
            self._onsite_needs_geometry = "reembed_cg" in cfg.onsite_mode
        else:
            self.onsite = None
            onsite_output_dim = 0
            self._onsite_needs_geometry = False

        # Self-attention over per-atom onsite features before pooling
        self.onsite_attn = None
        if cfg.onsite_self_attn and onsite_output_dim > 0:
            self.onsite_attn = nn.TransformerEncoderLayer(
                d_model=onsite_output_dim,
                nhead=cfg.onsite_self_attn_heads,
                dim_feedforward=4 * onsite_output_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # pre-norm for stability
            )

        # Direction 1: Multi-scale reembed — apply onsite at every MP layer
        self._multiscale_reembed = cfg.multiscale_reembed
        if cfg.multiscale_reembed:
            assert cfg.persistent_edge_state, \
                "multiscale_reembed requires persistent_edge_state=True"
            assert self.onsite is not None, \
                "multiscale_reembed requires onsite_mode != 'none'"
            self.layer_weights = nn.Parameter(torch.zeros(cfg.num_layers))

        # Edge onsite (compare bound vs unbound edge features at inter edges)
        # Build kwargs for edge comparison modules that need geometry
        edge_cg_kwargs = {}
        if "reembed_cg" in cfg.edge_onsite_mode:
            edge_cg_kwargs = dict(
                reembed_lmax=cfg.edge_onsite_reembed_lmax,
                cg_mode=cfg.edge_onsite_cg_mode,
                r_max=cfg.r_max,
                num_radial_basis=cfg.num_basis,
            )

        self.persistent_edge = None
        self._pe_readout_only = False  # True when persistent edges used only for raw readout
        _need_persistent = cfg.persistent_edge_state and (
            cfg.edge_onsite_mode != "none" or cfg.edge_readout
        )
        if _need_persistent and cfg.edge_onsite_mode != "none":
            self.persistent_edge = build_persistent_edge_onsite(
                node_scalar_dim=self.n_scalar_channels,
                edge_state_dim=cfg.edge_state_dim,
                edge_hidden_dim=cfg.edge_hidden_dim,
                edge_onsite_dim=cfg.edge_onsite_dim,
                mode=cfg.edge_onsite_mode,
                edge_feedback=cfg.edge_feedback,
                **edge_cg_kwargs,
            )
            self.edge_onsite = None
        elif _need_persistent:
            # edge_readout without comparison: build encoder + update only
            self.persistent_edge = build_persistent_edge_onsite(
                node_scalar_dim=self.n_scalar_channels,
                edge_state_dim=cfg.edge_state_dim,
                edge_hidden_dim=cfg.edge_hidden_dim,
                edge_onsite_dim=cfg.edge_onsite_dim,
                mode="delta",  # placeholder — comparison not used
                edge_feedback=cfg.edge_feedback,
            )
            self._pe_readout_only = True
            self.edge_onsite = None
        elif cfg.edge_onsite_mode != "none":
            self.edge_onsite = build_edge_onsite(
                node_scalar_dim=self.n_scalar_channels,
                edge_hidden_dim=cfg.edge_hidden_dim,
                edge_onsite_dim=cfg.edge_onsite_dim,
                mode=cfg.edge_onsite_mode,
                **edge_cg_kwargs,
            )
        else:
            self.edge_onsite = None

        # Virtual node onsite (graph-level comparison, one per stream)
        self.virtual_onsites = nn.ModuleDict()
        if cfg.virtual_onsite_mode != "none":
            for stream in cfg.virtual_pool_streams:
                self.virtual_onsites[stream] = VirtualNodeOnsite(
                    node_scalar_dim=self.n_scalar_channels,
                    num_virtual_nodes=cfg.num_virtual_nodes,
                    virtual_dim=cfg.virtual_node_dim,
                    dropout=cfg.virtual_dropout,
                )
        virtual_onsite_output_dim = sum(
            v.output_dim for v in self.virtual_onsites.values()
        )

        # Virtual edge onsite (K virtual nodes attend to persistent edge states)
        self.virtual_edge_onsite = None
        virtual_edge_onsite_output_dim = 0
        if cfg.virtual_edge_onsite_mode != "none":
            assert cfg.persistent_edge_state, \
                "virtual_edge_onsite requires persistent_edge_state=True"
            self.virtual_edge_onsite = VirtualNodeOnsite(
                node_scalar_dim=cfg.edge_state_dim,
                num_virtual_nodes=cfg.num_virtual_nodes,
                virtual_dim=cfg.virtual_node_dim,
            )
            virtual_edge_onsite_output_dim = self.virtual_edge_onsite.output_dim

        # Global SH pooling (angular descriptors, 2-copy only)
        self.global_sh = None
        global_sh_output_dim = 0
        if cfg.global_sh_pooling:
            self.global_sh = GlobalSHPooling(
                n_scalar_in=self.n_scalar_channels,
                n_channels=cfg.global_sh_channels,
                lmax=cfg.global_sh_lmax,
                out_dim=cfg.global_sh_out_dim,
                separate_proj=cfg.global_sh_separate_proj,
                mode=cfg.global_sh_mode,
                head=cfg.global_sh_head,
                n_heads=cfg.global_sh_n_heads,
                esm_proj_dim=32 if cfg.global_sh_atomica else 0,
                esm_input_dim=32 if cfg.global_sh_atomica else 1280,
                radial_form=cfg.global_sh_radial_form,
            )
            global_sh_output_dim = self.global_sh.output_dim

        # Edge global SH: angular descriptors of contact geometry
        # Uses edge direction vectors (protein→ligand) as positions,
        # edge scalar features as weights. Compares bound vs unbound contact patterns.
        self.edge_global_sh = None
        edge_global_sh_output_dim = 0
        if cfg.edge_global_sh:
            # Edge features come from persistent edge states (edge_state_dim) or
            # from node scalars at edge endpoints (n_scalar_channels)
            edge_sh_input_dim = cfg.edge_state_dim if cfg.persistent_edge_state else self.n_scalar_channels
            self.edge_global_sh = GlobalSHPooling(
                n_scalar_in=edge_sh_input_dim,
                n_channels=cfg.edge_global_sh_channels,
                lmax=cfg.edge_global_sh_lmax,
                out_dim=cfg.edge_global_sh_out_dim,
                separate_proj=True,  # bound and unbound edge states differ
            )
            edge_global_sh_output_dim = self.edge_global_sh.output_dim

        # Multi-Shell Radial readout (no SH; K parallel signed radial filters).
        # See models/msr.py. Default off, matches GlobalSHPooling forward signature
        # so it can plug into the same place in the readout pipeline.
        self.msr = None
        msr_output_dim = 0
        if cfg.msr_pooling:
            self.msr = MultiShellRadial(
                n_scalar_in=self.n_scalar_channels,
                n_channels=cfg.msr_n_channels,
                num_filters=cfg.msr_num_filters,
                out_dim=cfg.msr_out_dim,
                separate_proj=cfg.msr_separate_proj,
                num_rbf=cfg.msr_num_rbf,
                r_max=cfg.msr_r_max,
            )
            msr_output_dim = self.msr.output_dim

        # Residue-level global SH pooling (coarser angular descriptors from residue centroids)
        self.residue_global_sh = None
        self.residue_sh_pre_mlp = None
        residue_global_sh_output_dim = 0
        if cfg.residue_global_sh_pooling:
            self.residue_global_sh = GlobalSHPooling(
                n_scalar_in=self.n_scalar_channels,
                n_channels=cfg.residue_global_sh_channels,
                lmax=cfg.residue_global_sh_lmax,
                out_dim=cfg.residue_global_sh_out_dim,
                separate_proj=cfg.residue_global_sh_separate_proj,
                head=cfg.residue_global_sh_head,
            )
            residue_global_sh_output_dim = self.residue_global_sh.output_dim
            if cfg.residue_global_sh_pre_mlp:
                s = self.n_scalar_channels
                self.residue_sh_pre_mlp = nn.Sequential(
                    nn.Linear(s, s), nn.SiLU(), nn.Linear(s, s),
                )

        # Residue pool attention: learn per-atom importance within each residue
        self.residue_pool_attn_mlp = None
        if cfg.residue_pool_attn and cfg.residue_global_sh_pooling:
            s = self.n_scalar_channels
            self.residue_pool_attn_mlp = nn.Sequential(
                nn.Linear(s, s // 4), nn.SiLU(), nn.Linear(s // 4, 1),
            )

        # Cross-residue self-attention after atom→residue pooling
        self.residue_cross_attn = None
        if cfg.residue_cross_attn and cfg.residue_global_sh_pooling:
            s = self.n_scalar_channels
            self.residue_cross_attn = nn.TransformerEncoderLayer(
                d_model=s, nhead=cfg.residue_cross_attn_heads,
                dim_feedforward=4 * s, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            )

        # Residue-level onsite (pool atoms→residues, compare bound vs unbound)
        self.residue_onsite = None
        residue_onsite_output_dim = 0
        if cfg.residue_onsite_mode != "none":
            self.residue_onsite = ResidueOnsite(
                n_channels_in=self.n_scalar_channels,
                out_dim=cfg.residue_onsite_dim,
                agg=cfg.residue_agg,
                compare=cfg.residue_onsite_mode,
                reembed_lmax=cfg.residue_onsite_reembed_lmax,
            )
            residue_onsite_output_dim = self.residue_onsite.output_dim

        # Pooling streams
        self.pool_streams = cfg.pool_streams
        self._needs_ligand_mask = (
            any(s in ("ligand", "pocket") for s in cfg.pool_streams)
            or any(s in ("ligand", "pocket") for s in cfg.virtual_pool_streams)
            or self.edge_onsite is not None
            or self.persistent_edge is not None
            or self.global_sh is not None
            or self.residue_global_sh is not None
            or self.msr is not None
        )

        # Pooling
        self.pool = scatter_mean if cfg.pool_mode == "mean" else scatter_sum

        # Prediction head
        # Each stream contributes: scalars + onsite_output_dim
        # Edge onsite adds a single vector (not per-stream)
        if cfg.use_2copies:
            per_stream_dim = self._node_pool_dim + onsite_output_dim
        else:
            per_stream_dim = self._node_pool_dim  # no onsite in 1-copy mode
        if self.persistent_edge is not None and not self._pe_readout_only:
            edge_onsite_output_dim = self.persistent_edge.output_dim
        elif self.edge_onsite is not None:
            edge_onsite_output_dim = self.edge_onsite.output_dim
        else:
            edge_onsite_output_dim = 0
        edge_readout_dim = cfg.edge_state_dim if (cfg.edge_readout and cfg.persistent_edge_state) else 0
        if not cfg.use_2copies:
            # In 1-copy mode, no bound/unbound comparison → zero out 2-copy-only dims
            edge_onsite_output_dim = 0
            virtual_onsite_output_dim = 0
            virtual_edge_onsite_output_dim = 0
            edge_readout_dim = 0
            global_sh_output_dim = 0
            edge_global_sh_output_dim = 0
            residue_onsite_output_dim = 0
            residue_global_sh_output_dim = 0
        # Bound streams include onsite (per_stream_dim), unbound streams are scalars only (_node_pool_dim)
        unbound_per_stream_dim = self._node_pool_dim  # no onsite in unbound pool
        node_pool_dim = 0
        if cfg.node_readout:
            node_pool_dim = len(cfg.pool_streams) * per_stream_dim  # bound streams
            if cfg.pool_unbound and cfg.use_2copies:
                node_pool_dim += len(cfg.pool_streams) * unbound_per_stream_dim  # unbound streams

        # Readout fusion: combine streams via learned fusion before head
        self.readout_fusion = None
        if cfg.readout_fusion != "none":
            # Collect stream dims in the order they appear in forward()
            stream_dims = []
            if cfg.node_readout:
                for _ in cfg.pool_streams:
                    stream_dims.append(per_stream_dim)
            if edge_onsite_output_dim > 0:
                stream_dims.append(edge_onsite_output_dim)
            if edge_readout_dim > 0:
                stream_dims.append(edge_readout_dim)
            if virtual_onsite_output_dim > 0:
                stream_dims.append(virtual_onsite_output_dim)
            if virtual_edge_onsite_output_dim > 0:
                stream_dims.append(virtual_edge_onsite_output_dim)
            if global_sh_output_dim > 0:
                stream_dims.append(global_sh_output_dim)
            if edge_global_sh_output_dim > 0:
                stream_dims.append(edge_global_sh_output_dim)
            if residue_onsite_output_dim > 0:
                stream_dims.append(residue_onsite_output_dim)
            if residue_global_sh_output_dim > 0:
                stream_dims.append(residue_global_sh_output_dim)
            assert len(stream_dims) >= 2, (
                f"Readout fusion requires >= 2 streams, got {len(stream_dims)}"
            )
            self.readout_fusion = build_readout_fusion(
                cfg.readout_fusion, stream_dims, cfg.readout_fusion_dim
            )
            head_input_dim = self.readout_fusion.output_dim
        else:
            head_input_dim = (node_pool_dim
                              + edge_onsite_output_dim
                              + virtual_onsite_output_dim
                              + virtual_edge_onsite_output_dim
                              + edge_readout_dim
                              + global_sh_output_dim
                              + edge_global_sh_output_dim
                              + residue_onsite_output_dim
                              + residue_global_sh_output_dim
                              + msr_output_dim)

        # ESM readout fusion: project precomputed ESM embeddings and append to head
        self.esm_proj = None
        if cfg.esm_readout_dim > 0:
            self.esm_proj = nn.Sequential(
                nn.Linear(cfg.esm_input_dim, cfg.esm_readout_dim),
                nn.SiLU(),
            )
            head_input_dim += cfg.esm_readout_dim

        if cfg.head_norm == "batch" or cfg.head_norm is True:
            self.head_bn = nn.BatchNorm1d(head_input_dim)
        elif cfg.head_norm == "layer":
            self.head_bn = nn.LayerNorm(head_input_dim)
        else:
            self.head_bn = None
        head = [nn.Linear(head_input_dim, cfg.head_hidden), nn.SiLU(), nn.Dropout(cfg.dropout)]
        for _ in range(cfg.head_layers - 1):
            head.extend([nn.Linear(cfg.head_hidden, cfg.head_hidden), nn.SiLU(), nn.Dropout(cfg.dropout)])
        head.append(nn.Linear(cfg.head_hidden, 1))
        self.head = nn.Sequential(*head)

        # Store config values
        self.num_atom_types = cfg.num_atom_types
        self.num_basis = cfg.num_basis
        self.r_max = cfg.r_max
        self.feature_mask = cfg.feature_mask
        self.feature_mask_target = cfg.feature_mask_target

    def _split_bound_unbound(self, x: torch.Tensor, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split 2N node features into bound (first N) and unbound (last N) per graph."""
        ptr = data.ptr
        n_orig = data.num_nodes_original
        h_bound_list, h_unbound_list = [], []
        for i in range(len(n_orig)):
            start = ptr[i].item()
            n = n_orig[i].item() if isinstance(n_orig[i], torch.Tensor) else n_orig[i]
            h_bound_list.append(x[start : start + n])
            h_unbound_list.append(x[start + n : start + 2 * n])
        return torch.cat(h_bound_list), torch.cat(h_unbound_list)

    def _build_batch_original(self, data) -> torch.Tensor:
        """Build batch tensor for original (non-duplicated) nodes."""
        n_orig = data.num_nodes_original
        return torch.arange(len(n_orig), device=data.x.device).repeat_interleave(n_orig)

    def _build_mol_code_2copies(self, num_nodes: int, data, device) -> torch.Tensor:
        """Build mol_code for 2-copy data (mark ligand atoms in both copies)."""
        mol_code = torch.zeros(num_nodes, dtype=torch.long, device=device)
        nla = data.num_ligand_atoms
        n_orig = data.num_nodes_original
        ptr = data.ptr
        for i in range(len(nla)):
            n_lig = nla[i].item() if isinstance(nla[i], torch.Tensor) else nla[i]
            n_o = n_orig[i].item() if isinstance(n_orig[i], torch.Tensor) else n_orig[i]
            start = ptr[i].item()
            mol_code[start : start + n_lig] = 1
            mol_code[start + n_o : start + n_o + n_lig] = 1
        return mol_code

    def _build_ligand_mask(self, batch: torch.Tensor, num_ligand_atoms) -> torch.Tensor:
        """Build boolean mask for ligand atoms (True = ligand, False = pocket).

        Assumes ligand atoms come first in each graph, followed by pocket atoms.
        Works with both single-graph (int) and batched (tensor) num_ligand_atoms.
        """
        device = batch.device
        mask = torch.zeros(batch.size(0), dtype=torch.bool, device=device)

        if isinstance(num_ligand_atoms, int):
            num_ligand_atoms = [num_ligand_atoms]
        elif hasattr(num_ligand_atoms, 'tolist'):
            num_ligand_atoms = num_ligand_atoms.tolist()

        for b in range(batch.max().item() + 1):
            graph_indices = (batch == b).nonzero(as_tuple=True)[0]
            n_lig = num_ligand_atoms[b] if b < len(num_ligand_atoms) else num_ligand_atoms[0]
            mask[graph_indices[:n_lig]] = True

        return mask

    def _build_full_ligand_mask(self, data) -> torch.Tensor:
        """Build ligand mask over ALL nodes (both copies in 2-copy mode).

        In 2-copy mode, layout per graph is [bound_lig, bound_pock, unbound_lig, unbound_pock].
        Returns True for ligand atoms in both copies.
        """
        device = data.x.device
        total = data.x.size(0)
        mask = torch.zeros(total, dtype=torch.bool, device=device)
        ptr = data.ptr
        nla = data.num_ligand_atoms
        if hasattr(nla, 'tolist'):
            nla = nla.tolist()
        elif isinstance(nla, int):
            nla = [nla]

        has_2copies = hasattr(data, 'num_nodes_original')
        if has_2copies:
            n_orig = data.num_nodes_original
            if hasattr(n_orig, 'tolist'):
                n_orig = n_orig.tolist()
            for i in range(len(nla)):
                s = ptr[i].item()
                n, nl = n_orig[i], nla[i]
                mask[s:s + nl] = True            # bound ligand
                mask[s + n:s + n + nl] = True     # unbound ligand
        else:
            for i in range(len(nla)):
                s = ptr[i].item()
                mask[s:s + nla[i]] = True
        return mask

    def _extract_residue_ids(self, data):
        """Extract per-atom residue_id for the bound copy (first N nodes per graph)."""
        ptr = data.ptr
        n_orig = data.num_nodes_original
        rid_parts = []
        for i in range(len(n_orig)):
            start = ptr[i].item()
            n = n_orig[i].item() if isinstance(n_orig[i], torch.Tensor) else n_orig[i]
            rid_parts.append(data.residue_id[start : start + n])
        return torch.cat(rid_parts)

    def _build_copy_flag_and_remap(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build copy_flag (0=bound, 1=unbound) and remap (2N→N index mapping)."""
        ptr = data.ptr
        n_orig = data.num_nodes_original
        device = data.x.device
        total_nodes = data.x.size(0)

        copy_flag = torch.zeros(total_nodes, dtype=torch.long, device=device)
        remap = torch.zeros(total_nodes, dtype=torch.long, device=device)
        cumulative = 0
        for i in range(len(n_orig)):
            start = ptr[i].item()
            n = n_orig[i].item() if isinstance(n_orig[i], torch.Tensor) else n_orig[i]
            arange_n = torch.arange(n, device=device)
            remap[start:start + n] = arange_n + cumulative
            remap[start + n:start + 2 * n] = arange_n + cumulative
            copy_flag[start + n:start + 2 * n] = 1
            cumulative += n

        return copy_flag, remap

    def _extract_geometry_for_onsite(self, pos: torch.Tensor, data):
        """Extract original-node positions and per-copy edge indices.

        In 2-copy mode, the graph has 2N nodes (N bound + N unbound) per complex.
        Returns original positions and bound/unbound edge indices in N-node space.
        """
        copy_flag, remap = self._build_copy_flag_and_remap(data)
        src, dst = data.edge_index

        bound_mask = (copy_flag[src] == 0) & (copy_flag[dst] == 0)
        bound_edge_index = torch.stack([remap[src[bound_mask]], remap[dst[bound_mask]]])

        unbound_mask = (copy_flag[src] == 1) & (copy_flag[dst] == 1)
        unbound_edge_index = torch.stack([remap[src[unbound_mask]], remap[dst[unbound_mask]]])

        pos_bound = pos[copy_flag == 0]
        pos_unbound = pos[copy_flag == 1]
        return pos_bound, pos_unbound, bound_edge_index, unbound_edge_index

    def _extract_inter_edges(self, data, ligand_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract unique inter-molecular edges in N-node space.

        Returns (inter_src, inter_dst, edge_batch) where each edge appears
        once (deduplicated via src < dst).
        """
        copy_flag, remap = self._build_copy_flag_and_remap(data)
        src_all, dst_all = data.edge_index

        # Get bound-copy edges, remap to N-node space
        bound_mask = (copy_flag[src_all] == 0) & (copy_flag[dst_all] == 0)
        bound_src = remap[src_all[bound_mask]]
        bound_dst = remap[dst_all[bound_mask]]

        # Inter-molecular: ligand_mask differs between endpoints
        is_inter = ligand_mask[bound_src] != ligand_mask[bound_dst]
        inter_src = bound_src[is_inter]
        inter_dst = bound_dst[is_inter]

        # Deduplicate: keep only src < dst (undirected edges)
        keep = inter_src < inter_dst
        inter_src = inter_src[keep]
        inter_dst = inter_dst[keep]

        # Edge batch: graph assignment from source node
        batch_original = self._build_batch_original(data)
        edge_batch = batch_original[inter_src]

        return inter_src, inter_dst, edge_batch

    def _pool_stream(self, stream_name: str, node_scalars: torch.Tensor,
                     onsite_feats, batch: torch.Tensor, batch_size: int,
                     ligand_mask) -> torch.Tensor:
        """Pool a single stream (all/ligand/pocket) and return graph-level features."""
        if stream_name == "all":
            m = slice(None)  # all nodes
        elif stream_name == "ligand":
            m = ligand_mask
        elif stream_name == "pocket":
            m = ~ligand_mask
        else:
            raise ValueError(f"Unknown pool stream '{stream_name}'")

        if self.pool_attn is not None:
            # Gated attention pooling: softmax over nodes within each graph
            logits = self.pool_attn(node_scalars[m]).squeeze(-1)  # [N_stream]
            # Per-graph softmax: subtract max for stability, then exp / sum
            max_logits = scatter(logits, batch[m], dim=0, dim_size=batch_size, reduce="max")
            logits = logits - max_logits[batch[m]]
            exp_logits = logits.exp()
            Z = scatter_sum(exp_logits, batch[m], dim=0, dim_size=batch_size)
            attn = exp_logits / (Z[batch[m]] + 1e-8)  # [N_stream]
            attn = attn.unsqueeze(-1)  # [N_stream, 1]
            parts = [scatter_sum(attn * node_scalars[m], batch[m], dim=0, dim_size=batch_size)]
            if onsite_feats is not None:
                parts.append(scatter_sum(attn * onsite_feats[m], batch[m], dim=0, dim_size=batch_size))
        else:
            parts = [self.pool(node_scalars[m], batch[m], dim=0, dim_size=batch_size)]
            if onsite_feats is not None:
                parts.append(self.pool(onsite_feats[m], batch[m], dim=0, dim_size=batch_size))
        return torch.cat(parts, dim=-1)

    def forward(self, data) -> Tuple[torch.Tensor, Dict[str, Any]]:
        has_2copies = hasattr(data, 'num_nodes_original')

        # 1. Extract positions and features
        pos = data.x[:, :3]
        x_raw = data.x[:, 3:]

        # Feature masking (CORDIAL-like ablation)
        if self.feature_mask != "none":
            x_raw = x_raw.clone()
            if self.feature_mask_target != "both":
                lm = self._build_full_ligand_mask(data)
                nm = lm if self.feature_mask_target == "ligand" else ~lm
            else:
                nm = slice(None)
            if self.feature_mask == "no_atom_types":
                x_raw[nm, :self.num_atom_types] = 0.0
            elif self.feature_mask == "topology_only":
                x_raw[nm, :self.num_atom_types] = 0.0
                x_raw[nm, self.num_atom_types + 14:] = 0.0
            elif self.feature_mask == "geometry_only":
                x_raw[nm] = 0.0

        atom_type_onehot = x_raw[:, :self.num_atom_types]
        scalar_feats = x_raw[:, self.num_atom_types:]
        atom_types = atom_type_onehot.argmax(dim=-1)

        # Knob 6: Apply atom type embedding
        if self.atom_type_embed is not None:
            atom_emb = self.atom_type_embed(atom_types)
            x = torch.cat([atom_emb, scalar_feats], dim=-1)
        else:
            x = x_raw

        # Amino acid type embedding (pocket atoms only, ligand gets padding_idx=0)
        if self.aa_embed is not None:
            if hasattr(data, 'aa_type'):
                aa_emb = self.aa_embed(data.aa_type)
            else:
                aa_emb = torch.zeros(x.size(0), self.cfg.aa_embed_dim, device=x.device)
            x = torch.cat([x, aa_emb], dim=-1)

        # 2. Build edge attributes
        if self.use_precomputed_edges and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = self.edge_proj(data.edge_attr)
        else:
            bin_width = self.r_max / max(1, self.num_basis - 1)
            edge_attr = soft_one_hot_linspace(
                data.edge_length, 0.0, self.r_max + bin_width,
                self.num_basis, basis="smooth_finite", cutoff=True
            )
            edge_attr = edge_attr / edge_attr.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        src, dst = data.edge_index

        # Knob 2: Cross-object edge type embeddings
        if self.cross_object_embed is not None:
            if has_2copies:
                mol_code = self._build_mol_code_2copies(x.size(0), data, x.device)
            elif hasattr(data, 'mol_code'):
                mol_code = data.mol_code
            elif hasattr(data, 'num_ligand_atoms'):
                mol_code = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                nla = data.num_ligand_atoms
                if isinstance(nla, int):
                    mol_code[:nla] = 1
                else:
                    ptr = data.ptr
                    for i, n in enumerate(nla):
                        mol_code[ptr[i]:ptr[i] + n] = 1
            else:
                raise ValueError(
                    "cross_object_embed enabled but data has no mol_code or num_ligand_atoms"
                )

            src_mol = mol_code[src]
            dst_mol = mol_code[dst]
            edge_type = torch.where(
                src_mol == dst_mol,
                src_mol,
                torch.full_like(src_mol, 2)
            )
            cross_embed = self.cross_object_embed(edge_type)
            edge_attr = torch.cat([edge_attr, cross_embed], dim=-1)

        # Knob 5: Destination features
        if self.use_dst_features:
            dst_feats = x[dst]
            edge_attr = torch.cat([edge_attr, dst_feats], dim=-1)

        # 3. Input projection (to scalars only)
        if self.residue_input_proj is not None and hasattr(data, 'x_physchem'):
            # Separate projections: ligand atoms use input_proj, residue nodes use residue_input_proj
            lig_mask = self._build_full_ligand_mask(data)
            h = torch.zeros(x.size(0), self.irreps_scalar.dim, device=x.device, dtype=x.dtype)
            h[lig_mask] = self.input_proj(x[lig_mask])
            h[~lig_mask] = self.residue_input_proj(data.x_physchem[~lig_mask])
            x = h
        else:
            x = self.input_proj(x)

        # 3b. Context injection: add projected external embedding to all nodes
        if self.context_proj is not None and hasattr(data, 'esm_embedding'):
            ctx = self.context_proj(data.esm_embedding)  # [B, n_scalar_channels]
            ctx_per_node = ctx[data.batch]                # [2N, n_scalar_channels]
            x = x + ctx_per_node

        # 3c. Per-residue LM injection: add per-atom projected residue embeddings
        if self.residue_lm_proj is not None and hasattr(data, 'residue_lm_embedding'):
            x = x + self.residue_lm_proj(data.residue_lm_embedding)

        # 4. Message passing
        base_edge_attr = edge_attr
        layer_scalars = [] if self.jk_readout else None

        # Multi-scale reembed: collect per-layer bound/unbound for onsite
        _ms = self._multiscale_reembed and has_2copies
        _ms_bound = [] if _ms else None
        _ms_unbound = [] if _ms else None

        # Persistent edge state: pre-loop setup
        _pe = self.persistent_edge is not None and has_2copies
        e_bound = None
        e_unbound = None
        if _pe:
            _pe_cf, _pe_rm = self._build_copy_flag_and_remap(data)
            _pe_bo = self._build_batch_original(data)
            _pe_lm = self._build_ligand_mask(_pe_bo, data.num_ligand_atoms)
            _pe_src, _pe_dst, _pe_eb = self._extract_inter_edges(data, _pe_lm)
            _pe_N = x[_pe_cf == 0].size(0)
            e_bound = self.persistent_edge.init_edge_states(
                x[_pe_cf == 0], _pe_src, _pe_dst)
            e_unbound = self.persistent_edge.init_edge_states(
                x[_pe_cf == 1], _pe_src, _pe_dst)

        # Virtual node in MP: pre-loop setup (need copy_flag + batch for 2-copy)
        _vn = self.vn_mlps is not None and has_2copies
        if _vn and not _pe:
            _pe_cf, _ = self._build_copy_flag_and_remap(data)
            _pe_bo = self._build_batch_original(data)
        if _vn:
            _vn_bs = int(_pe_bo.max().item()) + 1

        # Asymmetric angular: mask for zeroing l>0 on pocket atoms
        _pso = self.pocket_scalar_only and has_2copies and self._l_gt0_start < x.size(-1)
        _pocket_mask = None

        for li, layer in enumerate(self.layers):
            if self.node_conditioned_radial:
                src_s = x[src, :self.n_scalar_channels]
                dst_s = x[dst, :self.n_scalar_channels]
                edge_attr = torch.cat([base_edge_attr, src_s, dst_s], dim=-1)
            x = layer(x, pos, data.edge_index, edge_attr)
            if _pso:
                if _pocket_mask is None:
                    _pocket_mask = ~self._build_full_ligand_mask(data)
                x = x.clone()
                x[_pocket_mask, self._l_gt0_start:] = 0.0
            if self.jk_readout:
                layer_scalars.append(x[:, :self.n_scalar_channels])
            if _vn:
                sc = self.n_scalar_channels
                bound_mask = _pe_cf == 0
                unbound_mask = _pe_cf == 1
                x_b_sc = x[bound_mask, :sc]
                x_u_sc = x[unbound_mask, :sc]
                g_b = scatter_mean(x_b_sc, _pe_bo, dim=0, dim_size=_vn_bs)
                g_u = scatter_mean(x_u_sc, _pe_bo, dim=0, dim_size=_vn_bs)
                vn_b = self.vn_mlps[li](g_b)[_pe_bo]
                vn_u = self.vn_mlps[li](g_u)[_pe_bo]
                x = x.clone()
                x[bound_mask, :sc] = x[bound_mask, :sc] + vn_b
                x[unbound_mask, :sc] = x[unbound_mask, :sc] + vn_u
            if _pe:
                sb = x[_pe_cf == 0, :self.n_scalar_channels]
                su = x[_pe_cf == 1, :self.n_scalar_channels]
                e_bound = self.persistent_edge.update_edge_states(
                    e_bound, sb, _pe_src, _pe_dst)
                e_unbound = self.persistent_edge.update_edge_states(
                    e_unbound, su, _pe_src, _pe_dst)
                if self.persistent_edge.edge_feedback:
                    fb_b = self.persistent_edge.scatter_feedback(
                        e_bound, _pe_src, _pe_dst, _pe_N)
                    fb_u = self.persistent_edge.scatter_feedback(
                        e_unbound, _pe_src, _pe_dst, _pe_N)
                    fb = torch.zeros(x.size(0), self.n_scalar_channels,
                                     device=x.device, dtype=x.dtype)
                    fb[_pe_cf == 0] = fb_b
                    fb[_pe_cf == 1] = fb_u
                    sc = x[:, :self.n_scalar_channels] + fb
                    if self.n_scalar_channels < x.size(-1):
                        x = torch.cat([sc, x[:, self.n_scalar_channels:]], dim=-1)
                    else:
                        x = sc
            if _ms:
                _ms_bound.append(x[_pe_cf == 0])
                _ms_unbound.append(x[_pe_cf == 1])

        # 5. Split bound/unbound (if 2-copy), project, compute onsite, pool
        if has_2copies:
            h_bound, h_unbound = self._split_bound_unbound(x, data)
            batch_original = self._build_batch_original(data)
            batch_size = int(batch_original.max().item()) + 1

            if self.jk_readout:
                jk_input = torch.cat(layer_scalars, dim=-1)
                jk_bound, _ = self._split_bound_unbound(jk_input, data)
                node_scalars = self.node_proj(jk_bound)
            else:
                node_scalars = self.node_proj(h_bound)

            onsite_feats = None
            if self.onsite is not None:
                onsite_kwargs = dict(batch=batch_original, batch_size=batch_size)
                if self._onsite_needs_geometry:
                    pos_b, pos_ub, bound_ei, unbound_ei = self._extract_geometry_for_onsite(pos, data)
                    onsite_kwargs.update(
                        pos=pos_b,
                        pos_unbound=pos_ub,
                        bound_edge_index=bound_ei,
                        unbound_edge_index=unbound_ei,
                    )
                if self._multiscale_reembed:
                    # Apply onsite at every MP layer, combine with learned weights
                    w = F.softmax(self.layer_weights, dim=0)
                    onsite_parts = []
                    for k in range(len(self.layers)):
                        ok = self.onsite(_ms_bound[k], _ms_unbound[k], **onsite_kwargs)
                        onsite_parts.append(w[k] * ok)
                    onsite_feats = sum(onsite_parts)
                else:
                    onsite_feats = self.onsite(
                        h_bound, h_unbound, **onsite_kwargs
                    )

            # Restrict on-site contribution to interface atoms only (atoms with
            # at least one inter-molecular edge incident). Tests the sum-domain
            # axis of the node-vs-edge comparison: aligns atom-on-site's scope
            # with edge-on-site's E_inter scope.
            if self.cfg.onsite_interface_mask and onsite_feats is not None:
                # Need ligand_mask in the original (single-copy) atom space to
                # extract inter-edges, then build a per-atom interface mask.
                _im_lm = self._build_ligand_mask(batch_original, data.num_ligand_atoms)
                _im_src, _im_dst, _ = self._extract_inter_edges(data, _im_lm)
                interface_atom_mask = torch.zeros(
                    onsite_feats.size(0), dtype=torch.bool, device=onsite_feats.device
                )
                interface_atom_mask[_im_src.unique()] = True
                interface_atom_mask[_im_dst.unique()] = True
                onsite_feats = onsite_feats * interface_atom_mask.unsqueeze(-1).to(onsite_feats.dtype)

            # Self-attention over per-atom onsite features (before pooling)
            if self.onsite_attn is not None and onsite_feats is not None:
                # Convert ragged batch to dense padded tensor for attention
                onsite_dense, attn_mask = to_dense_batch(onsite_feats, batch_original)
                # attn_mask: [B, N_max] True where valid, need inverted for key_padding_mask
                onsite_dense = self.onsite_attn(
                    onsite_dense, src_key_padding_mask=~attn_mask
                )
                # Convert back to ragged: select valid positions
                onsite_feats = onsite_dense[attn_mask]

            # Build ligand mask if needed
            ligand_mask = None
            if self._needs_ligand_mask:
                ligand_mask = self._build_ligand_mask(batch_original, data.num_ligand_atoms)

            # Edge onsite: compare bound vs unbound edge features at inter edges
            edge_onsite_feats = None
            if self.persistent_edge is not None and not self._pe_readout_only:
                pe_kwargs = {}
                if "reembed_cg" in self.cfg.edge_onsite_mode:
                    pos_original = pos[_pe_cf == 0]
                    pe_kwargs["rel_vec"] = pos_original[_pe_dst] - pos_original[_pe_src]
                edge_onsite_feats = self.persistent_edge.compare_and_pool(
                    e_bound, e_unbound, _pe_eb, batch_size, **pe_kwargs)
            elif self.edge_onsite is not None:
                inter_src, inter_dst, edge_batch = self._extract_inter_edges(
                    data, ligand_mask
                )
                sb = h_bound[:, :self.n_scalar_channels]
                su = h_unbound[:, :self.n_scalar_channels]
                eo_kwargs = {}
                if hasattr(self.edge_onsite, '_needs_rel_vec') and self.edge_onsite._needs_rel_vec:
                    pos_orig, _, _, _ = self._extract_geometry_for_onsite(pos, data)
                    eo_kwargs["pos"] = pos_orig
                edge_onsite_feats = self.edge_onsite(
                    sb, su, inter_src, inter_dst, edge_batch, batch_size,
                    **eo_kwargs
                )

            # Virtual node onsite: K learned attention heads compare bound vs unbound
            virtual_onsite_feats = None
            if self.virtual_onsites:
                parts_v = []
                for stream_name, vmod in self.virtual_onsites.items():
                    if stream_name == "all":
                        sb_v = h_bound[:, :self.n_scalar_channels]
                        su_v = h_unbound[:, :self.n_scalar_channels]
                        b_idx = batch_original
                    elif stream_name == "ligand":
                        sb_v = h_bound[ligand_mask, :self.n_scalar_channels]
                        su_v = h_unbound[ligand_mask, :self.n_scalar_channels]
                        b_idx = batch_original[ligand_mask]
                    elif stream_name == "pocket":
                        sb_v = h_bound[~ligand_mask, :self.n_scalar_channels]
                        su_v = h_unbound[~ligand_mask, :self.n_scalar_channels]
                        b_idx = batch_original[~ligand_mask]
                    else:
                        raise ValueError(f"Unknown virtual pool stream '{stream_name}'")
                    parts_v.append(vmod(sb_v, su_v, b_idx, batch_size))
                virtual_onsite_feats = torch.cat(parts_v, dim=-1)

            # Pool each stream (skip if node_readout disabled)
            parts = []
            if self.cfg.node_readout:
                # Apply optional projection before pooling
                if self.cfg.pool_delta:
                    node_scalars_ub = self.node_proj(h_unbound)
                    ns_pool = node_scalars - node_scalars_ub
                else:
                    ns_pool = node_scalars
                ns_pool = self.node_pool_proj(ns_pool) if self.node_pool_proj is not None else ns_pool
                stream_results = [
                    self._pool_stream(s, ns_pool, onsite_feats,
                                      batch_original, batch_size, ligand_mask)
                    for s in self.pool_streams
                ]
                parts.extend(stream_results)
                # Optionally also pool unbound node features
                if self.cfg.pool_unbound:
                    if not self.cfg.pool_delta:
                        node_scalars_ub = self.node_proj(h_unbound)
                    ns_ub = self.node_pool_proj(node_scalars_ub) if self.node_pool_proj is not None else node_scalars_ub
                    stream_results_ub = [
                        self._pool_stream(s, ns_ub, None,
                                          batch_original, batch_size, ligand_mask)
                        for s in self.pool_streams
                    ]
                    parts.extend(stream_results_ub)

            # Append edge onsite (single vector per graph, not per-stream)
            if edge_onsite_feats is not None:
                parts.append(edge_onsite_feats)

            # Append edge state pool (direct readout of contact geometry)
            if self.cfg.edge_readout and e_bound is not None:
                if self.cfg.edge_readout_mode == "delta":
                    edge_pool = scatter_sum(e_bound - e_unbound, _pe_eb, dim=0, dim_size=batch_size)
                else:  # "bound"
                    edge_pool = scatter_sum(e_bound, _pe_eb, dim=0, dim_size=batch_size)
                parts.append(edge_pool)

            # Append virtual node onsite (single vector per graph)
            if virtual_onsite_feats is not None:
                parts.append(virtual_onsite_feats)

            # Append virtual edge onsite (K virtual nodes attend to edge states)
            if self.virtual_edge_onsite is not None:
                virtual_edge_onsite_feats = self.virtual_edge_onsite(
                    e_bound, e_unbound, _pe_eb, batch_size)
                parts.append(virtual_edge_onsite_feats)

            # Edge global SH: angular descriptors of contact geometry
            pos_orig = None
            if self.edge_global_sh is not None and e_bound is not None:
                # _pe_src, _pe_dst are inter-molecular edge endpoints in N-node space
                # _pe_eb is the batch index per edge
                # Compute edge direction vectors: protein→ligand normalized
                if pos_orig is None:
                    pos_orig, _, _, _ = self._extract_geometry_for_onsite(pos, data)
                edge_midpoints = (pos_orig[_pe_src] + pos_orig[_pe_dst]) / 2.0
                # Compute ligand centroids per graph
                lig_pos = pos_orig[ligand_mask]
                lig_batch = batch_original[ligand_mask]
                lig_centroid = scatter_mean(lig_pos, lig_batch, dim=0, dim_size=batch_size)
                # Direction from ligand centroid to edge midpoint
                edge_rel = edge_midpoints - lig_centroid[_pe_eb]
                # Use edge states as features; create fake "all pocket" mask for GlobalSHPooling
                n_edges = e_bound.size(0)
                edge_fake_lig_mask = torch.zeros(n_edges, dtype=torch.bool, device=e_bound.device)
                # GlobalSHPooling expects: h_bound[N], h_unbound[N], pos[N], batch[N], ligand_mask[N]
                # We pass edge states as node features, edge_rel as positions, all marked as "pocket"
                # Need to prepend dummy ligand nodes for centroid computation
                # Simpler: add one dummy ligand node per graph with centroid position
                dummy_h = torch.zeros(batch_size, e_bound.size(-1), device=e_bound.device)
                all_h_b = torch.cat([dummy_h, e_bound], dim=0)
                all_h_u = torch.cat([dummy_h, e_unbound], dim=0)
                all_pos = torch.cat([lig_centroid, edge_midpoints], dim=0)
                dummy_batch = torch.arange(batch_size, device=e_bound.device)
                all_batch = torch.cat([dummy_batch, _pe_eb], dim=0)
                all_lig_mask = torch.zeros(batch_size + n_edges, dtype=torch.bool, device=e_bound.device)
                all_lig_mask[:batch_size] = True  # dummy ligand nodes

                edge_sh_feats = self.edge_global_sh(
                    all_h_b, all_h_u, all_pos, all_batch, all_lig_mask, batch_size)
                parts.append(edge_sh_feats)

            # Global SH pooling (angular descriptors from pocket shell)
            if (
                self.global_sh is not None
                or self.residue_global_sh is not None
                or self.msr is not None
            ):
                if pos_orig is None:
                    pos_orig, _, _, _ = self._extract_geometry_for_onsite(pos, data)
            # MSR readout (no SH; K parallel signed radial filters)
            if self.msr is not None:
                msr_feats = self.msr(
                    h_bound[:, :self.n_scalar_channels],
                    h_unbound[:, :self.n_scalar_channels],
                    pos_orig, batch_original, ligand_mask, batch_size,
                )
                parts.append(msr_feats)
            if self.global_sh is not None:
                # Broadcast ATOMICA graph embedding to all atoms if enabled
                atomica_per_atom = None
                if self.cfg.global_sh_atomica and hasattr(data, 'esm_embedding'):
                    atomica_per_atom = data.esm_embedding[batch_original]  # [N, 32]
                global_sh_feats = self.global_sh(
                    h_bound[:, :self.n_scalar_channels],
                    h_unbound[:, :self.n_scalar_channels],
                    pos_orig, batch_original, ligand_mask, batch_size,
                    esm_residue_features=atomica_per_atom,
                )
                parts.append(global_sh_feats)

            # Residue-level processing (shared residue_id extraction)
            _need_residue = self.residue_onsite is not None or self.residue_global_sh is not None
            if _need_residue:
                residue_id = self._extract_residue_ids(data)

                if self.residue_onsite is not None:
                    # Ensure pos_orig is available for reembed_cg mode
                    if not hasattr(self, '_residue_pos_orig') or pos_orig is None:
                        if self.global_sh is None and self.residue_global_sh is None:
                            pos_orig, _, _, _ = self._extract_geometry_for_onsite(pos, data)
                    residue_onsite_feats = self.residue_onsite(
                        h_bound[:, :self.n_scalar_channels],
                        h_unbound[:, :self.n_scalar_channels],
                        residue_id, batch_original, batch_size,
                        pos=pos_orig,
                    )
                    parts.append(residue_onsite_feats)

                if self.residue_global_sh is not None:
                    # Pool atoms → residues: positions and features
                    device = residue_id.device
                    max_res = scatter(residue_id, batch_original, dim=0,
                                      dim_size=batch_size, reduce='max')
                    cumsum = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
                    cumsum[1:] = torch.cumsum(max_res + 1, dim=0)
                    global_rid = residue_id + cumsum[batch_original]
                    num_residues = int(cumsum[-1].item())

                    h_b_scalars = h_bound[:, :self.n_scalar_channels]
                    h_u_scalars = h_unbound[:, :self.n_scalar_channels]

                    # Atom → residue pooling (mean or attention-weighted)
                    if self.residue_pool_attn_mlp is not None:
                        # Learn per-atom importance within each residue
                        attn_logits_b = self.residue_pool_attn_mlp(h_b_scalars).squeeze(-1)
                        attn_logits_u = self.residue_pool_attn_mlp(h_u_scalars).squeeze(-1)
                        # Per-residue softmax
                        attn_max_b = scatter(attn_logits_b, global_rid, dim=0,
                                             dim_size=num_residues, reduce='max')
                        attn_b = (attn_logits_b - attn_max_b[global_rid]).exp()
                        attn_Z_b = scatter_sum(attn_b, global_rid, dim=0, dim_size=num_residues)
                        attn_b = attn_b / (attn_Z_b[global_rid] + 1e-8)
                        attn_max_u = scatter(attn_logits_u, global_rid, dim=0,
                                             dim_size=num_residues, reduce='max')
                        attn_u = (attn_logits_u - attn_max_u[global_rid]).exp()
                        attn_Z_u = scatter_sum(attn_u, global_rid, dim=0, dim_size=num_residues)
                        attn_u = attn_u / (attn_Z_u[global_rid] + 1e-8)
                        h_b_res = scatter_sum(attn_b.unsqueeze(-1) * h_b_scalars,
                                              global_rid, dim=0, dim_size=num_residues)
                        h_u_res = scatter_sum(attn_u.unsqueeze(-1) * h_u_scalars,
                                              global_rid, dim=0, dim_size=num_residues)
                    else:
                        h_b_res = scatter_mean(h_b_scalars, global_rid, dim=0, dim_size=num_residues)
                        h_u_res = scatter_mean(h_u_scalars, global_rid, dim=0, dim_size=num_residues)

                    if self.residue_sh_pre_mlp is not None:
                        h_b_res = self.residue_sh_pre_mlp(h_b_res)
                        h_u_res = self.residue_sh_pre_mlp(h_u_res)

                    pos_res = scatter_mean(pos_orig, global_rid, dim=0, dim_size=num_residues)

                    # Build residue-level batch index and ligand mask
                    batch_res = torch.zeros(num_residues, dtype=torch.long, device=device)
                    batch_res.scatter_(0, global_rid, batch_original)

                    # Cross-residue self-attention (after pooling, before SH)
                    if self.residue_cross_attn is not None:
                        from torch_geometric.utils import to_dense_batch
                        h_b_dense, mask_b = to_dense_batch(h_b_res, batch_res)
                        h_u_dense, mask_u = to_dense_batch(h_u_res, batch_res)
                        h_b_dense = self.residue_cross_attn(
                            h_b_dense, src_key_padding_mask=~mask_b)
                        h_u_dense = self.residue_cross_attn(
                            h_u_dense, src_key_padding_mask=~mask_u)
                        h_b_res = h_b_dense[mask_b]
                        h_u_res = h_u_dense[mask_u]
                    lig_mask_res = torch.zeros(num_residues, dtype=torch.bool, device=device)
                    lig_rid = global_rid[ligand_mask]
                    lig_mask_res[lig_rid.unique()] = True

                    res_sh_feats = self.residue_global_sh(
                        h_b_res, h_u_res, pos_res, batch_res, lig_mask_res, batch_size)
                    parts.append(res_sh_feats)

            # Readout fusion or simple concatenation
            if self.readout_fusion is not None:
                pooled = self.readout_fusion(parts)
            else:
                pooled = torch.cat(parts, dim=-1)
        else:
            if self.jk_readout:
                jk_input = torch.cat(layer_scalars, dim=-1)
                node_scalars = self.node_proj(jk_input)
            else:
                node_scalars = self.node_proj(x)
            batch_size = int(data.batch.max().item()) + 1

            # Build ligand mask if needed
            ligand_mask = None
            if self._needs_ligand_mask:
                ligand_mask = self._build_ligand_mask(data.batch, data.num_ligand_atoms)

            # Pool each stream (no onsite in 1-copy mode)
            stream_results = [
                self._pool_stream(s, node_scalars, None,
                                  data.batch, batch_size, ligand_mask)
                for s in self.pool_streams
            ]
            pooled = torch.cat(stream_results, dim=-1)

        # ESM readout fusion: append projected precomputed embeddings
        if self.esm_proj is not None and hasattr(data, 'esm_embedding'):
            esm_feats = self.esm_proj(data.esm_embedding)  # [B, esm_readout_dim]
            pooled = torch.cat([pooled, esm_feats], dim=-1)

        # 6. Predict
        if self.head_bn is not None:
            pooled = self.head_bn(pooled)
        pred = self.head(pooled).squeeze(-1)

        if self.cfg.output_clamp > 0:
            pred = pred.clamp(-self.cfg.output_clamp, self.cfg.output_clamp)

        return pred, {}


def create_model(cfg: ModelConfig) -> EquivariantMPNN:
    return EquivariantMPNN(cfg)
