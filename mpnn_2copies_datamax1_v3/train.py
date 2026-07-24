#!/usr/bin/env python3
"""
Training script for MPNNv2 with GIGN features.

Supports two data formats (auto-detected):
- gign_exact: 35D nodes, no edge_attr, edge_index_intra/edge_index_inter naming
- gign_40d: 40D nodes, 21D edge_attr, edge_index/inter_edge_index naming
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "models"))

from model import ModelConfig, EquivariantMPNN


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021).

    Wraps a base optimizer (e.g. AdamW) with a two-step update that seeks
    flatter minima for better generalisation.  Usage::

        optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=2e-4)
        # training loop uses train_epoch(..., sam=True)
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)           # climb to the worst point
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # restore original weights
        self.base_optimizer.step()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


class CSVWriter:
    """Simple CSV logger."""
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        if write_header:
            self.writer.writeheader()

    def write(self, row: dict):
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of weights updated as:
        θ_shadow = decay * θ_shadow + (1 - decay) * θ_current

    Usage: call update() after each optimizer step, apply()/restore()
    around evaluation to swap in smoothed weights.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        """Swap in EMA weights for evaluation."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights after evaluation."""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


def train_val_split(data_list: List, val_fraction: float, seed: int) -> Tuple[List, List]:
    rng = random.Random(seed)
    indices = list(range(len(data_list)))
    rng.shuffle(indices)
    n_val = int(len(data_list) * val_fraction)
    val_indices = set(indices[:n_val])
    train_data = [data_list[i] for i in range(len(data_list)) if i not in val_indices]
    val_data = [data_list[i] for i in val_indices]
    return train_data, val_data


def _filter_isolated_nodes(data, use_inter_edges):
    """Remove nodes with zero edges from a graph.

    Considers intra edges always. Also considers inter edges if use_inter_edges=True.
    Re-indexes edges and updates num_ligand_atoms.
    """
    N = data.x.size(0)
    nlig = data.num_ligand_atoms

    # Determine connected nodes
    connected = set()
    ei_intra = data.edge_index_intra
    if ei_intra.size(1) > 0:
        connected.update(ei_intra[0].tolist())
        connected.update(ei_intra[1].tolist())
    if use_inter_edges and hasattr(data, 'edge_index_inter'):
        ei_inter = data.edge_index_inter
        if ei_inter.size(1) > 0:
            connected.update(ei_inter[0].tolist())
            connected.update(ei_inter[1].tolist())

    if len(connected) == N:
        return data  # nothing to filter

    # Build keep mask and old->new index mapping
    keep = sorted(connected)
    keep_mask = torch.zeros(N, dtype=torch.bool)
    keep_mask[keep] = True
    old_to_new = torch.full((N,), -1, dtype=torch.long)
    old_to_new[keep] = torch.arange(len(keep))

    # Filter node tensors
    data.x = data.x[keep_mask]
    data.pos = data.pos[keep_mask]
    if hasattr(data, 'x_ext'):
        data.x_ext = data.x_ext[keep_mask]
    if hasattr(data, 'x_bfactor'):
        data.x_bfactor = data.x_bfactor[keep_mask]
    if hasattr(data, 'aa_type'):
        data.aa_type = data.aa_type[keep_mask]
    if hasattr(data, 'residue_id'):
        data.residue_id = data.residue_id[keep_mask]
    if hasattr(data, 'residue_lm_embedding'):
        data.residue_lm_embedding = data.residue_lm_embedding[keep_mask]

    # Re-index edges
    data.edge_index_intra = old_to_new[data.edge_index_intra]
    if hasattr(data, 'edge_index_inter') and data.edge_index_inter.size(1) > 0:
        data.edge_index_inter = old_to_new[data.edge_index_inter]

    # Update num_ligand_atoms (count kept ligand nodes, i.e. original indices < nlig)
    data.num_ligand_atoms = int(keep_mask[:nlig].sum().item())

    return data


class PrepareData(BaseTransform):
    """Auto-detecting transform for both gign_exact and gign_40d data formats."""

    def __init__(self, use_inter_edges: bool = True, drop_isolated_nodes: bool = False):
        self.use_inter_edges = use_inter_edges
        self.drop_isolated_nodes = drop_isolated_nodes

    def forward(self, data):
        if self.drop_isolated_nodes and hasattr(data, 'edge_index_intra'):
            data = _filter_isolated_nodes(data, self.use_inter_edges)

        # Auto-detect format by checking for gign_exact naming
        is_gign_exact = hasattr(data, 'edge_index_intra')

        if is_gign_exact:
            return self._prepare_gign_exact(data)
        else:
            return self._prepare_gign_40d(data)

    def _prepare_gign_exact(self, data):
        """gign_exact: 35D nodes, edge_index_intra/edge_index_inter.
        Supports optional edge_attr_intra (21D) from augmented data."""
        # Rename edge_index_intra → edge_index
        data.edge_index = data.edge_index_intra
        del data.edge_index_intra

        # Check for precomputed edge features
        has_edge_attr = hasattr(data, 'edge_attr_intra') and data.edge_attr_intra is not None
        if has_edge_attr:
            data.edge_attr = data.edge_attr_intra
            del data.edge_attr_intra

        # Merge intermolecular edges
        if self.use_inter_edges and hasattr(data, 'edge_index_inter'):
            data.edge_index = torch.cat([data.edge_index, data.edge_index_inter], dim=1)
            if has_edge_attr:
                n_inter = data.edge_index_inter.size(1)
                inter_attr = torch.zeros(n_inter, data.edge_attr.size(1))
                if hasattr(data, 'edge_attr_inter') and data.edge_attr_inter is not None:
                    inter_attr[:, 11] = data.edge_attr_inter.squeeze()
                data.edge_attr = torch.cat([data.edge_attr, inter_attr], dim=0)

        # Compute edge_length (needed for RBF fallback)
        src, dst = data.edge_index
        data.edge_length = (data.pos[dst] - data.pos[src]).norm(dim=-1)

        if not has_edge_attr:
            data.edge_attr = None

        # Combine pos + features: [N, 3+35] = [N, 38]
        data.x = torch.cat([data.pos, data.x], dim=-1)

        return data

    def _prepare_gign_40d(self, data):
        """gign_40d: 40D nodes, 21D edge_attr, edge_index/inter_edge_index."""
        # Compute edge_length
        src, dst = data.edge_index
        data.edge_length = (data.pos[dst] - data.pos[src]).norm(dim=-1)

        # Combine pos + features: [N, 3+40] = [N, 43]
        data.x = torch.cat([data.pos, data.x], dim=-1)

        # Merge intermolecular edges
        if self.use_inter_edges and hasattr(data, 'inter_edge_index'):
            data.edge_index = torch.cat([data.edge_index, data.inter_edge_index], dim=1)
            inter_attr_padded = torch.zeros(data.inter_edge_attr.size(0), data.edge_attr.size(1))
            inter_attr_padded[:, 11] = data.inter_edge_attr.squeeze()
            data.edge_attr = torch.cat([data.edge_attr, inter_attr_padded], dim=0)
            src, dst = data.edge_index
            data.edge_length = (data.pos[dst] - data.pos[src]).norm(dim=-1)

        return data


class PrepareData_2copies(BaseTransform):
    """Transform that creates bound + unbound copies of each graph.

    Bound copy (nodes 0..N-1): sees intra + inter edges
    Unbound copy (nodes N..2N-1): sees intra edges only

    This enables onsite interactions: comparing a node's representation
    when it sees the binding partner vs when it doesn't.
    """

    def __init__(self, use_inter_edges: bool = True, drop_isolated_nodes: bool = False,
                 use_ext_features: bool = False, use_bfactor: bool = False,
                 ext_feature_mask: list = None):
        self.use_inter_edges = use_inter_edges
        self.drop_isolated_nodes = drop_isolated_nodes
        self.use_ext_features = use_ext_features
        self.use_bfactor = use_bfactor
        self.ext_feature_mask = ext_feature_mask  # e.g. [0,1,3,4] to select columns of x_ext

    def forward(self, data):
        assert hasattr(data, 'edge_index_intra'), "Only gign_exact format supported for 2-copy mode"
        if self.drop_isolated_nodes:
            data = _filter_isolated_nodes(data, self.use_inter_edges)

        # -- Concatenate optional extended features before duplication --
        if self.use_ext_features:
            if hasattr(data, 'x_ext'):
                ext = data.x_ext
            else:
                n_ext = 7
                ext = torch.zeros(data.x.size(0), n_ext)
            if self.ext_feature_mask is not None:
                assert all(0 <= i < ext.shape[1] for i in self.ext_feature_mask), \
                    f"ext_feature_mask={self.ext_feature_mask} out of bounds for ext dim={ext.shape[1]}"
                ext = ext[:, self.ext_feature_mask]
            data.x = torch.cat([data.x, ext], dim=-1)
        if self.use_bfactor:
            if hasattr(data, 'x_bfactor'):
                bf = data.x_bfactor
                if bf.dim() == 1:
                    bf = bf.unsqueeze(-1)
            else:
                bf = torch.zeros(data.x.size(0), 1)
            data.x = torch.cat([data.x, bf], dim=-1)

        N = data.x.size(0)
        edge_index_intra = data.edge_index_intra

        # -- Bound copy (0..N-1): intra + inter edges --
        bound_edges = edge_index_intra
        if self.use_inter_edges and hasattr(data, 'edge_index_inter'):
            bound_edges = torch.cat([bound_edges, data.edge_index_inter], dim=1)

        # -- Unbound copy (N..2N-1): intra edges only, shifted by N --
        unbound_edges = edge_index_intra + N

        # -- Merged edges --
        data.edge_index = torch.cat([bound_edges, unbound_edges], dim=1)

        # -- Duplicate node features and positions --
        pos_ub = data.pos_unbound if hasattr(data, 'pos_unbound') else data.pos
        pos_new = torch.cat([data.pos, pos_ub], dim=0)
        x_new = torch.cat([data.x, data.x], dim=0)
        data.x = torch.cat([pos_new, x_new], dim=-1)  # [2N, 3+D]
        data.pos = pos_new

        # -- Edge lengths (for RBF computation in forward) --
        src, dst = data.edge_index
        data.edge_length = (data.pos[dst] - data.pos[src]).norm(dim=-1)

        # -- Metadata --
        data.num_nodes_original = N
        data.edge_attr = None

        # -- Residue IDs (fallback: ligand=one residue, each pocket atom=own residue) --
        if not hasattr(data, 'residue_id'):
            num_lig = data.num_ligand_atoms if isinstance(data.num_ligand_atoms, int) \
                      else data.num_ligand_atoms.item()
            residue_id = torch.zeros(N, dtype=torch.long)
            residue_id[num_lig:] = torch.arange(1, N - num_lig + 1)
            data.residue_id = residue_id
        # Duplicate for 2-copy mode (both copies share same residue assignment)
        data.residue_id = torch.cat([data.residue_id, data.residue_id], dim=0)
        # Duplicate per-residue LM embeddings if present
        if hasattr(data, 'residue_lm_embedding'):
            data.residue_lm_embedding = torch.cat([data.residue_lm_embedding, data.residue_lm_embedding], dim=0)
        # Duplicate amino acid type if present
        if hasattr(data, 'aa_type'):
            data.aa_type = torch.cat([data.aa_type, data.aa_type], dim=0)
        # Duplicate physicochemical features for residue-level input
        if hasattr(data, 'x_physchem'):
            data.x_physchem = torch.cat([data.x_physchem, data.x_physchem], dim=0)

        # -- Clean up --
        del data.edge_index_intra
        if hasattr(data, 'edge_index_inter'):
            del data.edge_index_inter
        if hasattr(data, 'edge_attr_intra'):
            del data.edge_attr_intra
        if hasattr(data, 'edge_attr_inter'):
            del data.edge_attr_inter
        if hasattr(data, 'pos_unbound'):
            del data.pos_unbound
        # Remove stale N-sized tensors that weren't consumed (avoid misaligned collation)
        if hasattr(data, 'x_ext'):
            del data.x_ext
        if hasattr(data, 'x_bfactor'):
            del data.x_bfactor

        return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base (override wins). Returns new dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Support parent config inheritance: resolve relative to this config's dir
    parent_path = cfg.pop("parent", None)
    if parent_path is not None:
        parent_path = Path(config_path).parent / parent_path
        parent_cfg = load_config(str(parent_path))
        cfg = _deep_merge(parent_cfg, cfg)
    return cfg


def config_to_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    return ModelConfig(
        use_2copies=data_cfg.get("use_2copies", True),
        irreps_in=model_cfg.get("irreps_in", "40x0e"),
        irreps_hidden=model_cfg.get("irreps_hidden", "64x0e + 16x1o + 8x2e"),
        num_layers=model_cfg.get("num_layers", 5),
        scalars=model_cfg.get("scalars", 64),
        backbone=model_cfg.get("backbone", "tpconv"),
        lmax=model_cfg.get("lmax", 2),
        envelope_cutoff=model_cfg.get("envelope_cutoff", 0.0),
        r_max=model_cfg.get("r_max", 5.0),
        num_basis=model_cfg.get("num_basis", 14),
        use_precomputed_edges=model_cfg.get("use_precomputed_edges", True),
        precomputed_edge_dim=model_cfg.get("precomputed_edge_dim", 21),
        cross_object_embed_dim=model_cfg.get("cross_object_embed_dim", 0),
        use_dst_features=model_cfg.get("use_dst_features", False),
        num_atom_types=model_cfg.get("num_atom_types", 17),
        input_atom_embed_dim=model_cfg.get("input_atom_embed_dim", 0),
        aa_embed_dim=model_cfg.get("aa_embed_dim", 0),
        num_aa_types=model_cfg.get("num_aa_types", 22),
        pocket_scalar_only=model_cfg.get("pocket_scalar_only", False),
        residue_input_dim=model_cfg.get("residue_input_dim", 0),
        radial_hidden=model_cfg.get("radial_hidden", 128),
        radial_layers=model_cfg.get("radial_layers", 2),
        tp_mode=model_cfg.get("tp_mode", "full"),
        tp_sparsity=model_cfg.get("tp_sparsity", 0.25),
        self_cg=model_cfg.get("self_cg", "none"),
        use_norm=model_cfg.get("use_norm", True),
        node_conditioned_radial=model_cfg.get("node_conditioned_radial", True),
        aggr=model_cfg.get("aggr", "sum"),
        pool_mode=model_cfg.get("pool_mode", "sum"),
        head_hidden=model_cfg.get("head_hidden", 128),
        head_layers=model_cfg.get("head_layers", 2),
        dropout=model_cfg.get("dropout", 0.0),
        output_clamp=model_cfg.get("output_clamp", 0.0),
        head_norm=model_cfg.get("head_norm", "none"),
        num_scalar_features=model_cfg.get("num_scalar_features", 23),
        onsite_mode=model_cfg.get("onsite_mode", "none"),
        onsite_dim=model_cfg.get("onsite_dim", 64),
        pool_streams=model_cfg.get("pool_streams", ["all"]),
        onsite_reembed_lmax=model_cfg.get("onsite_reembed_lmax", 0),
        onsite_cg_mode=model_cfg.get("onsite_cg_mode", "uvu"),
        reembed_attention=model_cfg.get("reembed_attention", False),
        contraction_refs=model_cfg.get("contraction_refs", "unbound"),
        reembed_iterations=model_cfg.get("reembed_iterations", 1),
        dual_cg=model_cfg.get("dual_cg", False),
        degree2_invariants=model_cfg.get("degree2_invariants", False),
        multiscale_reembed=model_cfg.get("multiscale_reembed", False),
        jk_readout=model_cfg.get("jk_readout", False),
        edge_onsite_mode=model_cfg.get("edge_onsite_mode", "none"),
        edge_onsite_dim=model_cfg.get("edge_onsite_dim", 64),
        edge_hidden_dim=model_cfg.get("edge_hidden_dim", 128),
        persistent_edge_state=model_cfg.get("persistent_edge_state", False),
        edge_state_dim=model_cfg.get("edge_state_dim", 64),
        edge_feedback=model_cfg.get("edge_feedback", False),
        virtual_onsite_mode=model_cfg.get("virtual_onsite_mode", "none"),
        num_virtual_nodes=model_cfg.get("num_virtual_nodes", 4),
        virtual_node_dim=model_cfg.get("virtual_node_dim", 64),
        virtual_pool_streams=model_cfg.get("virtual_pool_streams", ["all"]),
        virtual_edge_onsite_mode=model_cfg.get("virtual_edge_onsite_mode", "none"),
        virtual_dropout=model_cfg.get("virtual_dropout", 0.0),
        node_readout=model_cfg.get("node_readout", True),
        edge_readout=model_cfg.get("edge_readout", False),
        edge_onsite_reembed_lmax=model_cfg.get("edge_onsite_reembed_lmax", 1),
        edge_onsite_cg_mode=model_cfg.get("edge_onsite_cg_mode", "uuu"),
        readout_fusion=model_cfg.get("readout_fusion", "none"),
        readout_fusion_dim=model_cfg.get("readout_fusion_dim", 128),
        feature_mask=model_cfg.get("feature_mask", "none"),
        feature_mask_target=model_cfg.get("feature_mask_target", "both"),
        global_sh_pooling=model_cfg.get("global_sh_pooling", False),
        global_sh_channels=model_cfg.get("global_sh_channels", 16),
        global_sh_lmax=model_cfg.get("global_sh_lmax", 2),
        global_sh_out_dim=model_cfg.get("global_sh_out_dim", 64),
        global_sh_separate_proj=model_cfg.get("global_sh_separate_proj", False),
        global_sh_mode=model_cfg.get("global_sh_mode", "cross"),
        global_sh_head=model_cfg.get("global_sh_head", "none"),
        global_sh_n_heads=model_cfg.get("global_sh_n_heads", 1),
        residue_global_sh_pooling=model_cfg.get("residue_global_sh_pooling", False),
        residue_global_sh_channels=model_cfg.get("residue_global_sh_channels", 16),
        residue_global_sh_lmax=model_cfg.get("residue_global_sh_lmax", 2),
        residue_global_sh_out_dim=model_cfg.get("residue_global_sh_out_dim", 64),
        residue_global_sh_separate_proj=model_cfg.get("residue_global_sh_separate_proj", False),
        residue_global_sh_head=model_cfg.get("residue_global_sh_head", "none"),
        residue_global_sh_pre_mlp=model_cfg.get("residue_global_sh_pre_mlp", False),
        residue_onsite_mode=model_cfg.get("residue_onsite_mode", "none"),
        residue_onsite_dim=model_cfg.get("residue_onsite_dim", 64),
        residue_onsite_reembed_lmax=model_cfg.get("residue_onsite_reembed_lmax", 1),
        residue_agg=model_cfg.get("residue_agg", "mean"),
        pool_unbound=model_cfg.get("pool_unbound", False),
        pool_delta=model_cfg.get("pool_delta", False),
        node_pool_proj_dim=model_cfg.get("node_pool_proj_dim", 0),
        pool_attention=model_cfg.get("pool_attention", False),
        onsite_self_attn=model_cfg.get("onsite_self_attn", False),
        onsite_self_attn_heads=model_cfg.get("onsite_self_attn_heads", 4),
        onsite_reembed_proj_dim=model_cfg.get("onsite_reembed_proj_dim", 0),
        onsite_out_proj_bias=model_cfg.get("onsite_out_proj_bias", True),
        edge_readout_mode=model_cfg.get("edge_readout_mode", "bound"),
        esm_readout_dim=model_cfg.get("esm_readout_dim", 0),
        esm_input_dim=model_cfg.get("esm_input_dim", 1280),
        virtual_node_mp=model_cfg.get("virtual_node_mp", False),
        context_embed_dim=model_cfg.get("context_embed_dim", 0),
        residue_pool_attn=model_cfg.get("residue_pool_attn", False),
        residue_cross_attn=model_cfg.get("residue_cross_attn", False),
        residue_cross_attn_heads=model_cfg.get("residue_cross_attn_heads", 4),
        edge_global_sh=model_cfg.get("edge_global_sh", False),
        edge_global_sh_channels=model_cfg.get("edge_global_sh_channels", 16),
        edge_global_sh_lmax=model_cfg.get("edge_global_sh_lmax", 2),
        edge_global_sh_out_dim=model_cfg.get("edge_global_sh_out_dim", 64),
        edge_global_sh_residue_pool=model_cfg.get("edge_global_sh_residue_pool", False),
        global_sh_atomica=model_cfg.get("global_sh_atomica", False),
        global_sh_radial_form=model_cfg.get("global_sh_radial_form", "learned"),
    )


def smoke_test(model: nn.Module, loader: DataLoader, device: torch.device) -> bool:
    print("\n" + "=" * 50)
    print("SMOKE TEST")
    print("=" * 50)

    model.eval()
    batch = next(iter(loader))
    batch = batch.to(device)

    with torch.no_grad():
        pred, aux = model(batch)

    checks_passed = True

    if torch.isnan(pred).any():
        print("[FAIL] NaN in predictions")
        checks_passed = False
    else:
        print("[PASS] No NaN in predictions")

    if pred.shape != (batch.num_graphs,):
        print(f"[FAIL] Wrong shape: {pred.shape}")
        checks_passed = False
    else:
        print(f"[PASS] Shape correct: {pred.shape}")

    if not torch.isfinite(pred).all():
        print("[FAIL] Non-finite values")
        checks_passed = False
    else:
        print("[PASS] All finite")

    print("=" * 50)
    print("PASSED" if checks_passed else "FAILED")
    print("=" * 50 + "\n")

    return checks_passed


def train_epoch(model, loader, optimizer, criterion, device, epoch, max_grad_norm=10.0, ema=None,
                coord_noise_std=0.0):
    model.train()
    total_loss = 0.0
    max_gnorm = 0.0
    num_batches = 0
    use_sam = isinstance(optimizer, SAM)

    for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        batch = batch.to(device)
        if coord_noise_std > 0:
            batch = batch.clone()
            batch.x = batch.x.clone()
            batch.x[:, :3] = batch.x[:, :3] + torch.randn_like(batch.x[:, :3]) * coord_noise_std
        optimizer.zero_grad()
        pred, _ = model(batch)
        loss = criterion(pred, batch.y.squeeze())
        loss.backward()
        if max_grad_norm > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
            if gnorm > max_gnorm:
                max_gnorm = gnorm

        if use_sam:
            optimizer.first_step()
            optimizer.zero_grad()
            pred2, _ = model(batch)
            loss2 = criterion(pred2, batch.y.squeeze())
            loss2.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.second_step()
        else:
            optimizer.step()

        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches, max_gnorm


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        pred, _ = model(batch)
        all_preds.append(pred.cpu().view(-1))
        all_targets.append(batch.y.cpu().view(-1))

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    mask = np.isfinite(preds) & np.isfinite(targets)
    preds, targets = preds[mask], targets[mask]

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    pearson = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0.0
    spearman = scipy.stats.spearmanr(preds, targets)[0] if len(preds) > 1 else 0.0

    return {"rmse": rmse, "mae": mae, "pearson": pearson, "spearman": spearman}, preds, targets


def save_plots(output_dir, epoch, test_preds, test_targets, test_rmses, train_losses, lrs, test_corr):
    preds_dir = output_dir / "preds"
    preds_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(test_targets, test_preds, alpha=0.7, edgecolors='black')
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    min_val, max_val = min(test_targets.min(), test_preds.min()), max(test_targets.max(), test_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.title(f"Pearson = {test_corr:.4f}")
    plt.savefig(preds_dir / f"preds_epoch{epoch}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(test_rmses, linewidth=2)
    plt.title("Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "test_rmse.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train MPNNv2 with GIGN features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke_test_only", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to experiment dir to resume from (loads latest_checkpoint.pt)")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to checkpoint (.pt) to load weights from (fresh optimizer/scheduler)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (overrides config)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze all params except onsite, edge_onsite, and head")
    parser.add_argument("--backbone_lr_scale", type=float, default=1.0,
                        help="Scale factor for backbone LR (e.g. 0.1 = 10x lower than head/onsite)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = config_to_model_config(cfg)
    train_cfg = cfg.get("train", {})
    deterministic = train_cfg.get("deterministic", False)
    set_seed(args.seed, deterministic=deterministic)
    data_cfg = cfg.get("data", {})
    output_cfg = cfg.get("output", {})

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _wall_start = time.monotonic()

    # Load data
    train_path = data_cfg.get("train_path")
    test_path = data_cfg.get("test_path")
    if not train_path or not test_path:
        raise ValueError("data.train_path and data.test_path must be set in config")

    print(f"Loading train data: {train_path}")
    train_data = torch.load(train_path, weights_only=False)

    print(f"Loading test data: {test_path}")
    test_data = torch.load(test_path, weights_only=False)

    test2_path = data_cfg.get("test2_path")
    # Auto-detect indep test set for cleansplit configs
    if not test2_path and "cleansplit_casf2016_5A" in test_path:
        candidate = test_path.replace("cleansplit_casf2016_5A", "cleansplit_casf2016_indep_5A")
        if Path(candidate).exists():
            test2_path = candidate
            print(f"Auto-detected indep test set: {test2_path}")
    test2_data = None
    if test2_path:
        print(f"Loading test2 data: {test2_path}")
        test2_data = torch.load(test2_path, weights_only=False)

    # Split train/val (or load presplit val)
    if data_cfg.get("use_seed_split") and "val_path" in data_cfg:
        # Merge train+val, re-split with current seed for multi-seed validation
        val_path = data_cfg["val_path"]
        print(f"Seed-split mode: merging train + val ({val_path})")
        val_data = torch.load(val_path, weights_only=False)
        combined = train_data + val_data
        test_ids = {d.pdb_id for d in test_data}
        combined = [d for d in combined if d.pdb_id not in test_ids]
        train_data, val_data = train_val_split(combined, 0.1, args.seed)
        print(f"Re-split with seed={args.seed}: train={len(train_data)}, val={len(val_data)}")
    elif data_cfg.get("use_presplit") and "val_path" in data_cfg:
        val_path = data_cfg["val_path"]
        print(f"Loading presplit val data: {val_path}")
        val_data = torch.load(val_path, weights_only=False)
        # Remove val/test PDBs from train to prevent data leakage
        held_out_ids = {d.pdb_id for d in val_data} | {d.pdb_id for d in test_data}
        n_before = len(train_data)
        train_data = [d for d in train_data if d.pdb_id not in held_out_ids]
        if n_before > len(train_data):
            print(f"Removed {n_before - len(train_data)} train samples overlapping with val/test")
    else:
        train_data, val_data = train_val_split(train_data, 0.1, args.seed)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Prepare data: 1-copy (original) or 2-copy (bound + unbound) mode
    use_inter = data_cfg.get("use_inter_edges", True)
    use_2copies = data_cfg.get("use_2copies", False)
    drop_isolated = data_cfg.get("drop_isolated_nodes", False)
    use_ext_features = data_cfg.get("use_ext_features", False)
    use_bfactor = data_cfg.get("use_bfactor", False)
    ext_feature_mask = data_cfg.get("ext_feature_mask", None)
    if use_2copies:
        transform = PrepareData_2copies(
            use_inter_edges=use_inter, drop_isolated_nodes=drop_isolated,
            use_ext_features=use_ext_features, use_bfactor=use_bfactor,
            ext_feature_mask=ext_feature_mask,
        )
        print("Using 2-copy mode (bound + unbound)")
        if use_ext_features:
            mask_str = f" (mask={ext_feature_mask})" if ext_feature_mask else ""
            print(f"  Extended features enabled{mask_str}")
        if use_bfactor:
            print("  B-factor features enabled")
    else:
        transform = PrepareData(use_inter_edges=use_inter, drop_isolated_nodes=drop_isolated)
        print("Using 1-copy mode (standard)")
    if drop_isolated:
        print("Dropping isolated nodes (zero edges)")
    train_data = [transform(d) for d in tqdm(train_data, desc="Preparing train")]
    val_data = [transform(d) for d in tqdm(val_data, desc="Preparing val")]
    test_data = [transform(d) for d in tqdm(test_data, desc="Preparing test")]
    if test2_data is not None:
        test2_data = [transform(d) for d in tqdm(test2_data, desc="Preparing test2")]

    # Attach precomputed ESM embeddings (if configured)
    esm_path = data_cfg.get("esm_embeddings_path")
    if esm_path:
        print(f"Loading ESM embeddings: {esm_path}")
        esm_dict = torch.load(esm_path, map_location="cpu", weights_only=False)
        n_found, n_missing = 0, 0
        esm_dim = next(iter(esm_dict.values())).shape[0]
        zero_emb = torch.zeros(esm_dim)
        for dataset in [train_data, val_data, test_data] + ([test2_data] if test2_data else []):
            for d in dataset:
                emb = esm_dict.get(d.pdb_id)
                if emb is not None:
                    d.esm_embedding = emb.unsqueeze(0)  # [1, 1280] for batching
                    n_found += 1
                else:
                    d.esm_embedding = zero_emb.unsqueeze(0)
                    n_missing += 1
        print(f"  ESM: {n_found} found, {n_missing} missing (zero-filled)")
        del esm_dict

    # Attach per-residue language model embeddings (if configured)
    residue_lm_path = data_cfg.get("residue_lm_path")
    if residue_lm_path:
        print(f"Loading per-residue LM embeddings: {residue_lm_path}")
        residue_lm_dict = torch.load(residue_lm_path, map_location="cpu", weights_only=False)
        lm_dim = next(iter(residue_lm_dict.values())).shape[-1]
        n_found, n_missing = 0, 0
        for dataset in [train_data, val_data, test_data] + ([test2_data] if test2_data else []):
            for d in dataset:
                per_residue = residue_lm_dict.get(d.pdb_id)
                if per_residue is not None and hasattr(d, 'residue_id'):
                    rid = d.residue_id  # [N_atoms], int indices
                    # Clamp to valid range (some atoms may have residue_id beyond the sequence)
                    n_res = per_residue.shape[0]
                    rid_clamped = rid.clamp(0, n_res - 1)
                    d.residue_lm_embedding = per_residue[rid_clamped]  # [N_atoms, lm_dim]
                    n_found += 1
                else:
                    n_atoms = d.x.shape[0] if hasattr(d, 'x') else d.pos.shape[0]
                    d.residue_lm_embedding = torch.zeros(n_atoms, lm_dim)
                    n_missing += 1
        print(f"  Residue LM: {n_found} found, {n_missing} missing (zero-filled), dim={lm_dim}")
        del residue_lm_dict

    batch_size = train_cfg.get("batch_size", 32)
    loader_seed = train_cfg.get("loader_seed", args.seed)
    loader_generator = torch.Generator().manual_seed(loader_seed)
    def worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                              generator=loader_generator, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test2_loader = DataLoader(test2_data, batch_size=batch_size, shuffle=False, num_workers=4) if test2_data is not None else None

    # Model
    model = EquivariantMPNN(model_cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    # Load finetune weights (before EMA init so EMA starts from finetuned weights)
    if args.finetune:
        ft_path = Path(args.finetune)
        if not ft_path.exists():
            raise ValueError(f"Finetune checkpoint not found: {ft_path}")
        print(f"Loading finetune weights from: {ft_path}")
        ckpt = torch.load(ft_path, weights_only=False, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Filter out keys with shape mismatches (e.g. head changes when adding on-site)
        model_state = model.state_dict()
        shape_mismatch = []
        for k in list(state_dict.keys()):
            if k in model_state and state_dict[k].shape != model_state[k].shape:
                shape_mismatch.append(f"{k}: ckpt {state_dict[k].shape} vs model {model_state[k].shape}")
                del state_dict[k]
        if shape_mismatch:
            print(f"  Shape-mismatched keys (randomly initialized): {shape_mismatch}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (randomly initialized): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
        print("  Loaded model weights (fresh optimizer/scheduler)")

    # Freeze backbone (everything except onsite, edge_onsite, head)
    if args.freeze_backbone:
        trainable_prefixes = ("onsite.", "edge_onsite.", "head.")
        n_frozen = 0
        n_trainable = 0
        for name, p in model.named_parameters():
            if name.startswith(trainable_prefixes):
                p.requires_grad = True
                n_trainable += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        print(f"Backbone frozen: {n_frozen:,} params frozen, {n_trainable:,} trainable")

    # EMA (Exponential Moving Average)
    ema_decay = train_cfg.get("ema_decay", 0.0)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None
    if ema is not None:
        print(f"EMA enabled with decay={ema_decay}")

    if not smoke_test(model, train_loader, device):
        sys.exit(1)

    if args.smoke_test_only:
        sys.exit(0)

    # Training setup
    huber_delta = train_cfg.get("huber_delta", 0.0)
    if huber_delta > 0:
        criterion = nn.HuberLoss(delta=huber_delta)
        print(f"Using Huber loss (delta={huber_delta})")
    else:
        criterion = nn.MSELoss()
    base_lr = train_cfg.get("learning_rate", 0.001)
    wd = train_cfg.get("weight_decay", 0.0001)
    sam_rho = train_cfg.get("sam_rho", 0.0)

    # Differential LR: backbone gets scaled LR, new modules (onsite/edge_onsite/head) get full LR
    if args.backbone_lr_scale != 1.0:
        new_prefixes = ("onsite.", "edge_onsite.", "head.", "esm_proj.")
        backbone_params = []
        new_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(new_prefixes):
                new_params.append(p)
            else:
                backbone_params.append(p)
        backbone_lr = base_lr * args.backbone_lr_scale
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": new_params, "lr": base_lr},
        ]
        n_bb = sum(p.numel() for p in backbone_params)
        n_new = sum(p.numel() for p in new_params)
        print(f"Differential LR: backbone ({n_bb:,} params) lr={backbone_lr:.1e}, "
              f"new modules ({n_new:,} params) lr={base_lr:.1e}")
    else:
        param_groups = model.parameters()

    if sam_rho > 0:
        optimizer = SAM(param_groups, AdamW, rho=sam_rho,
                        lr=base_lr, weight_decay=wd)
        print(f"Using SAM optimizer (rho={sam_rho})")
    else:
        optimizer = AdamW(param_groups, lr=base_lr, weight_decay=wd)
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    epochs = train_cfg.get("epochs", 100)
    max_grad_norm = train_cfg.get("max_grad_norm", 10.0)
    eval_warmup = train_cfg.get("eval_warmup", 300)
    cosine_T_max = train_cfg.get("cosine_T_max", epochs - warmup_epochs)
    sched_optim = optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer
    cosine_scheduler = CosineAnnealingLR(sched_optim, T_max=cosine_T_max,
                                          eta_min=train_cfg.get("min_lr", 1e-5))
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(sched_optim, start_factor=1e-2, total_iters=warmup_epochs)
        scheduler = SequentialLR(sched_optim, [warmup_scheduler, cosine_scheduler],
                                 milestones=[warmup_epochs])
    else:
        scheduler = cosine_scheduler

    # Early stopping config
    patience = train_cfg.get("patience", 0)
    if args.patience is not None:
        patience = args.patience

    # Output directory
    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            raise ValueError(f"Resume directory does not exist: {output_dir}")
        # Prefer latest_checkpoint.pt (full state), fall back to best_model_test.pt (weights only)
        ckpt_path = output_dir / "latest_checkpoint.pt"
        if not ckpt_path.exists():
            ckpt_path = output_dir / "best_model_test.pt"
        if not ckpt_path.exists():
            raise ValueError(f"No latest_checkpoint.pt or best_model_test.pt in {output_dir}")
        print(f"Resuming from: {ckpt_path}")
    else:
        config_name = Path(args.config).stem
        base_dir = Path(output_cfg.get("dir", "mpnn_datamax1_v2/experiments"))
        config_dir = base_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        idx = len(list(config_dir.glob(f"{date_str}_*"))) + 1
        output_dir = config_dir / f"{date_str}_{idx}_s{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")

    # Save config (include seed + hostname + environment for reproducibility)
    cfg["seed"] = args.seed
    cfg["hostname"] = socket.gethostname()
    cfg["config_path"] = str(Path(args.config).resolve())
    cfg["start_time"] = datetime.now(timezone.utc).isoformat()

    # Git info
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip())
        cfg["git"] = {"commit": git_hash, "dirty": git_dirty}
    except Exception:
        cfg["git"] = {"commit": "unknown", "dirty": None}

    # GPU info
    if torch.cuda.is_available():
        gpu_idx = device.index if device.index is not None else 0
        cfg["gpu"] = torch.cuda.get_device_name(gpu_idx)

    # Data checksums (SHA256, first 16 hex chars for brevity)
    def _file_sha256_short(path, n=16):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:n]

    cfg["data_checksums"] = {}
    for key in ["train_path", "test_path", "val_path", "test2_path"]:
        p = data_cfg.get(key)
        if p and Path(p).exists():
            cfg["data_checksums"][key] = _file_sha256_short(p)

    cfg["environment"] = {
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }
    for pkg_name in ["torch_geometric", "e3nn", "torch_scatter", "torch_sparse", "torch_cluster"]:
        try:
            pkg = __import__(pkg_name)
            cfg["environment"][pkg_name] = str(pkg.__version__)
        except ImportError:
            pass
    if torch.cuda.is_available():
        cfg["environment"]["cuda"] = str(torch.version.cuda)
        cfg["environment"]["cudnn"] = int(torch.backends.cudnn.version())
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Archive source files for reproducibility
    source_dir = output_dir / "source"
    source_dir.mkdir(exist_ok=True)
    for src_file in [
        Path(__file__),
        Path(__file__).parent / "models" / "model.py",
        Path(__file__).parent / "models" / "onsite.py",
        Path(__file__).parent / "models" / "edge_onsite.py",
        Path(__file__).parent / "models" / "virtual_onsite.py",
        Path(__file__).parent / "models" / "readout_fusion.py",
        Path(__file__).parent / "models" / "global_sh.py",
        Path(__file__).parent / "models" / "msr.py",
        Path(__file__).parent.parent / "src" / "layers" / "tpconv.py",
        Path(__file__).parent.parent / "gign_exact_v1" / "preprocess_canonical.py",
        Path(__file__).parent.parent / "gign_exact_v1" / "preprocess.py",
    ]:
        if src_file.exists():
            shutil.copy2(src_file, source_dir / src_file.name)

    # Loggers
    fields = ["epoch", "rmse", "mae", "pearson", "spearman", "lr", "val_rmse", "val_pearson", "max_grad_norm"]
    if test2_loader is not None:
        fields += ["test2_rmse", "test2_mae", "test2_pearson", "test2_spearman"]
    test_logger = CSVWriter(str(output_dir / "test_metrics.csv"), fields)

    # Training loop
    best_val_rmse = float("inf")
    best_val_r = -float("inf")  # best val Pearson R
    _snapshot_milestones = list(range(600, 1300, 100))  # 600,700,...,1200
    best_test_rmse = float("inf")
    best_test_r = -float("inf")  # best test Pearson R (absolute best, any epoch)
    vg_best_test_rmse = float("inf")  # val-gated: best test among val-improving epochs
    epochs_no_improve = 0
    test_rmses, train_losses, lrs = [], [], []
    best_val_info = {}
    best_test_info = {}
    best_val_r_info = {}
    best_test_r_info = {}
    vg_best_test_info = {}
    start_epoch = 0

    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        resume_sd = ckpt["model_state_dict"]
        model_sd = model.state_dict()
        # Filter shape mismatches for architecture changes (e.g. adding ESM projection)
        skip = []
        for k in list(resume_sd.keys()):
            if k in model_sd and resume_sd[k].shape != model_sd[k].shape:
                skip.append(k)
                del resume_sd[k]
        missing, unexpected = model.load_state_dict(resume_sd, strict=False)
        if skip:
            print(f"  Resume: shape-mismatched keys (random init): {skip}")
        if missing:
            print(f"  Resume: missing keys (random init): {missing}")
        start_epoch = ckpt["epoch"]

        if "optimizer_state_dict" in ckpt:
            # Full checkpoint (latest_checkpoint.pt)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_val_rmse = ckpt["best_val_rmse"]
            best_test_rmse = ckpt["best_test_rmse"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            best_val_info = ckpt.get("best_val_info", {})
            best_test_info = ckpt.get("best_test_info", {})
            best_val_r = ckpt.get("best_val_r", -float("inf"))
            best_val_r_info = ckpt.get("best_val_r_info", {})
            best_test_r = ckpt.get("best_test_r", -float("inf"))
            best_test_r_info = ckpt.get("best_test_r_info", {})
            vg_best_test_rmse = ckpt.get("vg_best_test_rmse", float("inf"))
            vg_best_test_info = ckpt.get("vg_best_test_info", {})
            test_rmses = ckpt.get("test_rmses", [])
            train_losses = ckpt.get("train_losses", [])
            lrs = ckpt.get("lrs", [])
            if ema is not None and "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            print(f"Full resume from epoch {start_epoch}")
        else:
            # Weights-only checkpoint (best_model_test.pt) — fresh optimizer, step scheduler forward
            for _ in range(start_epoch):
                scheduler.step()
            best_test_rmse = ckpt.get("test_rmse", float("inf"))
            best_val_rmse = ckpt.get("val_rmse", float("inf"))
            print(f"Weights-only resume from epoch {start_epoch} (fresh optimizer, scheduler stepped forward)")

        print(f"  best val RMSE: {best_val_rmse:.4f}, best test RMSE: {best_test_rmse:.4f}")

    def _make_info(epoch, lr, val_metrics, test_metrics, test2_m=None):
        info = {
            "epoch": epoch,
            "lr": lr,
            "val_rmse": float(val_metrics["rmse"]),
            "val_mae": float(val_metrics["mae"]),
            "val_pearson": float(val_metrics["pearson"]),
            "val_spearman": float(val_metrics["spearman"]),
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_pearson": float(test_metrics["pearson"]),
            "test_spearman": float(test_metrics["spearman"]),
        }
        if test2_m is not None:
            info.update({
                "test2_rmse": float(test2_m["rmse"]),
                "test2_mae": float(test2_m["mae"]),
                "test2_pearson": float(test2_m["pearson"]),
                "test2_spearman": float(test2_m["spearman"]),
            })
        return info

    for epoch in range(start_epoch + 1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

        train_loss, epoch_max_gnorm = train_epoch(model, train_loader, optimizer, criterion, device, epoch,
                                 max_grad_norm=max_grad_norm, ema=ema,
                                 coord_noise_std=train_cfg.get("coord_noise_std", 0.0))
        train_losses.append(train_loss)

        # Skip eval during warmup — val curves don't reach within 1% of best until ep ~300 (see eval_warmup config).
        if epoch <= eval_warmup:
            test_logger.write({"epoch": epoch, "lr": lr, "max_grad_norm": epoch_max_gnorm})
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | LR: {lr:.6f}  [eval skipped, ep <= eval_warmup={eval_warmup}]")
            scheduler.step()
            # Latest checkpoint still saved at end of loop (outside this block)
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_rmse": best_val_rmse,
                "best_test_rmse": best_test_rmse,
                "epochs_no_improve": epochs_no_improve,
                "best_val_info": best_val_info,
                "best_test_info": best_test_info,
                "best_val_r": best_val_r,
                "best_val_r_info": best_val_r_info,
                "best_test_r": best_test_r,
                "best_test_r_info": best_test_r_info,
                "vg_best_test_rmse": vg_best_test_rmse,
                "vg_best_test_info": vg_best_test_info,
                "train_losses": train_losses,
                "test_rmses": test_rmses,
                "lrs": lrs,
            }
            torch.save(ckpt_dict, output_dir / "latest_checkpoint.pt")
            continue

        # Swap in EMA weights for evaluation and checkpoint saving
        if ema is not None:
            ema.apply(model)

        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        test_metrics, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
        test_rmses.append(test_metrics["rmse"])

        log_row = {"epoch": epoch, "rmse": test_metrics["rmse"], "mae": test_metrics["mae"],
                   "pearson": test_metrics["pearson"], "spearman": test_metrics["spearman"], "lr": lr,
                   "val_rmse": val_metrics["rmse"], "val_pearson": val_metrics["pearson"],
                   "max_grad_norm": epoch_max_gnorm}
        test2_m = None
        if test2_loader is not None:
            test2_m, _, _ = evaluate(model, test2_loader, criterion, device)
            log_row.update({"test2_rmse": test2_m["rmse"], "test2_mae": test2_m["mae"],
                            "test2_pearson": test2_m["pearson"], "test2_spearman": test2_m["spearman"]})
        test_logger.write(log_row)

        scheduler.step()

        line = (f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val RMSE: {val_metrics['rmse']:.4f} | "
                f"Test RMSE: {test_metrics['rmse']:.4f} | Test R: {test_metrics['pearson']:.4f}")
        if test2_m is not None:
            line += f" | Test2 RMSE: {test2_m['rmse']:.4f}"
        line += f" | LR: {lr:.6f}"
        print(line)

        # Best val checkpoint
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            epochs_no_improve = 0
            best_val_info = _make_info(epoch, lr, val_metrics, test_metrics, test2_m)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "test_rmse": test_metrics["rmse"], "test_pearson": test_metrics["pearson"],
                        "val_rmse": val_metrics["rmse"]},
                       output_dir / "best_model_val.pt")
            # Val-gated best test: best test among val-improving epochs
            if test_metrics["rmse"] < vg_best_test_rmse:
                vg_best_test_rmse = test_metrics["rmse"]
                vg_best_test_info = _make_info(epoch, lr, val_metrics, test_metrics, test2_m)
            print(f"  -> Best val model saved")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                save_plots(output_dir, epoch, test_preds, test_targets, test_rmses, train_losses, lrs, test_metrics["pearson"])
                break

        # Best test checkpoint (by RMSE)
        if test_metrics["rmse"] < best_test_rmse:
            best_test_rmse = test_metrics["rmse"]
            best_test_info = _make_info(epoch, lr, val_metrics, test_metrics, test2_m)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "test_rmse": test_metrics["rmse"], "test_pearson": test_metrics["pearson"],
                        "val_rmse": val_metrics["rmse"]},
                       output_dir / "best_model_test.pt")
            print(f"  -> Best test model saved")

        # Best val R checkpoint
        if val_metrics["pearson"] > best_val_r:
            best_val_r = val_metrics["pearson"]
            best_val_r_info = _make_info(epoch, lr, val_metrics, test_metrics, test2_m)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "test_rmse": test_metrics["rmse"], "test_pearson": test_metrics["pearson"],
                        "val_rmse": val_metrics["rmse"], "val_pearson": val_metrics["pearson"]},
                       output_dir / "best_model_val_r.pt")
            print(f"  -> Best val R model saved")

        # Best test R checkpoint (absolute best, any epoch)
        if test_metrics["pearson"] > best_test_r:
            best_test_r = test_metrics["pearson"]
            best_test_r_info = _make_info(epoch, lr, val_metrics, test_metrics, test2_m)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "test_rmse": test_metrics["rmse"], "test_pearson": test_metrics["pearson"],
                        "val_rmse": val_metrics["rmse"]},
                       output_dir / "best_model_test_r.pt")
            print(f"  -> Best test R model saved")

        # Periodic checkpoint every 50 epochs after epoch 150
        if epoch >= 150 and epoch % 50 == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "test_rmse": test_metrics["rmse"], "test_pearson": test_metrics["pearson"],
                        "val_rmse": val_metrics["rmse"]},
                       output_dir / f"checkpoint_ep{epoch}.pt")

        # Restore real weights after EMA evaluation/saving
        if ema is not None:
            ema.restore(model)

        # Latest checkpoint (for resume)
        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_rmse": best_val_rmse,
            "best_test_rmse": best_test_rmse,
            "epochs_no_improve": epochs_no_improve,
            "best_val_info": best_val_info,
            "best_test_info": best_test_info,
            "best_val_r": best_val_r,
            "best_val_r_info": best_val_r_info,
            "best_test_r": best_test_r,
            "best_test_r_info": best_test_r_info,
            "vg_best_test_rmse": vg_best_test_rmse,
            "vg_best_test_info": vg_best_test_info,
            "test_rmses": test_rmses,
            "train_losses": train_losses,
            "lrs": lrs,
        }
        if ema is not None:
            ckpt_dict["ema_state_dict"] = ema.state_dict()
        torch.save(ckpt_dict, output_dir / "latest_checkpoint.pt")

        # Snapshot val-selected checkpoint at milestones (600,700,...,1200)
        if epoch in _snapshot_milestones:
            src = output_dir / "best_model_val.pt"
            dst = output_dir / f"best_model_val_at{epoch}.pt"
            if src.exists():
                val_ep = best_val_info.get("epoch", -1) if best_val_info else -1
                shutil.copy2(src, dst)
                print(f"  -> Val-selected snapshot saved: {dst.name} (val-best ep {val_ep})")

        if epoch % 5 == 0 or epoch == epochs:
            save_plots(output_dir, epoch, test_preds, test_targets, test_rmses, train_losses, lrs, test_metrics["pearson"])

    test_logger.close()

    # Save best_metrics.json with both val-selected and test-selected results
    best_metrics = {
        "num_params": num_params,
        "current_epoch": epoch,
        "best_val": best_val_info,
        "best_test": best_test_info,
        "best_val_r": best_val_r_info,
        "best_test_r": best_test_r_info,
        "val_gated_best_test": vg_best_test_info,
    }
    best_metrics["end_time"] = datetime.now(timezone.utc).isoformat()
    best_metrics["elapsed_seconds"] = round(time.monotonic() - _wall_start, 1)
    with open(output_dir / "best_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"\nDone. Best val RMSE: {best_val_rmse:.4f} | Best test RMSE: {best_test_rmse:.4f}")


if __name__ == "__main__":
    main()
