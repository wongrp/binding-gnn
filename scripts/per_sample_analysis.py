#!/usr/bin/env python3
"""Per-sample analysis: correlate per-complex features with per-model prediction error.

Computes for each test complex:
  - n_ligand_atoms: ligand heavy atom count
  - n_interface_atoms_T1: pocket atoms directly contacting any ligand atom (via inter edges)
  - n_interface_atoms_T2: T1 + their intra neighbors (2-hop interface)
  - n_inter_edges: total interface atom pairs (directed → /2 for undirected)
  - frac_ligand_contact: fraction of ligand atoms with any inter edge

Then runs inference per model checkpoint and reports:
  - Per-sample predictions and errors
  - Correlation of |error| with each feature

Usage:
    python scripts/per_sample_analysis.py \\
        --run-dirs <list of run dirs> \\
        --test gign_exact_v1/data/v3_with_residues/cleansplit_casf2016_5A.pt \\
        --out results/per_sample_features.csv
"""

import argparse
import copy
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import yaml
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_features(data):
    """Return dict of per-complex features computed from raw data object."""
    n_lig = int(data.num_ligand_atoms) if not isinstance(data.num_ligand_atoms, torch.Tensor) \
        else int(data.num_ligand_atoms.item())
    n_total = int(data.pos.shape[0])
    n_pkt = n_total - n_lig

    # Inter edges: ligand atom <-> pocket atom
    if hasattr(data, 'edge_index_inter') and data.edge_index_inter is not None \
            and data.edge_index_inter.numel() > 0:
        inter_src, inter_dst = data.edge_index_inter
        # pocket atoms involved in any inter edge (dst when src is ligand)
        lig_mask = inter_src < n_lig
        pkt_neighbors_of_lig = set(inter_dst[lig_mask].tolist())
        # Also the reverse direction (pocket → ligand)
        pkt_mask = inter_src >= n_lig
        pkt_neighbors_of_lig.update(inter_src[pkt_mask].tolist())
        # Only keep pocket atoms (indices >= n_lig)
        t1 = set(i for i in pkt_neighbors_of_lig if i >= n_lig)

        n_inter_edges = int(data.edge_index_inter.shape[1] // 2)  # undirected

        # Ligand atoms that have any inter edge
        lig_with_contact = set()
        for s, d in data.edge_index_inter.t().tolist():
            if s < n_lig: lig_with_contact.add(s)
            if d < n_lig: lig_with_contact.add(d)
        frac_lig_contact = len(lig_with_contact) / n_lig if n_lig > 0 else 0.0
    else:
        t1 = set()
        n_inter_edges = 0
        frac_lig_contact = 0.0

    # T2: add intra neighbors of T1 pocket atoms
    t2 = set(t1)
    if hasattr(data, 'edge_index_intra') and data.edge_index_intra.numel() > 0:
        src, dst = data.edge_index_intra
        for s, d in zip(src.tolist(), dst.tolist()):
            if s in t1 and d >= n_lig:
                t2.add(d)
            if d in t1 and s >= n_lig:
                t2.add(s)

    return {
        "n_ligand_atoms": n_lig,
        "n_pocket_atoms": n_pkt,
        "n_interface_T1": len(t1),
        "n_interface_T2": len(t2),
        "n_inter_edges": n_inter_edges,
        "frac_ligand_contact": frac_lig_contact,
    }


def load_model(run_dir, device):
    run_dir = Path(run_dir)
    source_dir = run_dir / "source"
    config_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "best_model_val.pt"

    for mn in list(sys.modules.keys()):
        if mn in ("model", "train", "global_sh", "onsite", "tpconv",
                  "edge_onsite", "virtual_onsite", "readout_fusion"):
            del sys.modules[mn]
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(source_dir))
    model_mod = importlib.import_module("model")
    archived_train = importlib.import_module("train")
    sys.path.pop(0)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = archived_train.config_to_model_config(cfg)
    model = model_mod.EquivariantMPNN(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    del sys.modules["train"]
    sys.path.insert(0, str(PROJECT_ROOT / "mpnn_2copies_datamax1_v3"))
    current_train = importlib.import_module("train")
    sys.path.pop(0)
    return model, cfg, current_train


def run_predictions(model, train_mod, cfg, raw_data, device, batch_size=64):
    data_cfg = cfg.get("data", {})
    kw = dict(
        use_inter_edges=data_cfg.get("use_inter_edges", True),
        use_ext_features=data_cfg.get("use_ext_features", False),
        use_bfactor=data_cfg.get("use_bfactor", False),
    )
    use_2copies = data_cfg.get("use_2copies", True)
    if use_2copies:
        transform = train_mod.PrepareData_2copies(**kw)
    else:
        transform = train_mod.PrepareData(use_inter_edges=kw["use_inter_edges"])

    transformed = [transform(copy.deepcopy(d)) for d in raw_data]
    loader = DataLoader(transformed, batch_size=batch_size, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            out = model(b)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.view(-1).cpu().numpy())
            targets.append(b.y.view(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None,
                   help="Short labels for each run dir (same length). Default: parent dir name")
    p.add_argument("--test", default=str(PROJECT_ROOT / "gign_exact_v1/data/v3_with_residues/cleansplit_casf2016_5A.pt"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    raw = torch.load(args.test, weights_only=False)
    print(f"Loaded {len(raw)} test samples")

    # Compute features for each sample
    feats = []
    for s in raw:
        f = compute_features(s)
        f["pdb_id"] = s.pdb_id if hasattr(s, "pdb_id") else "?"
        f["y"] = float(s.y.item()) if hasattr(s.y, "item") else float(s.y)
        feats.append(f)

    labels = args.labels or [Path(d).parent.name for d in args.run_dirs]
    assert len(labels) == len(args.run_dirs)

    # Get predictions from each model
    model_preds = {}
    for run_dir, label in zip(args.run_dirs, labels):
        run_dir = Path(run_dir)
        if not (run_dir / "best_model_val.pt").exists():
            print(f"  skip {label}: no best_model_val.pt")
            continue
        print(f"\n=== {label} ({run_dir.name}) ===")
        model, cfg, train_mod = load_model(run_dir, args.device)
        preds, targets = run_predictions(model, train_mod, cfg, raw, args.device)
        model_preds[label] = preds
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        print(f"  test RMSE: {rmse:.4f}")

    # Write combined CSV
    feat_keys = ["n_ligand_atoms", "n_pocket_atoms", "n_interface_T1",
                 "n_interface_T2", "n_inter_edges", "frac_ligand_contact"]
    cols = ["pdb_id", "y"] + feat_keys
    for lbl in labels:
        cols += [f"pred_{lbl}", f"err_{lbl}"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, feat in enumerate(feats):
            row = [feat["pdb_id"], feat["y"]] + [feat[k] for k in feat_keys]
            for lbl in labels:
                if lbl in model_preds:
                    p = model_preds[lbl][i]
                    row += [f"{p:.4f}", f"{p - feat['y']:.4f}"]
                else:
                    row += ["", ""]
            w.writerow(row)
    print(f"\nWrote {args.out}")

    # Summary: correlations of |error| vs each feature
    print(f"\n=== |error| Spearman correlation with features ===")
    header = "Feature                   | " + " | ".join(f"{l:<18}" for l in labels if l in model_preds)
    print(header)
    print("-" * len(header))
    for fkey in feat_keys:
        fvec = np.array([f[fkey] for f in feats])
        row = f"{fkey:<25} |"
        for lbl in labels:
            if lbl not in model_preds:
                continue
            abs_err = np.abs(model_preds[lbl] - np.array([f["y"] for f in feats]))
            rho, _ = scipy.stats.spearmanr(fvec, abs_err)
            row += f" rho={rho:+.3f} ({'↑' if rho > 0 else '↓'})  |"
        print(row)


if __name__ == "__main__":
    main()
