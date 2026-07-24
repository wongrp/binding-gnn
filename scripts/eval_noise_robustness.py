#!/usr/bin/env python3
"""Evaluate ablation checkpoints on CASF with Gaussian position noise.

Supports three perturbation modes:
  - "both":    perturb data.pos AND data.pos_unbound (same draw, identical shift)
  - "unbound": perturb only data.pos_unbound (bound geometry clean)
  - "bound":   perturb only data.pos (unbound clean; included for symmetry)

Stock CASF has no pos_unbound; we synthesize pos_unbound = pos.clone() before
applying noise. With sigma=0 this reproduces the original evaluation.

Usage:
    python scripts/eval_noise_robustness.py \\
        --run-dirs mpnn_2copies_datamax1_v3/experiments/cleansplit/ZC_series_ablation/zc*/2026-*s42 \\
        --sigmas 0 0.1 0.25 0.5 1.0 \\
        --draws 3 \\
        --mode both unbound \\
        --device cuda:0 \\
        --out results/noise_robustness.csv
"""

import argparse
import copy
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASF_DEFAULT = PROJECT_ROOT / "gign_exact_v1/data/v3_with_residues/cleansplit_casf2016_5A.pt"


def load_model(run_dir, device):
    """Load model from a run directory, using its archived source for model code,
    but the CURRENT train.py for transforms (needed for pos_unbound support)."""
    run_dir = Path(run_dir)
    source_dir = run_dir / "source"
    config_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "best_model_val.pt"

    for mod_name in list(sys.modules.keys()):
        if mod_name in ("model", "train", "global_sh", "onsite", "tpconv",
                        "edge_onsite", "virtual_onsite", "readout_fusion"):
            del sys.modules[mod_name]

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(source_dir))

    model_mod = importlib.import_module("model")
    archived_train = importlib.import_module("train")
    sys.path.pop(0)  # drop source_dir so current train.py wins next import

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = archived_train.config_to_model_config(cfg)
    model = model_mod.EquivariantMPNN(model_cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Load CURRENT train.py for transforms with pos_unbound support
    del sys.modules["train"]
    sys.path.insert(0, str(PROJECT_ROOT / "mpnn_2copies_datamax1_v3"))
    current_train = importlib.import_module("train")
    sys.path.pop(0)
    return model, cfg, current_train


def _rebuild_inter_edges(pos, num_ligand, threshold=5.0):
    """Rebuild ligand-pocket inter edges via 5 Å cutoff on (possibly perturbed) positions.

    Covalent intra edges are NOT rebuilt; bonds don't care about small displacements.
    """
    pos_lig = pos[:num_ligand]
    pos_pkt = pos[num_ligand:]
    # Pairwise distances via cdist
    d = torch.cdist(pos_lig, pos_pkt)  # [N_lig, N_pkt]
    lig_idx, pkt_idx = torch.where(d < threshold)
    pkt_idx_offset = pkt_idx + num_ligand
    src = torch.cat([lig_idx, pkt_idx_offset])
    dst = torch.cat([pkt_idx_offset, lig_idx])
    return torch.stack([src, dst], dim=0).long()


def perturbed_samples(raw_data, mode, sigma, rng, rebuild_edges=False):
    """Return deep-copied data list with Gaussian noise applied.

    Noise is zero-mean, isotropic, std=sigma (Å). For mode='both', bound and
    unbound get the SAME noise draw. For 'unbound' / 'bound', only the named
    copy is shifted.

    If rebuild_edges is True, edge_index_inter is recomputed from the perturbed
    bound positions (5 Å cutoff). Covalent edge_index_intra is always preserved.
    """
    out = []
    for d in raw_data:
        d2 = copy.deepcopy(d)
        if not hasattr(d2, "pos_unbound"):
            d2.pos_unbound = d2.pos.clone()
        if sigma > 0:
            noise = torch.from_numpy(
                rng.normal(0.0, sigma, size=d2.pos.shape).astype(np.float32)
            )
            if mode == "both":
                d2.pos = d2.pos + noise
                d2.pos_unbound = d2.pos_unbound + noise
            elif mode == "unbound":
                d2.pos_unbound = d2.pos_unbound + noise
            elif mode == "bound":
                d2.pos = d2.pos + noise
            else:
                raise ValueError(f"unknown mode: {mode}")

            if rebuild_edges:
                # Rebuild inter edges from bound (possibly perturbed) positions.
                # Inter edges live only in the bound copy of the 2-copy graph.
                num_lig = d2.num_ligand_atoms
                if isinstance(num_lig, torch.Tensor):
                    num_lig = int(num_lig.item())
                d2.edge_index_inter = _rebuild_inter_edges(d2.pos, num_lig, threshold=5.0)
        out.append(d2)
    return out


def run_inference(model, train_mod, cfg, data_list, device, batch_size=64):
    data_cfg = cfg.get("data", {})
    use_2copies = data_cfg.get("use_2copies", True)
    kwargs = dict(
        use_inter_edges=data_cfg.get("use_inter_edges", True),
        use_ext_features=data_cfg.get("use_ext_features", False),
        use_bfactor=data_cfg.get("use_bfactor", False),
    )
    if use_2copies:
        transform = train_mod.PrepareData_2copies(**kwargs)
    else:
        # PrepareData doesn't accept ext/bfactor; pass only compatible kwargs
        transform = train_mod.PrepareData(use_inter_edges=kwargs["use_inter_edges"])
    transformed = [transform(d) for d in data_list]
    loader = DataLoader(transformed, batch_size=batch_size, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def metrics(preds, targets):
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    R = float(scipy.stats.pearsonr(preds, targets).statistic)
    return rmse, R


def experiment_label(run_dir: Path) -> tuple[str, str]:
    # e.g. .../zc9_no_all_sh/2026-04-22_3_s72 → ("zc9_no_all_sh", "s72")
    exp_name = run_dir.parent.name
    run_name = run_dir.name
    seed = run_name.rsplit("_s", 1)[-1] if "_s" in run_name else "?"
    return exp_name, f"s{seed}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", nargs="+", required=True,
                   help="Glob-expanded paths like zc*/2026-*s42")
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--mode", nargs="+", default=["both", "unbound"],
                   choices=["both", "unbound", "bound"])
    p.add_argument("--test", default=str(CASF_DEFAULT))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True, help="CSV output path")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for noise draws")
    p.add_argument("--rebuild-edges", action="store_true",
                   help="Recompute ligand-pocket inter edges (5 Å cutoff) from perturbed positions")
    args = p.parse_args()

    raw = torch.load(args.test, weights_only=False)
    print(f"Loaded {len(raw)} samples from {args.test}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    f = open(args.out, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["experiment", "seed", "mode", "sigma", "draw", "rmse", "R"])

    device = args.device
    rng = np.random.default_rng(args.seed)

    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        if not (run_dir / "best_model_val.pt").exists():
            print(f"  skip {run_dir} (no best_model_val.pt)")
            continue
        exp_name, seed_label = experiment_label(run_dir)
        print(f"\n=== {exp_name} / {seed_label} ===")
        model, cfg, train_mod = load_model(run_dir, device)
        # Report param count for sanity
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")

        for mode in args.mode:
            for sigma in args.sigmas:
                n_draws = 1 if sigma == 0.0 else args.draws
                for draw in range(n_draws):
                    data_list = perturbed_samples(raw, mode, sigma, rng, rebuild_edges=args.rebuild_edges)
                    preds, targets = run_inference(model, train_mod, cfg, data_list, device)
                    rmse, R = metrics(preds, targets)
                    print(f"  {mode:<7} σ={sigma:<4} draw={draw}  RMSE={rmse:.4f}  R={R:.4f}")
                    writer.writerow([exp_name, seed_label, mode, sigma, draw, f"{rmse:.4f}", f"{R:.4f}"])
                    f.flush()

    f.close()
    print(f"\nDone. Wrote {args.out}")


if __name__ == "__main__":
    main()
