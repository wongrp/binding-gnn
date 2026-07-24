#!/usr/bin/env python3
"""Ensemble predictions from 5-fold CV checkpoints.

Loads each fold's best_model_val.pt, runs inference on test sets,
averages predictions, and computes ensemble metrics.

Usage:
    python scripts/ensemble_cv.py --cv-dir mpnn_2copies_datamax1_v3/experiments/cleansplit/U49_5fold_cv --prefix u49
    python scripts/ensemble_cv.py --cv-dir mpnn_2copies_datamax1_v3/experiments/cleansplit/U44_5fold_cv --prefix u44
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def find_run_dir(fold_dir, seed=None):
    """Find the best run subdirectory inside a fold directory.

    If multiple runs exist (e.g. different seeds), pick the one with the
    lowest val RMSE from its checkpoint metadata. If ``seed`` is given,
    restrict to run dirs ending in ``_s{seed}`` (enforces a single-seed,
    apples-to-apples ensemble).
    """
    subdirs = [d for d in fold_dir.iterdir() if d.is_dir()]
    if seed is not None:
        subdirs = [d for d in subdirs if d.name.endswith(f"_s{seed}")]
    if len(subdirs) == 0:
        raise ValueError(f"No run dirs in {fold_dir}" + (f" matching _s{seed}" if seed is not None else ""))
    if len(subdirs) == 1:
        return subdirs[0]
    # Multiple runs: pick best by val RMSE from checkpoint
    best_dir, best_val = None, float("inf")
    for d in subdirs:
        ckpt = d / "best_model_val.pt"
        if ckpt.exists():
            try:
                meta = torch.load(ckpt, map_location="cpu", weights_only=False)
                val = meta.get("val_rmse", float("inf"))
                if val < best_val:
                    best_val = val
                    best_dir = d
            except Exception:
                pass
    if best_dir is None:
        best_dir = subdirs[0]
    print(f"  {fold_dir.name}: {len(subdirs)} runs, picked {best_dir.name} (val={best_val:.4f})")
    return best_dir


def load_model_from_run(run_dir, device):
    """Load model from a run directory using its archived source code."""
    source_dir = run_dir / "source"
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "best_model_val.pt"

    # Import from the archived source
    sys.path.insert(0, str(source_dir))
    import importlib
    # Clear cached modules to avoid stale imports across folds
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("model", "train", "global_sh", "onsite", "tpconv",
                        "edge_onsite", "virtual_onsite", "readout_fusion"):
            del sys.modules[mod_name]

    model_mod = importlib.import_module("model")
    train_mod = importlib.import_module("train")
    sys.path.pop(0)

    # Build model from config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = train_mod.config_to_model_config(cfg)
    model = model_mod.EquivariantMPNN(model_cfg).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    return model, cfg


def get_transform(cfg):
    """Create the data transform from config."""
    # Import from current codebase (transforms are simple, no code drift risk)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mpnn_2copies_datamax1_v3"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mpnn_2copies_datamax1_v3" / "models"))
    from train import PrepareData, PrepareData_2copies
    sys.path.pop(0)
    sys.path.pop(0)

    data_cfg = cfg.get("data", {})
    use_inter = data_cfg.get("use_inter_edges", True)
    use_2copies = data_cfg.get("use_2copies", False)
    drop_isolated = data_cfg.get("drop_isolated_nodes", False)
    use_ext_features = data_cfg.get("use_ext_features", False)
    use_bfactor = data_cfg.get("use_bfactor", False)
    ext_feature_mask = data_cfg.get("ext_feature_mask", None)

    if use_2copies:
        return PrepareData_2copies(
            use_inter_edges=use_inter,
            drop_isolated_nodes=drop_isolated,
            use_ext_features=use_ext_features,
            use_bfactor=use_bfactor,
            ext_feature_mask=ext_feature_mask,
        )
    else:
        return PrepareData(use_inter_edges=use_inter, drop_isolated_nodes=drop_isolated)


@torch.no_grad()
def predict(model, loader, device):
    """Run inference, return (preds, targets) as numpy arrays."""
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred, _ = model(batch)
        all_preds.append(pred.cpu().view(-1))
        all_targets.append(batch.y.cpu().view(-1))
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return preds, targets


def metrics(preds, targets):
    mask = np.isfinite(preds) & np.isfinite(targets)
    p, t = preds[mask], targets[mask]
    rmse = np.sqrt(np.mean((p - t) ** 2))
    pearson = np.corrcoef(p, t)[0, 1] if len(p) > 1 else 0.0
    spearman = scipy.stats.spearmanr(p, t)[0] if len(p) > 1 else 0.0
    return {"rmse": rmse, "R": pearson, "rho": spearman}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-dir", required=True, help="CV experiment directory")
    parser.add_argument("--prefix", required=True, help="Fold prefix (e.g. u49 or u44)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None,
                        help="If set, only ensemble run dirs ending in _s{seed} (one per fold).")
    args = parser.parse_args()

    cv_dir = Path(args.cv_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Discover fold directories
    fold_dirs = []
    for i in range(args.folds):
        fold_name = f"{args.prefix}_f{i}"
        fold_path = cv_dir / fold_name
        if not fold_path.exists():
            print(f"WARNING: {fold_path} not found, skipping")
            continue
        fold_dirs.append((i, fold_path))

    print(f"Found {len(fold_dirs)} folds")

    # Load test data once (same for all folds)
    first_run = find_run_dir(fold_dirs[0][1], seed=args.seed)
    with open(first_run / "config.yaml") as f:
        first_cfg = yaml.safe_load(f)

    data_cfg = first_cfg.get("data", {})
    test_path = data_cfg["test_path"]
    print(f"Loading test data: {test_path}")
    test_data_raw = torch.load(test_path, weights_only=False)

    # Auto-detect indep test set
    test2_path = data_cfg.get("test2_path")
    if not test2_path and "cleansplit_casf2016_5A" in test_path:
        candidate = test_path.replace("cleansplit_casf2016_5A", "cleansplit_casf2016_indep_5A")
        if Path(candidate).exists():
            test2_path = candidate
    test2_data_raw = None
    if test2_path:
        print(f"Loading indep test data: {test2_path}")
        test2_data_raw = torch.load(test2_path, weights_only=False)

    # Transform test data
    transform = get_transform(first_cfg)
    test_data = [transform(d) for d in tqdm(test_data_raw, desc="Preparing test")]
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    test2_loader = None
    if test2_data_raw is not None:
        test2_data = [transform(d) for d in tqdm(test2_data_raw, desc="Preparing indep")]
        test2_loader = DataLoader(test2_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Collect predictions from each fold
    all_test_preds = []
    all_test2_preds = []
    test_targets = None
    test2_targets = None

    for i, fold_path in fold_dirs:
        run_dir = find_run_dir(fold_path, seed=args.seed)
        print(f"\n--- Fold {i}: {run_dir.name} ---")

        model, cfg = load_model_from_run(run_dir, device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {num_params:,}")

        preds, targets = predict(model, test_loader, device)
        all_test_preds.append(preds)
        if test_targets is None:
            test_targets = targets
        else:
            assert np.allclose(targets, test_targets), "Targets differ across folds!"

        m = metrics(preds, targets)
        print(f"  Test:  RMSE={m['rmse']:.4f}  R={m['R']:.4f}  rho={m['rho']:.4f}")

        if test2_loader is not None:
            preds2, targets2 = predict(model, test2_loader, device)
            all_test2_preds.append(preds2)
            if test2_targets is None:
                test2_targets = targets2
            else:
                assert np.allclose(targets2, test2_targets)
            m2 = metrics(preds2, targets2)
            print(f"  Indep: RMSE={m2['rmse']:.4f}  R={m2['R']:.4f}  rho={m2['rho']:.4f}")

        del model
        torch.cuda.empty_cache()

    # Ensemble: average predictions
    print("\n" + "=" * 60)
    print(f"ENSEMBLE ({len(all_test_preds)} folds)")
    print("=" * 60)

    ensemble_preds = np.mean(all_test_preds, axis=0)
    m = metrics(ensemble_preds, test_targets)
    print(f"Test:  RMSE={m['rmse']:.4f}  R={m['R']:.4f}  rho={m['rho']:.4f}")

    # Mean-of-individual for comparison
    individual_rmses = [metrics(p, test_targets)["rmse"] for p in all_test_preds]
    print(f"Mean individual RMSE: {np.mean(individual_rmses):.4f} +/- {np.std(individual_rmses):.4f}")

    if test2_loader is not None:
        ensemble_preds2 = np.mean(all_test2_preds, axis=0)
        m2 = metrics(ensemble_preds2, test2_targets)
        print(f"Indep: RMSE={m2['rmse']:.4f}  R={m2['R']:.4f}  rho={m2['rho']:.4f}")

        individual_rmses2 = [metrics(p, test2_targets)["rmse"] for p in all_test2_preds]
        print(f"Mean individual RMSE: {np.mean(individual_rmses2):.4f} +/- {np.std(individual_rmses2):.4f}")

    # Pairwise prediction correlation (diversity check)
    print(f"\nPairwise prediction correlation (test):")
    for i in range(len(all_test_preds)):
        for j in range(i + 1, len(all_test_preds)):
            r = np.corrcoef(all_test_preds[i], all_test_preds[j])[0, 1]
            print(f"  f{fold_dirs[i][0]} vs f{fold_dirs[j][0]}: R={r:.4f}")


if __name__ == "__main__":
    main()
