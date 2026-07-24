"""Ensemble pre-FT OOD predictions across folds.

For each (arch, cluster): load best_model_val.pt from each fold's run dir,
predict on the held-out OOD cluster test set, average predictions across folds,
report Pearson R + RMSE + Spearman.

Usage:
    python scripts/ensemble_ood_preft.py --archs zc12,zc49

The cluster list is hardcoded to the 7 PLINDER clusters in the paper.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import yaml
from torch_geometric.loader import DataLoader

_LAMBDA_ROOT = "/nfs/lambda_stor_01/homes/wongr/good_affinity_predictors/binding_gnn"
if os.path.isdir(_LAMBDA_ROOT):
    os.chdir(_LAMBDA_ROOT)  # lambda: hardcoded root; elsewhere stay in launcher's cwd (repo root)
sys.path.insert(0, "mpnn_2copies_datamax1_v3")
sys.path.insert(0, "mpnn_2copies_datamax1_v3/models")
sys.path.insert(0, ".")
from train import config_to_model_config, PrepareData_2copies

device = torch.device("cuda:0")
OOD_BASE = Path("mpnn_2copies_datamax1_v3/experiments/ood")
CLUSTERS = ["1nvq", "1sqa", "2p15", "2vw5", "3dd0", "3f3e", "3o9i"]
FOLDS = [0, 1, 2, 3, 4]


def load_model(source_dir, model_cfg, device):
    for mn in list(sys.modules.keys()):
        if mn in ("model", "global_sh", "onsite", "tpconv", "edge_onsite",
                  "virtual_onsite", "readout_fusion"):
            del sys.modules[mn]
    sys.path.insert(0, str(source_dir))
    from model import EquivariantMPNN
    m = EquivariantMPNN(model_cfg).to(device)
    sys.path.remove(str(source_dir))
    return m


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, tgts = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        out = out[0] if isinstance(out, tuple) else out
        preds.append(out.cpu().view(-1))
        tgts.append(batch.y.cpu().view(-1))
    return torch.cat(preds).numpy(), torch.cat(tgts).numpy()


def ensemble_one(arch, cluster):
    """Return dict with per-fold metrics + ensemble metrics, or None if no folds usable."""
    test_path = f"gign_exact_v1/data/v3_residue/ood/ood_{cluster}_test_5A.pt"
    if not Path(test_path).exists():
        return None
    test_data = torch.load(test_path, weights_only=False)

    fold_preds = []
    targets_arr = None
    loader = None
    per_fold = []

    for fold in FOLDS:
        run_dir = None
        # Find a run dir with best_model_val.pt. Run dirs live under an
        # {arch}_residue/ parent (per the config output.dir); fall back to the
        # flat layout for older runs that lacked it.
        fold_base = OOD_BASE / f"{arch}_residue" / f"{arch}_{cluster}_f{fold}"
        if not fold_base.exists():
            fold_base = OOD_BASE / f"{arch}_{cluster}_f{fold}"
        candidates = [d for d in fold_base.iterdir()
                      if d.is_dir() and (d / "best_model_val.pt").exists()] if fold_base.exists() else []
        if not candidates:
            per_fold.append((fold, None))
            continue
        # pick the latest run per fold by dir name; this reproduces the
        # published pre-FT ensemble for 6/7 clusters (2vw5 excepted -- see
        # results/ood_baseline_official.csv notes; one 2vw5 fold checkpoint drifted).
        run_dir = sorted(candidates, key=lambda d: d.name)[-1]
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        model_cfg = config_to_model_config(cfg)
        if loader is None:
            transform = PrepareData_2copies(
                use_inter_edges=cfg.get("data", {}).get("use_inter_edges", True),
                use_ext_features=cfg.get("data", {}).get("use_ext_features", False),
                use_bfactor=cfg.get("data", {}).get("use_bfactor", False),
            )
            loader = DataLoader([transform(d) for d in test_data], batch_size=64, shuffle=False)
        model = load_model(str(run_dir / "source"), model_cfg, device)
        ckpt = torch.load(run_dir / "best_model_val.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        p, t = predict(model, loader)
        if targets_arr is None:
            targets_arr = t
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        r = float(np.corrcoef(p, t)[0, 1])
        per_fold.append((fold, (rmse, r)))
        fold_preds.append(p)
        del model
        torch.cuda.empty_cache()

    if not fold_preds:
        return None
    ens = np.mean(fold_preds, axis=0)
    ens_rmse = float(np.sqrt(np.mean((ens - targets_arr) ** 2)))
    ens_r = float(np.corrcoef(ens, targets_arr)[0, 1])
    ens_rho = float(scipy.stats.spearmanr(ens, targets_arr)[0])
    return {
        "per_fold": per_fold,
        "n_folds": len(fold_preds),
        "ensemble_rmse": ens_rmse,
        "ensemble_r": ens_r,
        "ensemble_rho": ens_rho,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", required=True,
                    help="comma-separated archs, e.g. zc12,zc49")
    ap.add_argument("--out", default=None,
                    help="write per (arch,cluster) results to this CSV (tracked provenance)")
    args = ap.parse_args()

    out_rows = []
    for arch in args.archs.split(","):
        arch = arch.strip()
        print(f"\n========== {arch} (pre-FT 5-fold ensemble per cluster) ==========")
        cluster_r = []
        for cluster in CLUSTERS:
            res = ensemble_one(arch, cluster)
            if res is None:
                print(f"  {cluster}: NO FOLDS")
                continue
            print(f"  {cluster} ({res['n_folds']}/5 folds): "
                  f"ensemble RMSE={res['ensemble_rmse']:.4f}  R={res['ensemble_r']:.4f}  rho={res['ensemble_rho']:.4f}")
            for fold, m in res["per_fold"]:
                if m is None:
                    print(f"    f{fold}: skip")
                else:
                    print(f"    f{fold}: RMSE={m[0]:.4f}  R={m[1]:.4f}")
            cluster_r.append(res["ensemble_r"])
            out_rows.append({"arch": arch, "cluster": cluster, "n_folds": res["n_folds"],
                             "ensemble_R": round(res["ensemble_r"], 4),
                             "ensemble_RMSE": round(res["ensemble_rmse"], 4),
                             "ensemble_rho": round(res["ensemble_rho"], 4)})
        if cluster_r:
            print(f"\n  Mean cluster R across {len(cluster_r)} clusters: {np.mean(cluster_r):.4f}")

    if args.out and out_rows:
        import csv as _csv
        with open(args.out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {args.out}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
