"""Post-FT 5-fold ensemble for the 3 new residue archs (frrho1, frrho2, frdsym0).

Reads per-fold preds_ft.csv (and preds_baseline.csv for reference) under
experiments/ood_finetuning/<arch>_residue/<arch>_<cluster>_f<fold>_full_n25/
averages predictions per pdb_id across folds, reports pooled Pearson + RMSE.

Usage:
    python scripts/ensemble_residue_archs_postft.py --archs frrho1,frrho2,frdsym0
"""
import argparse, csv, os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

os.chdir('/nfs/lambda_stor_01/homes/wongr/good_affinity_predictors/binding_gnn')
FT_BASE = Path('mpnn_2copies_datamax1_v3/experiments/ood_finetuning')
CLUSTERS = ['1nvq', '1sqa', '2p15', '2vw5', '3dd0', '3f3e', '3o9i']
FOLDS = [0, 1, 2, 3, 4]


def load_preds(csv_path):
    out = {}
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            out[row['pdb_id']] = (float(row['target']), float(row['pred']))
    return out


def ensemble_one(arch, cluster, kind):
    """kind in {'baseline', 'ft'}. Return (n_folds_used, rmse, R, rho) or None."""
    fold_preds, targets_by_id = [], None
    n_used = 0
    for fold in FOLDS:
        rd = FT_BASE / f'{arch}_residue' / f'{arch}_{cluster}_f{fold}_full_n25'
        csv_p = rd / f'preds_{kind}.csv'
        if not csv_p.exists():
            continue
        p = load_preds(csv_p)
        if not p:
            continue
        if targets_by_id is None:
            targets_by_id = {pid: t for pid, (t, _) in p.items()}
        # Sort by pdb_id for consistent ordering across folds
        ids_sorted = sorted(p.keys())
        fold_preds.append(np.array([p[pid][1] for pid in ids_sorted]))
        if not hasattr(ensemble_one, '_ids'):
            ensemble_one._ids = ids_sorted
        n_used += 1
    if n_used == 0:
        return None
    ids = sorted(targets_by_id.keys())
    targets = np.array([targets_by_id[pid] for pid in ids])
    ens = np.mean(fold_preds, axis=0)
    rmse = float(np.sqrt(np.mean((ens - targets) ** 2)))
    r = float(np.corrcoef(ens, targets)[0, 1])
    rho = float(spearmanr(ens, targets)[0])
    return n_used, rmse, r, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', required=True)
    args = ap.parse_args()
    for arch in args.archs.split(','):
        arch = arch.strip()
        print(f'\n========== {arch} (post-FT 5-fold ensemble per cluster) ==========')
        baseline_rs, ft_rs = [], []
        for cluster in CLUSTERS:
            # Recompute baseline for reference too
            base = ensemble_one(arch, cluster, 'baseline')
            ft = ensemble_one(arch, cluster, 'ft')
            if base is None and ft is None:
                print(f'  {cluster}: NO FOLDS')
                continue
            line = f'  {cluster}:'
            if base is not None:
                n_b, _, r_b, _ = base
                line += f'  baseline R={r_b:.4f} ({n_b}/5)'
                baseline_rs.append(r_b)
            if ft is not None:
                n_f, rmse_f, r_f, rho_f = ft
                line += f'  FT R={r_f:.4f} ({n_f}/5)  RMSE={rmse_f:.4f}  rho={rho_f:.4f}'
                ft_rs.append(r_f)
                if base is not None:
                    line += f'  Δ={r_f - r_b:+.4f}'
            print(line)
        if baseline_rs:
            print(f'  ── {arch} mean baseline R = {np.mean(baseline_rs):.4f}  N={len(baseline_rs)}')
        if ft_rs:
            print(f'  ── {arch} mean FT       R = {np.mean(ft_rs):.4f}  N={len(ft_rs)}')


if __name__ == '__main__':
    main()
