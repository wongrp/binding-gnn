#!/usr/bin/env python
"""Ensemble OOD FT predictions across folds.

For each (prefix, cluster), loads per-sample predictions from each fold's
preds_{baseline,ft}.csv, averages predictions across folds (after aligning
on pdb_id), and reports pooled Pearson R and RMSE.

Usage:
  python scripts/ensemble_ood_preds.py \
    --prefix zc47 \
    --folds 0,1,2 \
    --clusters 1nvq,1sqa,2p15,2vw5,3dd0,3f3e,3o9i \
    --kind baseline      # or 'ft'
    --suffix _3fold
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

FT_DIR = Path("mpnn_2copies_datamax1_v3/experiments/ood_finetuning")


def load_preds(csv_path):
    """Return dict {pdb_id: (target, pred)}."""
    out = {}
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pid = row["pdb_id"]
            out[pid] = (float(row["target"]), float(row["pred"]))
    return out


def ensemble_cluster(prefix, cluster, folds, kind, suffix):
    fold_preds = []
    targets_by_id = None
    for f in folds:
        run_dir = FT_DIR / f"{prefix}_{cluster}_f{f}_full_n25{suffix}"
        csv_path = run_dir / f"preds_{kind}.csv"
        if not csv_path.exists():
            return None, f"missing {csv_path}"
        p = load_preds(csv_path)
        if targets_by_id is None:
            targets_by_id = {pid: t for pid, (t, _) in p.items()}
        fold_preds.append({pid: pr for pid, (_, pr) in p.items()})

    # Intersect pdb_ids across folds
    common = set(fold_preds[0].keys())
    for fp in fold_preds[1:]:
        common &= set(fp.keys())
    common = sorted(common)
    if not common:
        return None, "no common pdb_ids across folds"

    targets = np.array([targets_by_id[pid] for pid in common])
    preds_per_fold = np.stack(
        [[fp[pid] for pid in common] for fp in fold_preds], axis=0
    )  # [n_folds, n_samples]
    preds = preds_per_fold.mean(axis=0)

    mask = np.isfinite(preds) & np.isfinite(targets)
    p, t = preds[mask], targets[mask]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    pearson = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 else 0.0
    spear = float(spearmanr(p, t)[0]) if len(p) > 1 else 0.0
    return dict(
        n_samples=int(mask.sum()),
        rmse=rmse,
        pearson=pearson,
        spearman=spear,
        n_folds=len(folds),
    ), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="zc47")
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument(
        "--clusters",
        default="1nvq,1sqa,2p15,2vw5,3dd0,3f3e,3o9i",
    )
    ap.add_argument("--kind", choices=["baseline", "ft"], default="ft")
    ap.add_argument(
        "--suffix",
        default="_3fold",
        help="suffix on the run dir name after _full_n25 (e.g. '_3fold' or '_preview')",
    )
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    clusters = args.clusters.split(",")

    print(f"=== {args.prefix} FT-25 {args.kind} ensemble over folds {folds} ===")
    print(f"{'cluster':>8}  {'n':>4}  {'R':>7}  {'RMSE':>7}  {'Spearman':>8}")
    print("-" * 44)

    all_results = {}
    rs = []
    for c in clusters:
        res, err = ensemble_cluster(
            args.prefix, c, folds, args.kind, args.suffix
        )
        if res is None:
            print(f"{c:>8}  ----  ------   ------  --------    ({err})")
            continue
        all_results[c] = res
        rs.append(res["pearson"])
        print(
            f"{c:>8}  {res['n_samples']:>4}  "
            f"{res['pearson']:>7.4f}  {res['rmse']:>7.4f}  {res['spearman']:>8.4f}"
        )

    if rs:
        avg = float(np.mean(rs))
        print("-" * 44)
        print(f"{'average':>8}  {' ':>4}  {avg:>7.4f}")
        all_results["_average_R"] = avg

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
