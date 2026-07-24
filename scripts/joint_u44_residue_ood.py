import numpy as np, csv
from pathlib import Path
from scipy.stats import pearsonr

ROOT = Path("/nfs/lambda_stor_01/homes/wongr/good_affinity_predictors/binding_gnn")
NPZ = ROOT / "results/ood_ft_preds_bestof"
FT  = ROOT / "mpnn_2copies_datamax1_v3/experiments/ood_finetuning"
CLUS = ["1nvq","1sqa","2p15","2vw5","3dd0","3f3e","3o9i"]

def u44_folds(cluster):
    d = np.load(NPZ / f"u44_{cluster}_n25.npz", allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    y = d["y"].astype(float)
    folds = [d[f"p{i}"].astype(float) for i in range(5)]   # 5 atom fold preds, aligned to ids
    return ids, y, folds

def npz_folds_aligned(arch, cluster, ref_ids):
    d = np.load(NPZ / f"{arch}_{cluster}_n25.npz", allow_pickle=True)
    idx = {str(pid): k for k, pid in enumerate(d["ids"])}
    order = [idx[p] for p in ref_ids]                      # reorder to u44 id order
    return [d[f"p{i}"].astype(float)[order] for i in range(5)]

def csv_folds_aligned(arch, cluster, ref_ids):
    folds = []
    for f in range(5):
        p = FT / f"{arch}_residue" / f"{arch}_{cluster}_f{f}_full_n25" / "preds_ft.csv"
        m = {}
        with open(p) as fh:
            for row in csv.DictReader(fh):
                m[row["pdb_id"]] = float(row["pred"])
        folds.append(np.array([m[pid] for pid in ref_ids]))  # aligned to u44 ids
    return folds

def joint(residue_loader, arch):
    per = []
    for c in CLUS:
        ids, y, af = u44_folds(c)
        rf = residue_loader(arch, c, ids)
        pooled = np.stack(af + rf, axis=0)          # 10 x N
        pavg = pooled.mean(axis=0)                   # equal-weight over 10 folds
        r = pearsonr(pavg, y).statistic
        per.append(r)
        print(f"  {c}: R={r:.4f}  (n={len(y)})")
    print(f"  == mean R = {np.mean(per):.4f}")
    return np.mean(per)

print("VALIDATION  u44 + fr7 (should reproduce paper 0.592):")
joint(npz_folds_aligned, "fr7")
print("\nNEW  u44 + frrho2 (residue L=2):")
joint(csv_folds_aligned, "frrho2")
