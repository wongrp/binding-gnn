import torch, numpy as np
from pathlib import Path

ROOT = Path("/nfs/lambda_stor_01/homes/wongr/good_affinity_predictors/binding_gnn")
ESM2 = ROOT / "gign_exact_v1/data/esm_embeddings/esm2_pocket_embeddings.pt"
OOD  = ROOT / "gign_exact_v1/data/v3_residue/ood"
CLUSTERS = ["1nvq","1sqa","2p15","2vw5","3dd0","3f3e","3o9i"]

def ids(pt):
    return [d.pdb_id for d in torch.load(pt, weights_only=False)]

def pool(e):
    return e.mean(0) if e.dim() > 1 else e

print("loading ESM2 embeddings ...")
esm = torch.load(ESM2, map_location="cpu", weights_only=False)
print(f"  {len(esm)} pocket embeddings")

rows = []
for c in CLUSTERS:
    test_ids  = ids(OOD / f"ood_{c}_test_5A.pt")
    train_ids = ids(OOD / f"ood_{c}_f0_train_5A.pt")
    train = torch.stack([pool(esm[p]).float() for p in train_ids if p in esm])
    dists, miss = [], 0
    for p in test_ids:
        if p not in esm:
            miss += 1; continue
        q = pool(esm[p]).float().unsqueeze(0)
        dists.append(torch.cdist(q, train).min().item())
    dists = np.array(dists)
    rows.append((c, dists.mean(), dists.std(), len(dists), miss, len(train)))

rows.sort(key=lambda r: -r[1])
print(f"\n{'cluster':8} {'meanNNdist':>11} {'std':>7} {'n_test':>7} {'missing':>8} {'n_train':>8}")
for c,m,s,n,miss,nt in rows:
    print(f"{c:8} {m:11.3f} {s:7.3f} {n:7d} {miss:8d} {nt:8d}")
