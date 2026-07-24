"""Pre-FT 5-fold ensemble for the new residue archs (frrho1, frrho2, frdsym0).

Adapts ensemble_ood_preft.py to:
- Look under experiments/ood/<arch>_residue/<arch>_<cluster>_f<fold>/ (not flat)
- Use v3_residue/ood/ood_<cluster>_test_5A.pt (residue data, not atom)
- Partial-tolerant: report N-of-5 ensembles for any cluster with >=1 fold done

Usage:
    python scripts/ensemble_residue_archs_preft.py --archs frrho1,frrho2,frdsym0
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import scipy.stats
import torch
import yaml
from torch_geometric.loader import DataLoader

os.chdir('/nfs/lambda_stor_01/homes/wongr/good_affinity_predictors/binding_gnn')
sys.path.insert(0, 'mpnn_2copies_datamax1_v3')
sys.path.insert(0, 'mpnn_2copies_datamax1_v3/models')
sys.path.insert(0, '.')
from train import config_to_model_config, PrepareData_2copies

device = torch.device('cuda:0')
OOD_BASE = Path('mpnn_2copies_datamax1_v3/experiments/ood')
CLUSTERS = ['1nvq', '1sqa', '2p15', '2vw5', '3dd0', '3f3e', '3o9i']
FOLDS = [0, 1, 2, 3, 4]


def load_model(source_dir, model_cfg, device):
    for mn in list(sys.modules.keys()):
        if mn in ('model', 'global_sh', 'onsite', 'tpconv',
                  'edge_onsite', 'virtual_onsite', 'readout_fusion'):
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
    test_path = f'gign_exact_v1/data/v3_residue/ood/ood_{cluster}_test_5A.pt'
    if not Path(test_path).exists():
        return None
    test_data = torch.load(test_path, weights_only=False)

    fold_preds, per_fold = [], []
    targets_arr, loader = None, None

    arch_dir = OOD_BASE / f'{arch}_residue'
    for fold in FOLDS:
        tag_dir = arch_dir / f'{arch}_{cluster}_f{fold}'
        if not tag_dir.exists():
            per_fold.append((fold, None)); continue
        # newest run dir that has best_model_val.pt
        cands = sorted([d for d in tag_dir.iterdir()
                        if d.is_dir() and (d / 'best_model_val.pt').exists()],
                       key=lambda p: p.name, reverse=True)
        if not cands:
            per_fold.append((fold, None)); continue
        run_dir = cands[0]
        with open(run_dir / 'config.yaml') as f:
            cfg = yaml.safe_load(f)
        model_cfg = config_to_model_config(cfg)
        if loader is None:
            transform = PrepareData_2copies(
                use_inter_edges=cfg.get('data', {}).get('use_inter_edges', True),
                use_ext_features=cfg.get('data', {}).get('use_ext_features', False),
                use_bfactor=cfg.get('data', {}).get('use_bfactor', False),
            )
            loader = DataLoader([transform(d) for d in test_data], batch_size=64, shuffle=False)
        model = load_model(str(run_dir / 'source'), model_cfg, device)
        ckpt = torch.load(run_dir / 'best_model_val.pt', map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
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
        'per_fold': per_fold,
        'n_folds': len(fold_preds),
        'ensemble_rmse': ens_rmse,
        'ensemble_r': ens_r,
        'ensemble_rho': ens_rho,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', required=True)
    args = ap.parse_args()

    for arch in args.archs.split(','):
        arch = arch.strip()
        print(f'\n========== {arch} (pre-FT N/5-fold ensemble per cluster) ==========')
        ens_rs = []
        for cluster in CLUSTERS:
            res = ensemble_one(arch, cluster)
            if res is None:
                print(f'  {cluster}: NO FOLDS')
                continue
            print(f'  {cluster} ({res["n_folds"]}/5 folds): '
                  f'ensemble RMSE={res["ensemble_rmse"]:.4f}  R={res["ensemble_r"]:.4f}  rho={res["ensemble_rho"]:.4f}')
            for fold, m in res['per_fold']:
                if m is None:
                    print(f'    f{fold}: skip')
                else:
                    print(f'    f{fold}: RMSE={m[0]:.4f} R={m[1]:.4f}')
            ens_rs.append(res['ensemble_r'])
        if ens_rs:
            print(f'  ── {arch} mean of available cluster-ensembles R = {np.mean(ens_rs):.4f}  (N={len(ens_rs)})')


if __name__ == '__main__':
    main()
