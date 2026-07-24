#!/usr/bin/env python3
"""Generate eval predictions for fr7 finetuned checkpoints (which don't save
eval_preds.npy), then compute ensemble analysis combining with u44 finetuned
predictions (which DO save eval_preds.npy)."""

import copy
import csv
import importlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import yaml
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FT_DIR = PROJECT_ROOT / "mpnn_2copies_datamax1_v3/experiments/ood_finetuning"
CLUSTERS = ["1nvq", "1sqa", "2p15", "2vw5", "3dd0", "3f3e", "3o9i"]
DEVICE = "cuda:0"


def load_model(source_dir, config_path, ckpt_path, device):
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
    for k in ["esm_embeddings_path", "esm_input_dim", "esm_readout_dim", "esm_proj_dim"]:
        cfg.get("model", {}).pop(k, None)
        cfg.get("data", {}).pop(k, None)
    model_cfg = archived_train.config_to_model_config(cfg)
    model = model_mod.EquivariantMPNN(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    del sys.modules["train"]
    sys.path.insert(0, str(PROJECT_ROOT / "mpnn_2copies_datamax1_v3"))
    current_train = importlib.import_module("train")
    sys.path.pop(0)
    return model, cfg, current_train


def predict_on(model, cfg, train_mod, eval_data_path):
    raw = torch.load(str(eval_data_path), weights_only=False)
    data_cfg = cfg.get("data", {})
    kw = dict(
        use_inter_edges=data_cfg.get("use_inter_edges", True),
        use_ext_features=data_cfg.get("use_ext_features", False),
        use_bfactor=data_cfg.get("use_bfactor", False),
    )
    if data_cfg.get("use_2copies", True):
        transform = train_mod.PrepareData_2copies(**kw)
    else:
        transform = train_mod.PrepareData(use_inter_edges=kw["use_inter_edges"])
    transformed = [transform(copy.deepcopy(d)) for d in raw]
    loader = DataLoader(transformed, batch_size=64, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            out = model(b)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.view(-1).cpu().numpy())
            targets.append(b.y.view(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def get_fr7_source_and_config(cluster, fold):
    """Find the original fr7 OOD training run's source/config to use as template
    (finetune dirs don't archive source)."""
    base = PROJECT_ROOT / f"mpnn_2copies_datamax1_v3/experiments/ood/fr7_residue/fr7_{cluster}_f{fold}"
    run = sorted([d for d in base.iterdir() if d.is_dir() and (d / "source").exists()])
    if not run:
        return None, None
    r = run[-1]
    return r / "source", r / "config.yaml"


def generate_fr7_preds(cluster, n_ft):
    """For each fold, load the fr7 finetuned checkpoint and produce eval preds.
    Returns list of (preds, y, val_rmse) per fold."""
    results = []
    for f in range(5):
        ft_dir = FT_DIR / f"fr7_residue/fr7_{cluster}_f{f}_full_n{n_ft}"
        ckpt_path = ft_dir / "best_model_ft.pt"
        metrics_path = ft_dir / "metrics.json"
        if not ckpt_path.exists() or not metrics_path.exists():
            return None
        m = json.loads(metrics_path.read_text())
        eval_path = PROJECT_ROOT / m["eval_data"]
        source_dir, config_path = get_fr7_source_and_config(cluster, f)
        if source_dir is None:
            return None
        model, cfg, train_mod = load_model(source_dir, config_path, ckpt_path, DEVICE)
        preds, y = predict_on(model, cfg, train_mod, eval_path)
        del model; torch.cuda.empty_cache()
        val_rmse = m.get("best_holdout_rmse")
        results.append((preds, y, val_rmse, m))
    return results


def load_u44_preds(cluster, n_ft):
    """Load already-saved u44 eval predictions."""
    results = []
    for f in range(5):
        ft_dir = FT_DIR / f"u44_{cluster}_f{f}_full_n{n_ft}"
        preds_p = ft_dir / "eval_preds.npy"
        metrics_p = ft_dir / "metrics.json"
        if not (preds_p.exists() and metrics_p.exists()):
            return None
        preds = np.load(preds_p, allow_pickle=True).astype(float)
        m = json.loads(metrics_p.read_text())
        eval_path = PROJECT_ROOT / m["eval_data"]
        d = torch.load(str(eval_path), weights_only=False)
        y = np.array([float(s.y.item() if hasattr(s.y, "item") else s.y) for s in d])
        val_rmse = m.get("best_holdout_rmse")
        results.append((preds, y, val_rmse, m))
    return results


def analyze(cluster, u44_data, fr7_data):
    u_preds = [d[0] for d in u44_data]
    f_preds = [d[0] for d in fr7_data]
    y = u44_data[0][1]
    # Sanity: all u44 folds have same y
    for d in u44_data + fr7_data:
        if not np.allclose(d[1], y):
            return {"error": "y mismatch", "cluster": cluster}
    u_vals = [d[2] for d in u44_data]
    f_vals = [d[2] for d in fr7_data]

    def R(p): return float(scipy.stats.pearsonr(p, y).statistic)
    def RMSE(p): return float(np.sqrt(np.mean((p - y) ** 2)))

    u5 = np.mean(u_preds, axis=0)
    f5 = np.mean(f_preds, axis=0)
    t10 = np.mean(u_preds + f_preds, axis=0)

    # Val-ranked
    u_rank = sorted(range(5), key=lambda i: u_vals[i] if u_vals[i] else 1e9)
    f_rank = sorted(range(5), key=lambda i: f_vals[i] if f_vals[i] else 1e9)
    val_2u3f = np.mean([u_preds[i] for i in u_rank[:2]] + [f_preds[i] for i in f_rank[:3]], axis=0)
    val_3u2f = np.mean([u_preds[i] for i in u_rank[:3]] + [f_preds[i] for i in f_rank[:2]], axis=0)

    # Oracle
    best_R = -1; best_RMSE = 999
    for u_idx in combinations(range(5), 3):
        for f_idx in combinations(range(5), 2):
            mix = np.mean([u_preds[i] for i in u_idx] + [f_preds[i] for i in f_idx], axis=0)
            r = R(mix)
            if r > best_R:
                best_R = r; best_RMSE = RMSE(mix)

    # Error correlation
    rho = float(scipy.stats.pearsonr(u5 - y, f5 - y).statistic)

    return {
        "cluster": cluster, "n_eval": len(y),
        "u44_5f_R": R(u5), "u44_5f_RMSE": RMSE(u5),
        "fr7_5f_R": R(f5), "fr7_5f_RMSE": RMSE(f5),
        "tenfold_R": R(t10), "tenfold_RMSE": RMSE(t10),
        "val_2u3f_R": R(val_2u3f), "val_2u3f_RMSE": RMSE(val_2u3f),
        "val_3u2f_R": R(val_3u2f), "val_3u2f_RMSE": RMSE(val_3u2f),
        "oracle_3u2f_R": best_R, "oracle_3u2f_RMSE": best_RMSE,
        "err_rho": rho,
    }


def main():
    n_ft = 25
    out_rows = []
    preds_cache_dir = PROJECT_ROOT / "results/fr7_ft_preds"
    preds_cache_dir.mkdir(parents=True, exist_ok=True)

    for cluster in CLUSTERS:
        print(f"\n============ {cluster.upper()} ============", flush=True)
        u44_data = load_u44_preds(cluster, n_ft)
        if u44_data is None:
            print(f"  SKIP: missing u44 data for {cluster}", flush=True)
            continue
        # Generate / cache fr7 preds
        cache_path = preds_cache_dir / f"fr7_{cluster}_n{n_ft}.npz"
        if cache_path.exists():
            print(f"  loading cached fr7 preds from {cache_path}", flush=True)
            d = np.load(cache_path, allow_pickle=True)
            fr7_preds_list = [d[f"p{f}"] for f in range(5)]
            fr7_ys = [d[f"y{f}"] for f in range(5)]
            fr7_vals = list(d["vals"])
            fr7_data = list(zip(fr7_preds_list, fr7_ys, fr7_vals, [None]*5))
        else:
            print(f"  generating fr7 preds (5 folds)...", flush=True)
            fr7_data = generate_fr7_preds(cluster, n_ft)
            if fr7_data is None:
                print(f"  SKIP: fr7 generation failed", flush=True)
                continue
            # Save cache
            save = {"vals": np.array([d[2] if d[2] is not None else -1 for d in fr7_data])}
            for i, d in enumerate(fr7_data):
                save[f"p{i}"] = d[0]
                save[f"y{i}"] = d[1]
            np.savez_compressed(cache_path, **save)
            print(f"  cached → {cache_path}", flush=True)

        r = analyze(cluster, u44_data, fr7_data)
        if "error" in r:
            print(f"  {cluster} ERROR: {r['error']}", flush=True)
            # debug info
            u_y = u44_data[0][1]; f_y = fr7_data[0][1]
            print(f"    u44 y[:3]={u_y[:3]}, fr7 y[:3]={f_y[:3]}", flush=True)
            print(f"    u44 y sorted[:3]={np.sort(u_y)[:3]}, fr7 y sorted[:3]={np.sort(f_y)[:3]}", flush=True)
            # Try y-alignment: sort both by label and align
            # If labels are same set but different order, we can match by argsort
            if len(u_y) == len(f_y) and np.allclose(np.sort(u_y), np.sort(f_y)):
                print(f"    Labels match up to permutation; aligning by sort order", flush=True)
                u_order = np.argsort(u_y)
                f_order = np.argsort(f_y)
                aligned_u = [(d[0][u_order], d[1][u_order], d[2], d[3]) for d in u44_data]
                aligned_f = [(d[0][f_order], d[1][f_order], d[2], d[3]) for d in fr7_data]
                r = analyze(cluster, aligned_u, aligned_f)
            if "error" in r:
                continue
        out_rows.append(r)
        print(f"  n={r['n_eval']:4d}  u44_5f={r['u44_5f_R']:+.3f}  fr7_5f={r['fr7_5f_R']:+.3f}  "
              f"10fold={r['tenfold_R']:+.3f}  val_2u3f={r['val_2u3f_R']:+.3f}  "
              f"val_3u2f={r['val_3u2f_R']:+.3f}  oracle={r['oracle_3u2f_R']:+.3f}  ρ={r['err_rho']:+.3f}", flush=True)

    # Mean across clusters
    if out_rows:
        print(f"\n=== Mean R across {len(out_rows)} clusters ===", flush=True)
        for k in ["u44_5f_R", "fr7_5f_R", "tenfold_R", "val_2u3f_R", "val_3u2f_R", "oracle_3u2f_R"]:
            vals = [r[k] for r in out_rows]
            print(f"  {k:<16}  mean = {np.mean(vals):.4f}", flush=True)

    out = PROJECT_ROOT / "results/ood_finetune_ensemble.csv"
    if out_rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            for r in out_rows:
                w.writerow(r)
        print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
