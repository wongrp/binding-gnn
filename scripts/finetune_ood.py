#!/usr/bin/env python3
"""Fine-tune a pretrained OOD checkpoint on a small holdout set.

Standalone script. Does NOT modify train.py.

Usage:
    python scripts/finetune_ood.py \
      --checkpoint experiments/ood/u44_1nvq_f0/2026-03-04_1_s42/best_model_val.pt \
      --config configs/ood/u44_1nvq_f0.yaml \
      --holdout-data holdouts/ood_1nvq_holdout_n25_seed42.pt \
      --eval-data holdouts/ood_1nvq_eval_n25_seed42.pt \
      --strategy full \
      --ft-epochs 25 --ft-lr 1e-4 \
      --output-dir experiments/ood_finetuning/u44_1nvq_f0_full_n25
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add paths for model imports
sys.path.insert(0, str(Path(__file__).parent.parent / "mpnn_2copies_datamax1_v3"))
sys.path.insert(0, str(Path(__file__).parent.parent / "mpnn_2copies_datamax1_v3" / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import load_config, config_to_model_config, PrepareData_2copies, PrepareData


def load_model_from_source(source_dir, model_cfg, device):
    """Load EquivariantMPNN using archived source code from experiment dir.

    This handles the case where model.py or its dependencies have changed
    since the checkpoint was trained.
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        # Fallback to current code
        from model import EquivariantMPNN
        return EquivariantMPNN(model_cfg).to(device)

    # Inject archived source directory at front of sys.path so imports resolve there
    source_str = str(source_dir)
    sys.path.insert(0, source_str)
    try:
        # Force reload of model module from archived source
        if "model" in sys.modules:
            del sys.modules["model"]
        # Also reload dependencies that may have changed
        for mod_name in ["global_sh", "onsite", "edge_onsite", "virtual_onsite", "readout_fusion"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        from model import EquivariantMPNN
        return EquivariantMPNN(model_cfg).to(device)
    finally:
        sys.path.remove(source_str)


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    all_preds, all_targets, all_pdb_ids = [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred, _ = model(batch)
        all_preds.append(pred.cpu().view(-1))
        all_targets.append(batch.y.cpu().view(-1))
        if hasattr(batch, "pdb_id"):
            ids = batch.pdb_id
            if isinstance(ids, (list, tuple)):
                all_pdb_ids.extend(list(ids))
            else:
                all_pdb_ids.extend([str(x) for x in ids])
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    mask = np.isfinite(preds) & np.isfinite(targets)
    p_clean, t_clean = preds[mask], targets[mask]
    rmse = float(np.sqrt(np.mean((p_clean - t_clean) ** 2)))
    mae = float(np.mean(np.abs(p_clean - t_clean)))
    pearson = float(np.corrcoef(p_clean, t_clean)[0, 1]) if len(p_clean) > 1 else 0.0
    spearman = float(scipy.stats.spearmanr(p_clean, t_clean)[0]) if len(p_clean) > 1 else 0.0
    metrics = {"rmse": rmse, "mae": mae, "pearson": pearson, "spearman": spearman}
    if return_preds:
        return metrics, preds, targets, all_pdb_ids
    return metrics


def apply_freeze_strategy(model, strategy):
    """Freeze parameters according to strategy."""
    if strategy == "full":
        return  # all trainable

    # Identify parameter name prefixes to keep trainable
    trainable_prefixes = ["head."]
    if strategy == "last_layer":
        n_layers = len(model.layers)
        trainable_prefixes.append(f"layers.{n_layers - 1}.")
        trainable_prefixes.append("node_proj.")

    for name, param in model.named_parameters():
        if not any(name.startswith(p) for p in trainable_prefixes):
            param.requires_grad = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Strategy '{strategy}': {n_train:,} / {n_total:,} params trainable "
          f"({100*n_train/n_total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OOD checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--holdout-data", type=str, required=True)
    parser.add_argument("--eval-data", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["full", "head", "last_layer"])
    parser.add_argument("--ft-epochs", type=int, default=25)
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--ft-wd", type=float, default=1e-4)
    parser.add_argument("--ft-min-lr", type=float, default=0.0,
                        help="Minimum LR for cosine schedule (0 = no schedule, flat LR)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--source-dir", type=str, default=None,
                        help="Path to archived source/ dir from experiment (for compat)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--esm-readout-dim", type=int, default=0,
                        help="ESM readout projection dim (0 = no ESM). Adds esm_proj + new head.")
    parser.add_argument("--esm-embeddings", type=str, default=None,
                        help="Path to precomputed ESM embeddings .pt file")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    t0 = time.monotonic()

    # Load config and build model
    cfg = load_config(args.config)
    model_cfg = config_to_model_config(cfg)

    # Auto-detect source dir from checkpoint path if not specified
    source_dir = args.source_dir
    if source_dir is None:
        ckpt_parent = Path(args.checkpoint).parent
        candidate = ckpt_parent / "source"
        if candidate.exists():
            source_dir = str(candidate)
            print(f"Auto-detected source dir: {source_dir}")

    # Build model — if ESM requested, build with ESM dims so head is the right size
    if args.esm_readout_dim > 0:
        model_cfg_esm = model_cfg
        # Override esm fields in the ModelConfig
        from dataclasses import fields as dc_fields
        # Infer ESM input dim from the embeddings file
        esm_embs = torch.load(args.esm_embeddings, map_location="cpu", weights_only=False)
        esm_dim = next(iter(esm_embs.values())).shape[0]
        del esm_embs
        overrides = {"esm_readout_dim": args.esm_readout_dim, "esm_input_dim": esm_dim}
        model_cfg_esm = type(model_cfg)(**{
            f.name: overrides.get(f.name, getattr(model_cfg, f.name))
            for f in dc_fields(model_cfg)
        })
        model = load_model_from_source(source_dir, model_cfg_esm, device)

        # Warm-start: load pretrained weights into the ESM-augmented model
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
        old_state = ckpt["model_state_dict"]
        new_state = model.state_dict()

        for key in new_state:
            if key in old_state and old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]
            elif key == "head.0.weight":
                # First head layer grew: [H, old_dim] -> [H, old_dim + esm_readout_dim]
                old_w = old_state[key]  # [H, old_dim]
                new_state[key][:, :old_w.shape[1]] = old_w
                new_state[key][:, old_w.shape[1]:] = 0.0  # zero-init ESM columns
                print(f"  Warm-start head.0.weight: {old_w.shape} -> {new_state[key].shape}")
            elif key == "head.0.bias":
                new_state[key] = old_state[key]
            elif key.startswith("esm_proj"):
                pass  # keep random init for ESM projection
            elif key in old_state:
                print(f"  WARNING: shape mismatch for {key}: {old_state[key].shape} vs {new_state[key].shape}")
            else:
                print(f"  New key (random init): {key}")

        model.load_state_dict(new_state)
        print(f"Loaded checkpoint with ESM warm-start (epoch {ckpt.get('epoch', '?')})")

        # Freeze everything except head + esm_proj
        for name, param in model.named_parameters():
            if name.startswith("head") or name.startswith("esm_proj"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"ESM finetune: {n_train:,} / {n_total:,} params trainable ({100*n_train/n_total:.1f}%)")
    else:
        model = load_model_from_source(source_dir, model_cfg, device)

        # Load checkpoint
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

        # Apply freeze strategy
        apply_freeze_strategy(model, args.strategy)

    # Prepare data transform
    data_cfg = cfg.get("data", {})
    use_inter = data_cfg.get("use_inter_edges", True)
    use_2copies = data_cfg.get("use_2copies", False)
    drop_isolated = data_cfg.get("drop_isolated_nodes", False)
    use_ext_features = data_cfg.get("use_ext_features", False)
    use_bfactor = data_cfg.get("use_bfactor", False)
    ext_feature_mask = data_cfg.get("ext_feature_mask", None)
    if use_2copies:
        transform = PrepareData_2copies(
            use_inter_edges=use_inter, drop_isolated_nodes=drop_isolated,
            use_ext_features=use_ext_features, use_bfactor=use_bfactor,
            ext_feature_mask=ext_feature_mask,
        )
    else:
        transform = PrepareData(use_inter_edges=use_inter, drop_isolated_nodes=drop_isolated)

    # Load and transform data
    holdout_raw = torch.load(args.holdout_data, weights_only=False)
    eval_raw = torch.load(args.eval_data, weights_only=False)
    holdout_data = [transform(d) for d in holdout_raw]
    eval_data = [transform(d) for d in eval_raw]
    print(f"Holdout: {len(holdout_data)}, Eval: {len(eval_data)}")

    # Attach ESM embeddings if requested
    if args.esm_embeddings:
        esm_dict = torch.load(args.esm_embeddings, map_location="cpu", weights_only=False)
        esm_dim = next(iter(esm_dict.values())).shape[0]
        zero_emb = torch.zeros(esm_dim)
        n_found = 0
        for dataset in [holdout_data, eval_data]:
            for d in dataset:
                emb = esm_dict.get(d.pdb_id)
                if emb is not None:
                    d.esm_embedding = emb.unsqueeze(0)
                    n_found += 1
                else:
                    d.esm_embedding = zero_emb.unsqueeze(0)
        print(f"ESM embeddings: {n_found}/{len(holdout_data)+len(eval_data)} found")
        del esm_dict

    holdout_loader = DataLoader(holdout_data, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    # Evaluate baseline (before finetuning) — save baseline preds for ensembling
    criterion = nn.MSELoss()
    baseline_holdout = evaluate(model, holdout_loader, criterion, device)
    baseline_eval, base_preds, base_targets, base_ids = evaluate(
        model, eval_loader, criterion, device, return_preds=True)
    print(f"Baseline holdout: RMSE={baseline_holdout['rmse']:.4f} R={baseline_holdout['pearson']:.4f}")
    print(f"Baseline eval:    RMSE={baseline_eval['rmse']:.4f} R={baseline_eval['pearson']:.4f}")

    # Fine-tuning
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.ft_lr, weight_decay=args.ft_wd)
    scheduler = None
    if args.ft_min_lr > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.ft_epochs, eta_min=args.ft_min_lr)

    best_holdout_loss = float("inf")
    best_eval_metrics = None
    best_epoch = 0
    best_state_dict = None

    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in holdout_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred, _ = model(batch)
            loss = criterion(pred, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 10.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        train_loss = total_loss / n_batches

        # Evaluate on holdout (for model selection) and eval (for reporting)
        holdout_metrics = evaluate(model, holdout_loader, criterion, device)
        eval_metrics = evaluate(model, eval_loader, criterion, device)

        print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
              f"Holdout RMSE: {holdout_metrics['rmse']:.4f} | "
              f"Eval RMSE: {eval_metrics['rmse']:.4f} R: {eval_metrics['pearson']:.4f}")

        # Save best model based on holdout RMSE
        if holdout_metrics["rmse"] < best_holdout_loss:
            best_holdout_loss = holdout_metrics["rmse"]
            best_eval_metrics = eval_metrics
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "strategy": args.strategy,
        "ft_epochs": args.ft_epochs,
        "ft_lr": args.ft_lr,
        "ft_min_lr": args.ft_min_lr,
        "ft_wd": args.ft_wd,
        "esm_readout_dim": args.esm_readout_dim,
        "checkpoint": args.checkpoint,
        "holdout_data": args.holdout_data,
        "eval_data": args.eval_data,
        "n_holdout": len(holdout_data),
        "n_eval": len(eval_data),
        "baseline_holdout": baseline_holdout,
        "baseline_eval": baseline_eval,
        "best_epoch": best_epoch,
        "best_holdout_rmse": best_holdout_loss,
        "best_eval": best_eval_metrics,
        "elapsed_seconds": round(time.monotonic() - t0, 1),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save per-sample predictions for downstream ensembling.
    def _save_preds_csv(path, ids, preds, targets):
        with open(path, "w") as f:
            f.write("pdb_id,target,pred\n")
            for i in range(len(preds)):
                pid = ids[i] if i < len(ids) else f"idx_{i}"
                f.write(f"{pid},{float(targets[i]):.6f},{float(preds[i]):.6f}\n")
    _save_preds_csv(out_dir / "preds_baseline.csv", base_ids, base_preds, base_targets)

    # Predictions at best FT epoch — reload best state, re-evaluate on eval set.
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        _, ft_preds, ft_targets, ft_ids = evaluate(
            model, eval_loader, criterion, device, return_preds=True)
        _save_preds_csv(out_dir / "preds_ft.csv", ft_ids, ft_preds, ft_targets)

    if best_state_dict is not None:
        torch.save({
            "epoch": best_epoch,
            "model_state_dict": best_state_dict,
        }, out_dir / "best_model_ft.pt")

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best eval: RMSE={best_eval_metrics['rmse']:.4f} "
          f"R={best_eval_metrics['pearson']:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
