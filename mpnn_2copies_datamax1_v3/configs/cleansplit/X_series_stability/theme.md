# X Series: Training Stability

Parent: u44_dual_scale_sh (dual-scale SH, atom + residue)

Goal: reduce run-to-run variance and improve val-selected checkpoint quality.

Motivation: u44 f0 replicates (8 runs, lambda1) show val-selected RMSE spread of
1.307-1.367, with val-gated picks consistently better and earlier (ep 208-584 vs
val-selected ep 689-774). The cosine schedule's second half adds noise without
improving generalization.

All configs are identical to u44 except for the noted change.

| Config | Change from u44 | Rationale |
|--------|----------------|-----------|
| x1 | lr=1.5e-4 | Lower peak lr, less oscillation |
| x2 | min_lr=5e-5 | Higher lr floor, less overshoot at end |
| x3 | cosine_T_max=800 | Reach floor by ep 800, coast 400ep at low lr |
| x4 | batch_size=128 | Smoother gradients |
| x5 | warmup=20 | Gentler ramp-up |
| x6 | lr=1.5e-4, min_lr=2e-5, cosine_T_max=900 | Combined mild adjustments |
| x7 | channels=320, head_hidden=192 | Smaller backbone, bigger head |
| x8 | wd=5e-5 | Less regularization pressure |
| x9 | min_lr=7.5e-5 | Between x2 and x10 |
| x10 | min_lr=1e-4 | Highest floor (only 2x decay from peak) |

## Multi-seed variance study (min_lr floor sweep)

All on f0, seeds 48-55 where possible.

| Config | min_lr | Seeds | Machine |
|--------|--------|-------|---------|
| u44 (baseline) | 1e-5 | 48-55 (8 runs) | lambda1 |
| x2 | 5e-5 | 48-55 (8 runs) | lambda2 |
| x9 | 7.5e-5 | 48-49 (2 runs) | lambda0 |
| x10 | 1e-4 | 48-50 (3 runs) | lambda0 |
