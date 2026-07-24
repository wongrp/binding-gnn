"""Generate residue OOD configs for 3 new architectures by template-modifying fr7_residue.

Architectures:
  frrho1  — residue + ⊗^ρ_CG at backbone L=1  (analogue of zc10 at residue scale)
  frrho2  — residue + ⊗^ρ_CG at backbone L=2  (analogue of zc12)
  frdsym0 — residue + Δ_sym at backbone L=0   (analogue of zc47; Q1 mechanism test)

Each architecture: 7 clusters × 5 folds = 35 configs.

Usage:
  python scripts/gen_residue_ood_configs.py                    # default: 800 epochs
  python scripts/gen_residue_ood_configs.py --epochs 1200      # 1200-epoch variants
Writes to mpnn_2copies_datamax1_v3/configs/ood/{arch}_residue[_1200]/{arch}_{cluster}_f{fold}.yaml.
"""
import argparse
import re
from pathlib import Path

ROOT = Path("/nfs/lambda_stor_01/homes/wongr/good_affinity_predictors/binding_gnn")
SRC_DIR = ROOT / "mpnn_2copies_datamax1_v3" / "configs" / "ood" / "fr7_residue"
OUT_BASE = ROOT / "mpnn_2copies_datamax1_v3" / "configs" / "ood"
EXP_BASE = "mpnn_2copies_datamax1_v3/experiments/ood"

CLUSTERS = ["1nvq", "1sqa", "2p15", "2vw5", "3dd0", "3f3e", "3o9i"]
FOLDS = [0, 1, 2, 3, 4]

# Per-architecture diffs from the fr7_residue (lmax=0, 320x0e, reembed_cg) template
ARCHS = {
    "frrho1": dict(
        description="residue 2L + ⊗^ρ_CG at backbone L=1 (zc10 analogue at residue scale)",
        irreps_hidden="224x0e + 224x1o",
        lmax=1,
        onsite_mode="reembed_cg",
        onsite_reembed_lmax=1,
    ),
    "frrho2": dict(
        description="residue 2L + ⊗^ρ_CG at backbone L=2 (zc12 analogue at residue scale)",
        irreps_hidden="160x0e + 160x1o + 160x2e",
        lmax=2,
        onsite_mode="reembed_cg",
        onsite_reembed_lmax=1,
    ),
    "frdsym0": dict(
        description="residue 2L + Δ_sym at backbone L=0 (zc47 analogue at residue scale; Q1 mechanism test)",
        irreps_hidden="384x0e",
        lmax=0,
        scalars=160,
        onsite_mode="delta_sym",
        # delta_sym has no reembed_lmax / cg_mode — those will be removed
    ),
    "frdsym1": dict(
        description="residue 2L + Δ_sym at backbone L=1 (zc48 analogue at residue scale; completes residue Δ_sym L sweep)",
        irreps_hidden="224x0e + 224x1o",
        lmax=1,
        onsite_mode="delta_sym",
    ),
    "frdsym2": dict(
        description="residue 2L + Δ_sym at backbone L=2 (zc49 analogue at residue scale; completes residue Δ_sym L sweep)",
        irreps_hidden="160x0e + 160x1o + 160x2e",
        lmax=2,
        onsite_mode="delta_sym",
    ),
    "frcg1": dict(
        description="residue 2L + ⊗_CG (direct CG, no density expansion) at backbone L=1 (zc35 analogue at residue scale; tests whether density re-embedding or graph scale carries the noise robustness)",
        irreps_hidden="224x0e + 224x1o",
        lmax=1,
        onsite_mode="cg",
        onsite_reembed_lmax=1,
    ),
    "frcg2": dict(
        description="residue 2L + ⊗_CG (direct CG, no density expansion) at backbone L=2 (zc36 analogue at residue scale; tests whether density re-embedding or graph scale carries the noise robustness)",
        irreps_hidden="160x0e + 160x1o + 160x2e",
        lmax=2,
        onsite_mode="cg",
        onsite_reembed_lmax=1,
    ),
}


def patch_yaml(text, arch_name, diff, cluster, fold, epochs, out_arch_suffix=""):
    """Apply per-arch overrides + redirect data/output paths."""
    out = text

    # epochs override (in train: block)
    out = re.sub(r'^(\s*)epochs:\s*\d+',
                 rf'\g<1>epochs: {epochs}',
                 out, count=1, flags=re.M)

    # Description (lineage block)
    out = re.sub(
        r'(\s*description:)\s*.*',
        rf'\1 "OOD {cluster} fold {fold}, {arch_name} ({diff["description"]})"',
        out,
        count=2,  # first hit is lineage.description; we want to update both occurrences
    )

    # irreps_hidden — match only the model: section (exactly 2-space indent),
    # not the lineage.changes block (4-space indent)
    new_irreps = diff["irreps_hidden"]
    quoted = f'"{new_irreps}"' if "+" in new_irreps else new_irreps
    out = re.sub(r'^  irreps_hidden:.*$',
                 f'  irreps_hidden: {quoted}',
                 out, count=1, flags=re.M)

    # lmax (model section only — 2-space indent)
    out = re.sub(r'^  lmax:\s*\d+', f'  lmax: {diff["lmax"]}', out, count=1, flags=re.M)

    # onsite_mode — anchor at line start so we don't touch edge_onsite_mode
    out = re.sub(r'^(\s*)onsite_mode:\s*\S+',
                 rf'\g<1>onsite_mode: "{diff["onsite_mode"]}"',
                 out, flags=re.M)

    # onsite_reembed_lmax — keep for rho variants; remove or zero for delta_sym
    if "onsite_reembed_lmax" in diff:
        out = re.sub(r'onsite_reembed_lmax:\s*\d+',
                     f'onsite_reembed_lmax: {diff["onsite_reembed_lmax"]}', out)
    else:
        # delta_sym: drop the reembed_lmax and onsite_cg_mode lines
        out = re.sub(r'^\s*onsite_reembed_lmax:.*\n', '', out, flags=re.M)
        out = re.sub(r'^\s*onsite_cg_mode:.*\n', '', out, flags=re.M)

    # Output directory rerouting (last block)
    out = re.sub(
        r'output:\s*\n\s*dir:\s*\S+',
        f'output:\n  dir: {EXP_BASE}/{arch_name}_residue{out_arch_suffix}',
        out,
    )

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=800,
                    help="Training epochs (default 800 = fr7_residue convention; 1200 matches atom OOD)")
    args = ap.parse_args()

    # 1200-ep variants get an explicit suffix so they live alongside 800-ep configs
    suffix = "" if args.epochs == 800 else f"_{args.epochs}"

    n_written = 0
    for arch_name, diff in ARCHS.items():
        out_dir = OUT_BASE / f"{arch_name}_residue{suffix}"
        out_dir.mkdir(exist_ok=True)
        for cluster in CLUSTERS:
            for fold in FOLDS:
                src = SRC_DIR / f"fr7_{cluster}_f{fold}.yaml"
                if not src.exists():
                    print(f"  MISSING template {src}")
                    continue
                text = src.read_text()
                patched = patch_yaml(text, arch_name, diff, cluster, fold,
                                     epochs=args.epochs, out_arch_suffix=suffix)
                out_path = out_dir / f"{arch_name}_{cluster}_f{fold}.yaml"
                out_path.write_text(patched)
                n_written += 1
        print(f"  {arch_name}: {len(CLUSTERS)*len(FOLDS)} configs written to {out_dir}")
    print(f"\nTotal: {n_written} configs across {len(ARCHS)} architectures (epochs={args.epochs}).")


if __name__ == "__main__":
    main()
