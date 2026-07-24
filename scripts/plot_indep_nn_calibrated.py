"""Nearest-neighbor distance to training for CASF indep vs non-indep, CALIBRATED.

The original version (plot_indep_nn_distances.py) plotted raw NN distances, which
have no scale: "far from training" is uninterpretable without knowing how far a
training complex is from its OWN nearest training neighbor.

This version adds that reference. Each panel is normalized by the median
train->nearest-other-train distance, so the reference sits at 1.0. Then:
  < 1.0  = closer to training than training complexes are to each other
  ~ 1.0  = as close as training complexes are to each other (i.e. ordinary)
  > 1.0  = genuinely farther from training than the training set's own spacing

Usage:
  python scripts/plot_indep_nn_calibrated.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "gign_exact_v1" / "data" / "v3_with_residues"
EMB = PROJECT_ROOT / "gign_exact_v1" / "data"


def load_pdb_ids(pt_path):
    return [d.pdb_id for d in torch.load(pt_path, weights_only=False)]


def pool(e):
    return e.mean(0) if e.dim() > 1 else e


def nn_to(train_embs, query_embs):
    """Min distance from each query row to any training row."""
    return torch.cdist(query_embs, train_embs).min(dim=1).values


def main():
    casf_full = load_pdb_ids(DATA_DIR / "cleansplit_casf2016_5A.pt")
    indep_ids = set(load_pdb_ids(DATA_DIR / "cleansplit_casf2016_indep_5A.pt"))
    train_ids = load_pdb_ids(DATA_DIR / "cleansplit_f0_train_5A.pt")

    spaces = [
        ("ESM2\n(pocket sequence)", EMB / "esm_embeddings/esm2_pocket_embeddings.pt"),
        ("ANKH\n(pocket sequence)", EMB / "ankh_embeddings/ankh_pocket_embeddings.pt"),
        ("ChemBERTa\n(ligand)", EMB / "chemberta_embeddings/chemberta_cls_embeddings.pt"),
        ("ATOMICA\n(complex)", EMB / "atomica_embeddings/atomica_graph_embeddings.pt"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)

    for ax, (name, path) in zip(axes, spaces):
        d = torch.load(path, map_location="cpu", weights_only=False)

        trk = [p for p in train_ids if p in d]
        tr = torch.stack([pool(d[p]) for p in trk]).float()

        # in-distribution reference: train -> nearest OTHER train
        D = torch.cdist(tr, tr)
        D.fill_diagonal_(float("inf"))
        ref_d = D.min(dim=1).values
        ref = ref_d.median().item()
        del D

        ind = [p for p in casf_full if p in indep_ids and p in d]
        non = [p for p in casf_full if p not in indep_ids and p in d]
        ind_d = nn_to(tr, torch.stack([pool(d[p]) for p in ind]).float())
        non_d = nn_to(tr, torch.stack([pool(d[p]) for p in non]).float())

        # normalize by the training set's own internal spacing
        R = ref_d.numpy() / ref
        N = non_d.numpy() / ref
        I = ind_d.numpy() / ref

        _, pval = mannwhitneyu(I, N, alternative="two-sided")
        _, p_far = mannwhitneyu(I, R, alternative="greater")

        parts = ax.violinplot([R, N, I], positions=[0, 1, 2],
                              showmeans=False, showmedians=True, showextrema=False)
        face = ["#d1d5db", "#93c5fd", "#fca5a5"]
        edge = ["#6b7280", "#2563eb", "#dc2626"]
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(face[i])
            pc.set_edgecolor(edge[i])
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")

        ax.axhline(1.0, color="#6b7280", ls="--", lw=1, zorder=0)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([f"Training\n(n={len(R)})",
                            f"Non-indep.\n({len(N)})",
                            f"Indep.\n({len(I)})"], fontsize=15)
        ax.set_title(name, fontsize=18)
        # common scale: everything is in units of the training set's own NN distance,
        # so all four panels must share one axis for the 1.0 line to be comparable.
        ax.set_ylim(0, 2.5)

        # median ratio under each violin
        for x, v in zip([1, 2], [N, I]):
            ax.text(x, 0.04, f"{np.median(v):.2f}×", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=15, fontweight="bold",
                    color=edge[x])

        star = "***" if pval < 1e-3 else ("*" if pval < 0.05 else "n.s.")
        ax.text(0.5, 0.97, f"indep vs non-indep: {star}\np = {pval:.1e}" if pval < 1e-3
                else f"indep vs non-indep: {star}\np = {pval:.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=13,
                color="#dc2626" if pval < 0.05 else "#6b7280")

        print(f"{name.splitlines()[0]:10s} ref={ref:.4f}  non={np.median(N):.2f}x  "
              f"indep={np.median(I):.2f}x  indep>ref p={p_far:.3f}  ind-vs-non p={pval:.2e}")

    axes[0].set_ylabel("NN distance to training\n(÷ training's own NN distance)", fontsize=15)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=13)
    plt.tight_layout()

    out = PROJECT_ROOT / "scripts" / "sh_analysis_plots" / "casf_indep_nn_calibrated.pdf"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
