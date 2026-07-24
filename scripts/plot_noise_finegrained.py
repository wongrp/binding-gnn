#!/usr/bin/env python3
"""Plot the fine-grained noise-robustness sweep for stratify_test.tex.

Reads results/noise_robustness_finegrained_merged.csv and produces
paper_pnas/figs/noise_dose_response_finegrained.pdf.

Style: single-panel dose-response, log-spaced sigma axis (with sigma=0
shown as a separate left column), 9 curves, shaded vertical bands for
refinement / thermal / apo-holo / docking noise regimes.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "noise_robustness_finegrained_merged.csv"
OUT_FULL = ROOT / "paper_pnas" / "figs" / "noise_dose_response_finegrained.pdf"
OUT_LEAN = ROOT / "paper_pnas" / "figs" / "noise_dose_response_lean.pdf"
OUT_ARCH = ROOT / "paper_pnas" / "figs" / "noise_dose_response_arch.pdf"

# Lean (deployment) figure: canonical, residue, vanilla baselines.
LEAN_CONFIGS = {
    "u44_dual_scale_sh", "fr7_f0",
    "zc20_vanilla_2copy", "zc21_vanilla_2copy_lmax1", "zc22_vanilla_2copy_lmax2",
}
# Architectural-sensitivity figure: all four atom paired-readout families across
# L_max where defined, to show (i) higher L_max → less robust, (ii) readout
# family matters.
ARCH_CONFIGS = {
    # ⊗_CG^ρ family (canonical)
    "u44_dual_scale_sh", "zc10_u44_mp1_par_matched", "zc12_u44_mp2_par_matched",
    # Δ+MLP family
    # zc32/33/34 (original asymmetric Δ+MLP) excluded; the paper reports the
    # symmetric variants zc50/51/52 below.
    # ⊗_CG family
    "zc40_u44_direct_cg", "zc35_u44_mp1_direct_cg", "zc36_u44_mp2_direct_cg",
    # Δ family (symmetric-reference contraction — the paper's reported Δ)
    "zc47_u44_delta_sym", "zc48_u44_mp1_delta_sym", "zc49_u44_mp2_delta_sym",
    # Δ+MLP family (symmetric-reference contraction)
    "zc50_u44_delta_sym_mlp", "zc51_u44_mp1_delta_sym_mlp", "zc52_u44_mp2_delta_sym_mlp",
    # ‖Δ‖² family (norm-squared difference)
    "zc53_u44_delta_norm2", "zc54_u44_mp1_delta_norm2", "zc55_u44_mp2_delta_norm2",
    # ⟨h_b, h_ub⟩ family (overlap)
    "zc56_u44_overlap", "zc57_u44_mp1_overlap", "zc58_u44_mp2_overlap",
    # NOTE: original asymmetric-reference Δ (zc37-39) and Δ+MLP (zc32-34) are
    # preserved in the merged CSV but excluded from this figure; the paper
    # reports the symmetric-reference variants.
}


CONFIGS = [
    # (csv_name, display_label, color, linestyle, linewidth, marker, group)
    # Headline architectures (canonical MPNN+PR variants)
    ("u44_dual_scale_sh",          r"MPNN+PR ($\otimes_{\mathrm{CG}}^{\rho}$, $L_{\max}{=}0$)", "#1F77B4", "-",  2.0, "o", "PR"),
    ("fr7_f0",                     "MPNN+PR (residue, 2L)",                                      "#D62728", "-",  2.0, "s", "PR"),
    # ⊗_CG^ρ family across L_max
    ("zc10_u44_mp1_par_matched",   r"MPNN+PR ($\otimes_{\mathrm{CG}}^{\rho}$, $L_{\max}{=}1$)",  "#1F77B4", "--", 1.4, "o", "PR"),
    ("zc12_u44_mp2_par_matched",   r"MPNN+PR ($\otimes_{\mathrm{CG}}^{\rho}$, $L_{\max}{=}2$)",  "#1F77B4", ":",  1.4, "o", "PR"),
    # Δ+MLP family across L_max
    # Original asymmetric Δ+MLP (zc32-34) removed; symmetric variants below
    # are the paper's reported Δ+MLP. Commented-out entries for re-rendering
    # are at the end of this list.
    # ⊗_CG family across L_max
    ("zc40_u44_direct_cg",         r"MPNN+PR ($\otimes_{\mathrm{CG}}$, $L_{\max}{=}0$)",         "#9467BD", "-",  1.6, "*", "PR"),
    ("zc35_u44_mp1_direct_cg",     r"MPNN+PR ($\otimes_{\mathrm{CG}}$, $L_{\max}{=}1$)",         "#9467BD", "--", 1.6, "*", "PR"),
    ("zc36_u44_mp2_direct_cg",     r"MPNN+PR ($\otimes_{\mathrm{CG}}$, $L_{\max}{=}2$)",         "#9467BD", ":",  1.6, "*", "PR"),
    # Δ family (symmetric-reference contraction; the paper's reported Δ)
    ("zc47_u44_delta_sym",            r"MPNN+PR ($\Delta$, $L_{\max}{=}0$)",                     "#FF7F0E", "-",  1.4, "X", "PR"),
    ("zc48_u44_mp1_delta_sym",        r"MPNN+PR ($\Delta$, $L_{\max}{=}1$)",                     "#FF7F0E", "--", 1.4, "X", "PR"),
    ("zc49_u44_mp2_delta_sym",        r"MPNN+PR ($\Delta$, $L_{\max}{=}2$)",                     "#FF7F0E", ":",  1.4, "X", "PR"),
    # Δ+MLP family (symmetric-reference contraction)
    ("zc50_u44_delta_sym_mlp",        r"MPNN+PR ($\Delta$+MLP, $L_{\max}{=}0$)",                 "#2CA02C", "-",  1.6, "P", "PR"),
    ("zc51_u44_mp1_delta_sym_mlp",    r"MPNN+PR ($\Delta$+MLP, $L_{\max}{=}1$)",                 "#2CA02C", "--", 1.6, "P", "PR"),
    ("zc52_u44_mp2_delta_sym_mlp",    r"MPNN+PR ($\Delta$+MLP, $L_{\max}{=}2$)",                 "#2CA02C", ":",  1.6, "P", "PR"),
    # ‖Δ‖² family (norm-squared of bound-unbound difference; scalar readout, no angular output)
    ("zc53_u44_delta_norm2",          r"MPNN+PR ($\|\Delta\|^2$, $L_{\max}{=}0$)",               "#17BECF", "-",  1.4, "h", "PR"),
    ("zc54_u44_mp1_delta_norm2",      r"MPNN+PR ($\|\Delta\|^2$, $L_{\max}{=}1$)",               "#17BECF", "--", 1.4, "h", "PR"),
    ("zc55_u44_mp2_delta_norm2",      r"MPNN+PR ($\|\Delta\|^2$, $L_{\max}{=}2$)",               "#17BECF", ":",  1.4, "h", "PR"),
    # ⟨h_b, h_ub⟩ family (inner product; scalar readout)
    ("zc56_u44_overlap",              r"MPNN+PR ($\langle h_b,h_{ub}\rangle$, $L_{\max}{=}0$)",  "#E377C2", "-",  1.4, "D", "PR"),
    ("zc57_u44_mp1_overlap",          r"MPNN+PR ($\langle h_b,h_{ub}\rangle$, $L_{\max}{=}1$)",  "#E377C2", "--", 1.4, "D", "PR"),
    ("zc58_u44_mp2_overlap",          r"MPNN+PR ($\langle h_b,h_{ub}\rangle$, $L_{\max}{=}2$)",  "#E377C2", ":",  1.4, "D", "PR"),
    # NOTE: original asymmetric-reference Δ (zc37-39) and Δ+MLP (zc32-34) entries
    # are retained below (commented out) for historical reference. To render the
    # original numbers, swap these in and remove the symmetric variants above.
    # ("zc37_u44_linear_onsite",     r"MPNN+PR ($\Delta_{\mathrm{asym}}$, $L_{\max}{=}0$)",      "#B85C00", "-",  1.4, "X", "PR"),
    # ("zc38_u44_mp1_linear_onsite", r"MPNN+PR ($\Delta_{\mathrm{asym}}$, $L_{\max}{=}1$)",      "#B85C00", "--", 1.4, "X", "PR"),
    # ("zc39_u44_mp2_linear_onsite", r"MPNN+PR ($\Delta_{\mathrm{asym}}$, $L_{\max}{=}2$)",      "#B85C00", ":",  1.4, "X", "PR"),
    # ("zc32_u44_scalar_onsite",     r"MPNN+PR ($\Delta_{\mathrm{asym}}$+MLP, $L_{\max}{=}0$)",  "#1B5E20", "-",  1.6, "P", "PR"),
    # ("zc33_u44_mp1_scalar_onsite", r"MPNN+PR ($\Delta_{\mathrm{asym}}$+MLP, $L_{\max}{=}1$)",  "#1B5E20", "--", 1.6, "P", "PR"),
    # ("zc34_u44_mp2_scalar_onsite", r"MPNN+PR ($\Delta_{\mathrm{asym}}$+MLP, $L_{\max}{=}2$)",  "#1B5E20", ":",  1.6, "P", "PR"),
    # Component ablations (greyed out)
    ("zc31_u44_shlmax0",           r"ablation (scalar pocket SH, $\ell{=}0$)",                    "#888888", ":",  1.0, "v", "PR"),
    ("zc7_no_all_onsite",          "ablation (no atom/edge readouts)",                            "#888888", "--", 1.0, "^", "PR"),
    ("zc9_no_all_sh",              "ablation (no pocket readouts)",                               "#888888", "-.", 1.0, "D", "PR"),
    ("zc18_2copy_pool_both",       "ablation (no paired readouts)",                               "#444444", "-",  1.0, "x", "PR"),
    # Vanilla MPNN baselines (no paired readouts at all)
    ("zc20_vanilla_2copy",         r"vanilla $L_{\max}{=}0$",   "#666666", "--", 1.4, "o", "vanilla"),
    ("zc21_vanilla_2copy_lmax1",   r"vanilla $L_{\max}{=}1$",   "#666666", "--", 1.4, "s", "vanilla"),
    ("zc22_vanilla_2copy_lmax2",   r"vanilla $L_{\max}{=}2$",   "#666666", "--", 1.4, "^", "vanilla"),
]

# Vanilla colors: greys, distinguished by marker
VANILLA_COLORS = {"zc20_vanilla_2copy": "#999999",
                  "zc21_vanilla_2copy_lmax1": "#666666",
                  "zc22_vanilla_2copy_lmax2": "#333333"}

# Color override: keep CONFIGS' choices (no override; legacy hook)
COLOR_OVERRIDE = {}


# Noise regime bands: (sigma_min, sigma_max, label, color)
REGIMES = [
    (0.05, 0.15, "refinement",          "#E8F4FD"),
    (0.20, 0.40, "thermal",              "#E8F8E8"),
    (0.50, 1.00, "apo--holo / cryo-EM",  "#FDF8E0"),
    (1.00, 2.00, "docked / predicted",   "#FCEAE8"),
]


def main():
    df = pd.read_csv(CSV)

    # Pull in 20-draw redo data for suspect (config, seed) pairs (overrides 3-draw).
    import os, glob, re
    redo_files = glob.glob(str(ROOT / "results/noise_robustness_redo/*_20draws.csv"))
    cfg_map = {
        "u44": "u44_dual_scale_sh", "zc18": "zc18_2copy_pool_both",
        "zc32": "zc32_u44_scalar_onsite", "zc10": "zc10_u44_mp1_par_matched",
        "zc33": "zc33_u44_mp1_scalar_onsite", "zc35": "zc35_u44_mp1_direct_cg",
    }
    redo_pairs = set()
    redo_frames = []
    for f in redo_files:
        name = os.path.basename(f).replace("_20draws.csv", "").replace("_partial87", "")
        m = re.match(r"(\w+?)_s(\d+)$", name)
        if not m:
            continue
        short, seed = m.group(1), "s" + m.group(2)
        config = cfg_map.get(short, short)
        rd = pd.read_csv(f)
        rd["experiment"] = config
        rd["seed"] = seed
        redo_frames.append(rd)
        redo_pairs.add((config, seed))
    if redo_frames:
        df = df[~df.apply(lambda r: (r.experiment, r.seed) in redo_pairs, axis=1)]
        df = pd.concat([df] + redo_frames, ignore_index=True)

    # Per (config, seed, sigma) median over draws (robust to outlier draws);
    # then mean and IQR across seeds at each sigma.
    per_seed = df.groupby(["experiment", "seed", "sigma"], as_index=False)["rmse"].median()
    agg = per_seed.groupby(["experiment", "sigma"], as_index=False)["rmse"].mean()
    spread = per_seed.groupby(["experiment", "sigma"], as_index=False)["rmse"].agg(
        lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).rename(columns={"rmse": "iqr"})
    agg = agg.merge(spread, on=["experiment", "sigma"])

    desired_order = [
        # canonical and residue
        r"MPNN+PR ($\otimes_{\mathrm{CG}}^{\rho}$, $L_{\max}{=}0$)",
        "MPNN+PR (residue, 2L)",
        # ⊗_CG^ρ across L_max
        r"MPNN+PR ($\otimes_{\mathrm{CG}}^{\rho}$, $L_{\max}{=}1$)",
        r"MPNN+PR ($\otimes_{\mathrm{CG}}^{\rho}$, $L_{\max}{=}2$)",
        # Δ+MLP across L_max
        r"MPNN+PR ($\Delta$+MLP, $L_{\max}{=}0$)",
        r"MPNN+PR ($\Delta$+MLP, $L_{\max}{=}1$)",
        r"MPNN+PR ($\Delta$+MLP, $L_{\max}{=}2$)",
        # ⊗_CG across L_max
        r"MPNN+PR ($\otimes_{\mathrm{CG}}$, $L_{\max}{=}0$)",
        r"MPNN+PR ($\otimes_{\mathrm{CG}}$, $L_{\max}{=}1$)",
        r"MPNN+PR ($\otimes_{\mathrm{CG}}$, $L_{\max}{=}2$)",
        # Δ across L_max
        r"MPNN+PR ($\Delta$, $L_{\max}{=}0$)",
        r"MPNN+PR ($\Delta$, $L_{\max}{=}1$)",
        r"MPNN+PR ($\Delta$, $L_{\max}{=}2$)",
        # ablations
        r"ablation (scalar pocket SH, $\ell{=}0$)",
        "ablation (no atom/edge readouts)",
        "ablation (no pocket readouts)",
        "ablation (no paired readouts)",
        # vanillas
        r"vanilla $L_{\max}{=}0$",
        r"vanilla $L_{\max}{=}1$",
        r"vanilla $L_{\max}{=}2$",
    ]

    def _plot_panel(ax, sigma_max, configs_filter=None, ylim=None, show_labels=True):
        """Plot one σ-axis panel; configs_filter restricts to a subset by csv_name."""
        for sigma_min_b, sigma_max_b, label, color in REGIMES:
            if sigma_min_b >= sigma_max:
                continue
            ax.axvspan(sigma_min_b, min(sigma_max_b, sigma_max), color=color, alpha=0.55, zorder=0)
        for cfg_name, label, _color, ls, lw, marker, group in CONFIGS:
            if configs_filter is not None and cfg_name not in configs_filter:
                continue
            sub = agg[agg["experiment"] == cfg_name].sort_values("sigma")
            if sub.empty:
                continue
            sub = sub[sub["sigma"] <= sigma_max + 1e-6]
            sigmas = sub["sigma"].values
            rmses = sub["rmse"].values
            iqrs = sub["iqr"].values
            color = VANILLA_COLORS.get(cfg_name, COLOR_OVERRIDE.get(cfg_name, _color))
            if cfg_name in ("u44_dual_scale_sh", "fr7_f0"):
                ax.fill_between(sigmas, rmses - iqrs/2, rmses + iqrs/2,
                                color=color, alpha=0.12, zorder=1.5)
            ax.plot(sigmas, rmses, color=color, linestyle=ls, linewidth=lw,
                    marker=marker, markersize=5, label=label,
                    zorder=3 if group == "PR" else 2,
                    alpha=0.95 if group == "PR" else 0.8)
        # matplotlib mathtext has no \AA; use the literal glyph or it prints "\AA".
        ax.set_xlabel("Position noise $\\sigma$ (Å)", fontsize=11)
        ax.set_ylabel("CASF-2016 test RMSE (pK)", fontsize=11)
        ax.set_xlim(-0.02 * sigma_max, sigma_max * 1.02)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":", zorder=1)
        ax.tick_params(axis="both", labelsize=10)
        if show_labels:
            ymax = ax.get_ylim()[1]
            label_y = ymax + 0.012
            for sigma_min_b, sigma_max_b, label, _ in REGIMES:
                x_mid = 0.5 * (sigma_min_b + min(sigma_max_b, sigma_max))
                if x_mid > sigma_max:
                    continue
                ax.text(x_mid, label_y, label, ha="center", va="bottom",
                        fontsize=8, color="#444444", style="italic", clip_on=False)

    def _save_two_panel(out_path, configs_filter, full_ylim, thermal_ylim):
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                       gridspec_kw={"width_ratios": [1.0, 1.6]})
        _plot_panel(axA, sigma_max=0.40, configs_filter=configs_filter, ylim=thermal_ylim)
        _plot_panel(axB, sigma_max=2.10, configs_filter=configs_filter, ylim=full_ylim)
        # Panel labels in axes corner (avoids collision with regime labels above)
        axA.text(0.02, 0.97, "(a)", transform=axA.transAxes, fontsize=12,
                 fontweight="bold", va="top", ha="left")
        axB.text(0.02, 0.97, "(b)", transform=axB.transAxes, fontsize=12,
                 fontweight="bold", va="top", ha="left")

        # Shared legend on the right of panel B
        handles, labels = axB.get_legend_handles_labels()
        ordered = sorted(zip(handles, labels),
                         key=lambda hl: desired_order.index(hl[1]) if hl[1] in desired_order else 99)
        handles, labels = zip(*ordered)
        axB.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=8, ncol=1, frameon=False)
        # Drop the redundant legend on panel A
        if axA.get_legend() is not None:
            axA.get_legend().remove()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
        print(f"wrote {out_path}")
        print(f"wrote {out_path.with_suffix('.png')}")
        plt.close(fig)

    # Full 17-line figure (stratify_test, appendix)
    _save_two_panel(OUT_FULL, configs_filter=None,
                    full_ylim=(1.27, 2.05), thermal_ylim=(1.27, 1.50))
    # Lean figure (main.tex Fig 1): canonical + residue + vanilla baselines
    _save_two_panel(OUT_LEAN, configs_filter=LEAN_CONFIGS,
                    full_ylim=(1.27, 1.85), thermal_ylim=(1.27, 1.46))
    # Architectural-sensitivity figure (main.tex Fig 2): ⊗_CG^ρ family vs Δ+MLP family
    _save_two_panel(OUT_ARCH, configs_filter=ARCH_CONFIGS,
                    full_ylim=(1.27, 1.95), thermal_ylim=(1.27, 1.46))


if __name__ == "__main__":
    main()
