# Glossary — codenames to paper labels

The configuration files and prediction tables carry the internal codenames the
experiments were run under. This table decodes them to the labels used in the
paper. The checkpoint folders and top-level data files have already been renamed
to readable names; this glossary is the key for the codenames that remain inside
`configs/` and in the `config_id` column of `manifest.csv`.

## Reading the codenames

A readout family has three angular settings. The codename encodes them as a
suffix: the plain name is Lmax = 0, `_mp1` is Lmax = 1, `_mp2` is Lmax = 2.
"L0/L1/L2" below is that maximum angular degree of the message passing.

## Headline models

| codename | paper label | what it is |
|---|---|---|
| `u44` | atom model / `Atom-ρCG-L0` | atom scale, 5 layers, 192 channels, RMSE 1.258 |
| `fr7` | residue model / `Res-ρCG-L0` | residue scale, 2 layers, 160 channels, RMSE 1.282 |

## Paired-readout families (Table 3)

| codename group | paper label | readout |
|---|---|---|
| `u44`, `zc10…mp1`, `zc12…mp2` | `Atom-ρCG-{L0,L1,L2}` | density re-embedding, ⊗^ρ_CG |
| `zc40`, `zc35…mp1`, `zc36…mp2` | `Atom-CG-{L0,L1,L2}` | direct tensor product, ⊗_CG |
| `zc37`, `zc38`, `zc39` | `Atom-Δ-{L0,L1,L2}` | contraction against unbound copy |
| `zc32`, `zc33`, `zc34` | `Atom-ΔMLP-{L0,L1,L2}` | Δ followed by an MLP |
| `zc47`, `zc48`, `zc49` | `Atom-Δsym-{L0,L1,L2}` | symmetric Δ |
| `zc50`, `zc51`, `zc52` | `Atom-ΔsymMLP-{L0,L1,L2}` | symmetric Δ followed by an MLP |
| `zc53`, `zc54`, `zc55` | `Atom-normΔ2-{L0,L1,L2}` | squared norm of Δ |
| `zc56`, `zc57`, `zc58` | `Atom-overlap-{L0,L1,L2}` | inner product of bound and unbound |
| `zc18`, `zc18…_lmax1`, `zc18…_lmax2` | `Atom-noPR-{L0,L1,L2}` | no paired readout |

## Vanilla configurations (Table 1)

`zc19` through `zc30` decode directly from their names to
`Vanilla-{1c,2c}-{5L,2L}-L{0,1,2}`, where `1c`/`2c` is the number of copies and
`5L`/`2L` is the layer count. For example `zc23_vanilla_2copy_lmax0_2L` is
`Vanilla-2c-2L-L0`.

## Ablation controls (Table 2)

| codename | paper label |
|---|---|
| `zc1_no_node_onsite` | remove atom comparison |
| `zc2_no_edge_onsite` | remove edge comparison |
| `zc3_no_atom_sh` | remove atom global comparison |
| `zc4_no_residue_sh` | remove residue global comparison |
| `zc5_no_feedback` | remove feedback |
| `zc6_no_persist_edges` | remove edge updates |
| `zc7_no_all_onsite` | remove local comparisons |
| `zc8_1copy` | single copy |
| `zc9_no_all_sh` | remove global comparisons |

## Residue models (Tables 6, 7, 11, 12)

| codename | paper label | readout |
|---|---|---|
| `fr7` | `Res-ρCG-L0` | density re-embedding |
| `frrho1`, `frrho2` | `Res-ρCG-L1`, `Res-ρCG-L2` | density re-embedding |
| `frcg1`, `frcg2` | `Res-CG-L1`, `Res-CG-L2` | direct tensor product |
| `frdsym0`, `frdsym1`, `frdsym2` | `Res-Δsym-{L0,L1,L2}` | symmetric Δ |
| `fr7nos`, `frv0` | `Res-noPR-L0` | no paired readout |

`fr7nos` removes the onsite comparison; `frv0` is the same setting reached with
the readout mode set to none. Both are the no-paired-readout residue control.

## Data file names

The files were renamed from their run-time names:

| current name | was |
|---|---|
| `per_sample/predictions_casf2016_headline_ensembles.csv` | `per_sample_ensemble_official.csv` |
| `per_sample/predictions_casf2016_ablations.csv` | `per_sample_all_ablation.csv` |
| `per_sample/predictions_casf2016_residue.csv` | `per_sample_fr7.csv` |
| `ensembles/ood_preFT_ensembles.csv` | `ood_baseline_official.csv` |
| `ensembles/ood_postFT_joint_ensembles.csv` | `ood_ft_joint_ensembles.csv` |
| `ensembles/ood_preFT_residue_survey.txt` | `full_ensemble_3archs_jun15.txt` |
| `ensembles/ood_postFT_residue_survey.txt` | `postft_ensemble_3archs_jun15.txt` |
| `ensembles/ood_joint_atom_residue.txt` | `joint_u44_residue_ood.txt` |
