# Equivariant MPNN with paired readouts

Code for the paper on paired readouts for protein–ligand binding affinity
prediction. The model is an equivariant message-passing network that carries two
copies of the pocket — bound and unbound — and compares them at readout.

Trained weights and the predictions behind every table are deposited separately
(see *Data* below); this repository holds the code and the run configurations.

## Layout

```
mpnn_2copies_datamax1_v3/
  train.py            training entry point
  models/             the network: message passing, onsite comparisons, readouts
  configs/
    cleansplit/       CleanSplit training and cross-validation
    ood/              held-out pocket cluster runs
src/layers/           tensor product convolution the model builds on
scripts/              reproduction, ensembling, evaluation, figures
docs/
  manifest.csv        every reported number -> the file it came from
  checkpoint_provenance.csv   each released checkpoint -> the number it produces
  GLOSSARY.md         config codenames -> the paper's labels
```

Configuration files carry the codenames the experiments ran under (`u44`, `fr7`,
`zc47`, …). `docs/GLOSSARY.md` decodes them.

## The two models

Both are ensembles of five models trained on five splits, predictions averaged.
RMSE is on CASF-2016 (285 complexes).

| model | pocket representation | size | RMSE |
|---|---|---|---|
| atom | one node per pocket atom | 5 layers, 192 channels | 1.258 |
| residue | one node per pocket residue | 2 layers, 160 channels | 1.282 |

Averaging all ten gives 1.233.

## Running

```bash
python mpnn_2copies_datamax1_v3/train.py \
  --config mpnn_2copies_datamax1_v3/configs/cleansplit/U44_5fold_cv/u44_f0.yaml \
  --seed 42 --device cuda
```

A run writes its checkpoints, metrics, and a copy of the source it was trained
with into its own output directory. That archived copy is what the released
checkpoints load against, so a checkpoint stays loadable after the code moves.

## Reproducing the tables

`scripts/reproduce_tables.py` recomputes the reported values from the prediction
files and prints any that disagree. It reads the data from the deposit, so fetch
that first and point the script at it.

`docs/manifest.csv` is the record behind it: one row per reported number, naming
the source file, the metric, the value, and whether it recomputes.

## Data

Trained weights, per-complex predictions, out-of-distribution ensembles, and the
noise sweep are deposited at [Zenodo DOI pending].

Training data is built from PDBbind and CASF-2016 by the preprocessing scripts;
the held-out clusters come from PLINDER (2024-06/v2), pocket lDDT communities at
the 50% threshold.

## Environment

```
python 3.10   torch 2.5.1+cu124   e3nn 0.5.4
torch_geometric 2.6.1   numpy 1.26.4   scipy 1.13.1
```

`e3nn` 0.5.4 matters — the tensor product API shifted in later releases. Results
were checked to reproduce under 0.5.5 within run-to-run variation.

Because scatter operations on GPU accumulate in nondeterministic order, repeated
runs at one seed differ by roughly 0.02–0.03 in RMSE. Comparisons in the paper
are made across five splits rather than between single runs.
