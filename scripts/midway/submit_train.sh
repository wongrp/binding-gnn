#!/bin/bash
# Generic Midway Slurm submitter for a single train.py run.
#
# Usage: sbatch -J <jobname> scripts/midway/submit_train.sh <config.yaml> [seed]
#   seed defaults to 42.
#
# Run from the repo root. $BINDING_GNN_ROOT should be set in ~/.bashrc; falls
# back to the current dir.
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/midway/%x_%j.out

set -e
cd "${BINDING_GNN_ROOT:-$PWD}"

CFG="${1:?usage: submit_train.sh <config.yaml> [seed]}"
SEED="${2:-42}"

source ~/.bashrc 2>/dev/null || true
conda activate nn_dta_copy

mkdir -p logs/midway
echo "[$(date)] host=$(hostname) cfg=$CFG seed=$SEED"
python mpnn_2copies_datamax1_v3/train.py --config "$CFG" --seed "$SEED" --device cuda
