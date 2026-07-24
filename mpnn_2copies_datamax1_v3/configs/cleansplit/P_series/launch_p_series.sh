#!/bin/bash
# Launch P_series (patience=150 reruns of top-13 models).
# All 8 GPUs on lambda2; round 2 launched as GPUs free up.
#
# Round 1 (8 GPUs):
#   GPU 0: p1_d2   (~7h)    GPU 4: p5_h2   (~11h)
#   GPU 1: p2_f13  (~13h)   GPU 5: p6_d15  (~16h)
#   GPU 2: p3_e8   (~10h)   GPU 6: p7_e14  (~9h)
#   GPU 3: p4_e11  (~13h)   GPU 7: p8_d18  (~10h)
#
# Round 2 (as GPUs free from round 1):
#   p9_d5 (~2h), p10_d1 (~7h), p11_d3 (~2h), p12_d31 (~11h), p13_h3 (~9h)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CFG_DIR="mpnn_2copies_datamax1_v3/configs/cleansplit/P_series"
LOG_DIR="mpnn_2copies_datamax1_v3/experiments/cleansplit/P_series"
mkdir -p "$LOG_DIR"

launch() {
    local gpu=$1 cfg=$2
    echo "GPU $gpu: $cfg"
    CUDA_VISIBLE_DEVICES=$gpu nohup conda run -n nn_dta_copy \
        python mpnn_2copies_datamax1_v3/train.py \
        --config "${CFG_DIR}/${cfg}.yaml" \
        > "${LOG_DIR}/${cfg}.log" 2>&1 &
}

echo "=== Round 1 ==="
launch 0 p1_d2
launch 1 p2_f13
launch 2 p3_e8
launch 3 p4_e11
launch 4 p5_h2
launch 5 p6_d15
launch 6 p7_e14
launch 7 p8_d18

echo "Round 1 launched. PIDs:"
jobs -p
echo ""
echo "Round 2 (launch manually when GPUs free):"
echo "  launch <gpu> p9_d5"
echo "  launch <gpu> p10_d1"
echo "  launch <gpu> p11_d3"
echo "  launch <gpu> p12_d31"
echo "  launch <gpu> p13_h3"
