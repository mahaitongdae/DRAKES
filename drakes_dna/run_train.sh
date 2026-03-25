#!/bin/bash
export DRAKES_DATA_BASE_PATH=/home/ubuntu/haitong-south-2/DRAKES/data_and_model
wandb_group=gr_linear_compare

mkdir -p logs/${wandb_group}

set -euo pipefail

cleanup() {
    local code=$?
    echo "Stopping all running jobs..."
    # Kill the entire process group for this script.
    kill -- -$$ >/dev/null 2>&1 || true
    exit "$code"
}

trap cleanup INT TERM

source .venv/bin/activate

run_gpu_tasks() {
    local gpu_count=$1
    local reweight_type=$2
    local temp=$3
    local lower_bound=$4
    shift 4
    local seeds=$@

    local timestamp=$(date +%Y%m%d_%H%M%S)
    for seed in $seeds; do
        CUDA_VISIBLE_DEVICES=$gpu_count python finetune_weighted_bc.py --bootstrap_from_new_model true --reweight_type $reweight_type --advantage_temp $temp --lower_bound $lower_bound \
        --seed $seed > logs/${wandb_group}/${gpu_count}_${timestamp}_r${reweight_type}_t${temp}_lb${lower_bound}_s${seed}.log 2>&1 
    done
}

run_gpu_tasks 0 wd1 1.0 0.0 1 2 &
run_gpu_tasks 1 wd1 1.0 -0.5 0 2 &
run_gpu_tasks 2 wd1 1.0 -1.0 0 1 &
run_gpu_tasks 3 wd1 1.0 -0.3 0 2 &    

wait
echo "All tasks finished at $(date)"