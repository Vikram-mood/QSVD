#!/bin/bash
# Magnitude sparsification experiment
# Usage: bash scripts/run_magnitude.sh [rank_ratio] [seed] [cache_file] [basepath]

rank_ratio=${1:-0.9}
seed=${2:-0}
cache_file=${3:-"../cache_file/llava-next-7b"}
basepath=${4:-"../"}

cd "$(dirname "$0")/../fake_quant"

python magnitude_experiment.py \
    --model llava-hf/llava-v1.6-vicuna-7b-hf \
    --nsamples 16 \
    --vitnsamples 16 \
    --seed "$seed" \
    --cal_dataset ScienceQA_Train \
    --eval_dataset ScienceQA_TEST \
    --cache_file "$cache_file" \
    --basepath "$basepath" \
    --calib_method abs_mean \
    --rank_ratio "$rank_ratio" \
    --act_aware \
    --had_rank \
    --qkv_fuse \
    --grad_info \
    --use_cache true \
    --percentiles 50 70 80 90 95 99 \
    --run_svd_baseline \
    --svd_rank_ratio 0.5 \
    --setting "magnitude_sweep/seed${seed}" \
    --a_clip_ratio 0.9
