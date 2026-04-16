#!/bin/bash

# Environment Setup
source /data1/vikram/miniconda3/etc/profile.d/conda.sh
conda activate QSVD
export PYTHONPATH=$PYTHONPATH:$(pwd)/fake_quant

# Configuration
MODEL="llava-hf/llava-v1.6-vicuna-7b-hf"
DATASET="ScienceQA_Train"
EVAL_DATASET="ScienceQA_TEST"
W_BITS=4
A_BITS=4
K_BITS=16
V_BITS=16

# Percentiles to sweep
PERCENTILES=(10 15 20 25 30 40 50)

# Base directory for experiments
EXP_DIR="threshold_experiments_w4a4"
mkdir -p $EXP_DIR

# Results file
RESULTS_FILE="$EXP_DIR/results_summary.csv"
echo "Percentile,Sparsity,Accuracy" > $RESULTS_FILE

# Function to run experiment
run_exp() {
    local P=$1
    local SETTING=$2
    local EXTRA_ARGS=""
    if [ "$P" != "0" ]; then
        EXTRA_ARGS="--threshold_percentile $P"
    fi

    echo "Running experiment for Percentile: $P% ($SETTING)..."
    local LOG="$EXP_DIR/${SETTING}.log"
    if [ ! -f "$LOG" ] || ! grep -q "Overall" "$LOG"; then
        python -u fake_quant/mainllavanext.py \
            --model $MODEL \
            --a_bits $A_BITS \
            --w_bits $W_BITS \
            --k_bits $K_BITS \
            --v_bits $V_BITS \
            --cal_dataset $DATASET \
            --eval_dataset $EVAL_DATASET \
            --tasks None \
            --w_rtn \
            --w_clip \
            --a_clip_ratio 0.9 \
            --nsamples 256 \
            --vitnsamples 256 \
            --seed 0 \
            --qkv_fuse \
            --basepath "/data1/vikram/QVLM/" \
            $EXTRA_ARGS \
            --setting "$SETTING" \
            --rotate \
            --vit_module \
            --beta_then_svd > "$LOG" 2>&1
    else
        echo "Log for $SETTING already exists. Skipping experiment and extracting results."
    fi
    
    # Extract results
    local LOG="$EXP_DIR/${SETTING}.log"
    # Extraction logic for VLMEvalKit output
    local ACC=$(grep -i "Overall" "$LOG" | tail -n 1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9.]+$/) {print $i; exit}}')
    [ -z "$ACC" ] && ACC="N/A"

    local SPARSITY=$(grep "Total Sparsity in Attention Layers:" "$LOG" | tail -n 1 | awk '{print $NF}')
    [ -z "$SPARSITY" ] && SPARSITY="0%"
    
    echo "$P,$SPARSITY,$ACC" >> $RESULTS_FILE
}

# Run Baseline
run_exp 0 "baseline_w4a4"

# Run Sweep
for P in "${PERCENTILES[@]}"
do
    run_exp $P "threshold_${P}_w4a4"
done

echo "Sweep complete. Results saved in $RESULTS_FILE"
cat $RESULTS_FILE
