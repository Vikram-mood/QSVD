#!/bin/bash
source /data1/vikram/miniconda3/etc/profile.d/conda.sh
conda activate QSVD
cd /data1/vikram/QSVD/QSVD
python -u /data1/vikram/QSVD/QSVD/fake_quant/run_threshold_sweep.py \
    --model llava-hf/llava-v1.6-vicuna-7b-hf  \
    --a_bits 4 \
    --w_bits 4 \
    --k_bits 16 \
    --v_bits 16 \
    --cal_dataset ScienceQA_Train \
    --eval_dataset ScienceQA_TEST \
    --nsamples 32 \
    --vitnsamples 32 \
    --seed 0 \
    --qkv_fuse \
    --grad_info \
    --basepath "/data1/vikram/QVLM/" \
    --setting "threshold_sweep_experiment"
