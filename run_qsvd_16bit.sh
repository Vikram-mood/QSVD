#!/bin/bash
source /data1/vikram/miniconda3/etc/profile.d/conda.sh
conda activate QSVD
cd /data1/vikram/QSVD/QSVD

# QSVD Experiment (Importance-based)
# - Enabled --grad_info (QSVD core)
# - Enabled --beta_then_svd
# - 16-bit to match magnitude baseline

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u /data1/vikram/QSVD/QSVD/fake_quant/mainllavanext.py \
    --model llava-hf/llava-v1.6-vicuna-7b-hf  \
    --a_bits 16 \
    --w_bits 16 \
    --k_bits 16 \
    --v_bits 16 \
    --cal_dataset ScienceQA_Train \
    --eval_dataset ScienceQA_TEST \
    --tasks None \
    --nsamples 128 \
    --vitnsamples 128 \
    --seed 0 \
    --svd_mode 0.5 \
    --qkv_fuse \
    --rank_ratio 1.5 \
    --svd_lm \
    --label_mode 'qa-qa' \
    --basepath "/data1/vikram/QVLM/" \
    --setting "QSVD/sqa/llavanext_qsvd_16bit_baseline" \
    --rotate \
    --vit_module \
    --grad_info \
    --beta_then_svd
