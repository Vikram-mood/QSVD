#!/bin/bash
source /data1/vikram/miniconda3/etc/profile.d/conda.sh
conda activate QSVD
cd /data1/vikram/QSVD/QSVD

# Standard SVD Experiment (Magnitude-based)
# - No --grad_info or --fisher_info
# - No --act_aware (to isolate pure weight importance)
# - 16-bit to avoid quantization noise for now

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u /data1/vikram/QSVD/QSVD/fake_quant/mainllavanext.py \
    --model llava-hf/llava-v1.6-vicuna-7b-hf  \
    --a_bits 16 \
    --w_bits 16 \
    --k_bits 16 \
    --v_bits 16 \
    --cal_dataset ScienceQA_Train \
    --eval_dataset ScienceQA_TEST \
    --tasks None \
    --nsamples 256 \
    --vitnsamples 256 \
    --seed 0 \
    --svd_mode 0.5 \
    --qkv_fuse \
    --rank_ratio 1.5 \
    --svd_lm \
    --label_mode 'qa-qa' \
    --basepath "/data1/vikram/QVLM/" \
    --setting "QSVD/sqa/llavanext_magnitude_baseline" \
    --rotate \
    --vit_module
