#!/bin/bash
source /data1/vikram/miniconda3/etc/profile.d/conda.sh
conda activate QSVD
cd /data1/vikram/QSVD/QSVD

# Running Magnitude-based global rank allocation (Data-free, Faster)
# Fix applied: increased nsamples to 128, disabled vit_module, set svd_mode to UV
python -u /data1/vikram/QSVD/QSVD/fake_quant/mainllavanext.py \
    --model llava-hf/llava-v1.6-vicuna-7b-hf  \
    --a_bits 4 \
    --w_bits 4 \
    --k_bits 16 \
    --v_bits 16 \
    --cal_dataset ScienceQA_Train \
    --eval_dataset ScienceQA_TEST \
    --seed 0 \
    --svd_mode "UV" \
    --qkv_fuse \
    --rank_ratio 1.5 \
    --svd_lm \
    --magnitude_info \
    --no-grad_info \
    --basepath "/data1/vikram/QVLM/" \
    --setting "QSVD/sqa/magnitude_only_r1.5_fixed_v1" \
    --nsamples 128 \
    --beta_then_svd

echo "PYTHON_EXIT_CODE=$?"
