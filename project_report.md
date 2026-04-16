# Post-Training Compression of Vision-Language Models via Efficient Singular Value Importance Approximation

**Authors:** Vikram et al.  
**Date:** April 16, 2026  

---

## 1. Introduction
Large-scale Vision-Language Models (VLMs), such as LLaVA, have demonstrated remarkable capabilities in multi-modal reasoning and understanding. However, their sheer parameter count poses significant challenges for deployment, particularly concerning memory footprint and computational latency. Model compression is essential to enable the execution of these models on edge devices and consumer-grade hardware. This report explores structured compression via Quantized Singular Value Decomposition (QSVD) as a means to reduce model size while preserving the sophisticated reasoning abilities required for benchmarks like ScienceQA.

## 2. Motivation
The high memory usage of VLMs is a primary bottleneck for real-time applications. Traditional post-training quantization (PTQ) methods often struggle to maintain reasoning performance at very low bit-widths (e.g., W4A4). Low-rank approximation via SVD offers a complementary path to compression, but determining which singular values are "important" remains a core challenge. Preserving the model's ability to correlate visual features with linguistic context is critical, yet computationally expensive importance metrics can disqualify the benefits of compression by imposing prohibitive overhead during the optimization phase.

## 3. Existing Work / Literature Review
Structured compression often relies on low-rank approximations. The **QSVD paper approach** [1] proposes an importance score based on Fisher Information. This method computes the sensitivity of the model's loss to changes in singular values, typically involving the calculation of:
$$I_i = \mathbb{E} [ (\nabla_{\sigma_i} \mathcal{L})^2 ]$$
where $\sigma_i$ are the singular values. This requires either the explicit computation of $\Delta W$ or a memory-intensive gradient pass over a calibration dataset.

Other relevant works include:
- **Post-Training Quantization (PTQ):** Methods like *bitsandbytes* and W4A4 quantization (RTN/GPTQ) which target weight precision.
- **LoRA (Low-Rank Adaptation):** Using low-rank matrices for efficient fine-tuning, which shares the mathematical foundation of low-rank structure with our compression approach.

## 4. Problem Statement
The primary objective is to reduce the memory and computational burden of the importance scoring phase in QSVD without sacrificing the quality of the compressed model. Specifically, we aim to:
1. Avoid the OOM (Out of Memory) issues associated with computing $\Delta W$ or large-scale gradients on limited-resource servers.
2. Minimize the time complexity of the compression pipeline.
3. Maintain high accuracy on reasoning-intensive benchmarks like ScienceQA.

## 5. Methodology
### 5.1. SVD Decomposition
Given a weight matrix $W \in \mathbb{R}^{H \times W}$, we perform Singular Value Decomposition:
$$W = U \Sigma V^T$$
where $\Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_k)$ contains the singular values in descending order.

### 5.2. QSVD (Fisher-based Importance)
The QSVD paper approach scales the singular values by their importance scores derived from Fisher Information. This identifies dimensions that are most critical to the model's objective function rather than just the weight magnitude.

### 5.3. Magnitude-based Weight Thresholding
As a baseline for unstructured compression, we implement element-wise weight thresholding:
$$W_{i,j} = \begin{cases} W_{i,j} & \text{if } |W_{i,j}| \geq \tau \\ 0 & \text{otherwise} \end{cases}$$
The threshold $\tau$ is determined per layer (Attention Q, K, V) based on a global percentile to achieve the desired model-wide sparsity. This method serves as a comparison against structured low-rank methods.

### 5.4. Proposed $\sigma_i^2$-based Importance Approximation
... (rest of the section) ...

## 6. Experimental Setup
... (rest of the section) ...

## 7. Results & Analysis
The following table summarizes the performance across the thresholding sweep and compares it with our proposed importance-aware SVD method.

### 7.1. Magnitude-based Thresholding Sweep (W4A4)
| Threshold Percentile | Sparsity (Attention) | Accuracy (ScienceQA) |
| :--- | :--- | :--- |
| **0% (Baseline)** | 0.00% | 61.43% |
| **10%** | **9.97%** | **62.27%** |
| **15%** | 14.96% | 60.44% |
| **20%** | 19.94% | 59.49% |
| **25%** | 24.94% | 58.30% |
| **30%** | 29.91% | 54.83% |
| **40%** | 39.91% | 53.89% |
| **50%** | 49.90% | 37.18% |

### 7.2. Comparison with SVD-based Methods
| Approach | Compression Mode | Accuracy (ScienceQA) | Memory Benefit |
| :--- | :--- | :--- | :--- |
| **W4A4 Baseline** | Quantization only | 61.4% | ~4.2 GB |
| **Weight Thresholding (10%)** | Unstructured | **62.3%** | ~4.1 GB |
| **QSVD (Fisher W4A4)** | Low-rank (ratio 1.5) | 59.2% | ~3.8 GB |
| **QSVD ($\sigma_i^2$ Approx)** | Low-rank (ratio 1.5) | 58.7% | **~3.8 GB** |

### 7.3. Analysis of Findings
1. **The Regularization Effect:** A surprising result is the accuracy boost at 10% sparsity. Zeroing out the smallest 10% of weights (by magnitude) in the attention layers improved accuracy from 61.43% to 62.27%. This suggests that magnitude-based thresholding acts as a form of "drop-path" or noise reduction that generalizes better in the presence of 4-bit quantization noise.
2. **Resilience to Sparsity:** LLaVA-Next shows remarkable resilience up to ~25% sparsity. Beyond this point, the accuracy begins to drop significantly, with a precipitous collapse observed at 50% sparsity (37.2%).
3. **Thresholding vs. SVD:** While thresholding is effective at low sparsity, SVD-based compression (Section 7.2) is expected to provide better scaling at higher compression ratios. However, at the current bit-widths (W4A4), the overhead of SVD rank reduction (e.g., in `run_me.sh`) results in a lower baseline (59.25%) compared to the pure 4-bit baseline.

## 8. Challenges Faced
1. **Out of Memory (OOM):** Fixed by improving GPU memory management during layer-wise rotation.
2. **Discrepancy in Baselines:** Identified that `run_me.sh` results were lower due to active SVD compression flags which weren't present in the pure W4A4 sweep.

## 9. Conclusion
Magnitude-based thresholding is a highly effective, zero-cost method for initial compression of VLMs in quantized environments. Specifically, a 10% thresholding percentile optimizes the accuracy for LLaVA-Next. Future work should integrate this unstructured pruning with structured SVD to achieve even higher compression ratios.
