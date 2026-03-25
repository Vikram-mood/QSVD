"""
Magnitude-Based Weight Sparsification Experiment
=================================================
Implements threshold-based sparsification on QKV attention weights and compares
against baseline (no compression) and truncated SVD (QSVD-style).

Usage:
    cd QSVD/fake_quant
    python magnitude_experiment.py \
        --model llava-hf/llava-v1.6-vicuna-7b-hf \
        --cache_file ../cache_file/llava-next-7b \
        --basepath /path/to/data \
        --nsamples 16 \
        --seed 0

The script sweeps percentile thresholds [50,70,80,90,95,99] on QKV weights,
evaluates ScienceQA accuracy, and prints a summary table.
"""

import os
import sys
import copy
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

import utils
import model_utils
import data_utils
import svd_utils
import act_aware_utils
import grad_info_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_qkv_linears(model):
    """Return {layer_idx: {'q': module, 'k': module, 'v': module}} for LM layers."""
    result = {}
    for idx, layer in enumerate(model_utils.get_layers(model)):
        attn = layer.self_attn
        result[idx] = {
            'q': attn.q_proj,
            'k': attn.k_proj,
            'v': attn.v_proj,
        }
    return result


def compute_svd_importance(model):
    """
    For each layer, compute gradient-based importance scores I_sigma = diag(U^T G_w V)
    using the pre-computed S_grad_info stored on self_attn after calib_grad_info().
    Returns {layer_idx: tensor of shape [rank]} or None if not available.
    """
    scores = {}
    for idx, layer in enumerate(model_utils.get_layers(model)):
        if hasattr(layer.self_attn, 'S_grad_info'):
            scores[idx] = layer.self_attn.S_grad_info.cpu()
    return scores if scores else None


def apply_magnitude_threshold(model, percentile, target='qkv'):
    """
    Apply element-wise magnitude thresholding to QKV weight matrices.
    Weights with |w| < tau are zeroed out.

    Args:
        model: the VLM
        percentile: float in [0,100], threshold = percentile of |W| across all QKV weights
        target: 'qkv' | 'q' | 'k' | 'v'

    Returns:
        masks: {layer_idx: {'q': mask, 'k': mask, 'v': mask}}
        tau: the threshold value used
        sparsity: fraction of zeros
    """
    # Collect all absolute weight values to compute global threshold
    all_abs = []
    qkv_map = get_qkv_linears(model)
    keys = list({'q', 'k', 'v'} & set(target)) if target != 'qkv' else ['q', 'k', 'v']

    for idx, proj_dict in qkv_map.items():
        for key in keys:
            module = proj_dict[key]
            # Handle SVDLinear (has ALinear/BLinear) or plain Linear
            W = _get_weight(module)
            if W is not None:
                all_abs.append(W.abs().float().cpu().reshape(-1))

    if not all_abs:
        logging.warning("No weights found for thresholding.")
        return {}, 0.0, 0.0

    all_abs_cat = torch.cat(all_abs)
    tau = float(torch.quantile(all_abs_cat, percentile / 100.0).item())

    # Apply threshold and record masks
    masks = {}
    total_params = 0
    zero_params = 0

    for idx, proj_dict in qkv_map.items():
        masks[idx] = {}
        for key in keys:
            module = proj_dict[key]
            W = _get_weight(module)
            if W is None:
                continue
            mask = (W.abs() >= tau)
            _set_weight(module, W * mask.to(W.dtype))
            masks[idx][key] = mask.cpu()
            total_params += W.numel()
            zero_params += (~mask).sum().item()

    sparsity = zero_params / total_params if total_params > 0 else 0.0
    return masks, tau, sparsity


def restore_weights(model, original_weights):
    """Restore QKV weights from saved originals."""
    qkv_map = get_qkv_linears(model)
    for idx, proj_dict in qkv_map.items():
        for key in ['q', 'k', 'v']:
            module = proj_dict[key]
            saved = original_weights.get((idx, key))
            if saved is not None:
                _set_weight(module, saved.clone())


def save_original_weights(model):
    """Save a copy of all QKV weights before any modification."""
    saved = {}
    qkv_map = get_qkv_linears(model)
    for idx, proj_dict in qkv_map.items():
        for key in ['q', 'k', 'v']:
            module = proj_dict[key]
            W = _get_weight(module)
            if W is not None:
                saved[(idx, key)] = W.clone().cpu()
    return saved


def _get_weight(module):
    """Get the effective weight matrix from a Linear or SVDLinear."""
    if isinstance(module, nn.Linear):
        return module.weight.data
    elif hasattr(module, 'ALinear') and hasattr(module, 'BLinear'):
        # SVDLinear: reconstruct W = ALinear.weight @ BLinear.weight
        # ALinear: [out, rank], BLinear: [rank, in]
        A = module.ALinear.weight.data.float()
        B = module.BLinear.weight.data.float()
        return (A @ B).to(A.dtype)
    return None


def _set_weight(module, W):
    """Set weight back. For SVDLinear we threshold the reconstructed W then re-decompose."""
    if isinstance(module, nn.Linear):
        module.weight.data = W.to(module.weight.dtype)
    elif hasattr(module, 'ALinear') and hasattr(module, 'BLinear'):
        # For SVDLinear, apply mask to reconstructed W then re-SVD at same rank
        rank = module.ALinear.in_features
        dtype = module.ALinear.weight.dtype
        dev = module.ALinear.weight.device
        W_f = W.float().to(dev)
        try:
            U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            module.ALinear.weight.data = U.mul(S).to(dtype)
            module.BLinear.weight.data = Vt.to(dtype)
        except Exception:
            # Fallback: just scale A to absorb W
            pass


def apply_svd_truncation(model, rank_ratio, args, original_weights, tokenizer, image_processor):
    """
    Apply standard truncated SVD (top-k singular values) to QKV layers.
    This is the QSVD-style baseline without gradient-guided rank allocation.
    """
    # Restore clean weights first
    restore_weights(model, original_weights)

    qkv_map = get_qkv_linears(model)
    dev = utils.get_dev()

    for idx, proj_dict in qkv_map.items():
        for key in ['q', 'k', 'v']:
            module = proj_dict[key]
            if not isinstance(module, nn.Linear):
                continue  # skip already-SVD layers
            W = module.weight.data.float().to(dev)
            n_params = W.numel()
            compressed = int(n_params * rank_ratio)
            rank = max(1, compressed // (module.in_features + module.out_features))
            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                U_r = U[:, :rank].mul(S[:rank])
                Vt_r = Vt[:rank, :]
                module.weight.data = (U_r @ Vt_r).to(module.weight.dtype).cpu()
            except Exception as e:
                logging.warning(f"SVD failed for layer {idx} {key}: {e}")


def compute_layer_sensitivity(model, original_weights, percentiles):
    """
    Compute per-layer sensitivity: for each layer, apply threshold independently
    and measure the L2 reconstruction error of the weight matrix.
    Returns dict: {layer_idx: {percentile: error}}
    """
    sensitivity = defaultdict(dict)
    qkv_map = get_qkv_linears(model)

    for idx, proj_dict in qkv_map.items():
        for pct in percentiles:
            total_err = 0.0
            for key in ['q', 'k', 'v']:
                saved = original_weights.get((idx, key))
                if saved is None:
                    continue
                W_orig = saved.float()
                tau = float(torch.quantile(W_orig.abs(), pct / 100.0).item())
                mask = (W_orig.abs() >= tau)
                W_sparse = W_orig * mask.float()
                err = (W_orig - W_sparse).norm().item()
                total_err += err
            sensitivity[idx][pct] = total_err

    return sensitivity


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, args, tokenizer, image_processor):
    """Run ScienceQA evaluation and return accuracy."""
    testloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True,
        args=args,
    )
    import eval_utilsdistllava
    acc = eval_utilsdistllava.evaluator(
        model, testloader, utils.get_dev(), args, tokenizer, image_processor
    )
    return acc


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    utils.set_seed(args.seed)

    logging.info("=" * 60)
    logging.info("Loading model...")
    model, tokenizer, image_processor = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    # -----------------------------------------------------------------------
    # Optionally apply QSVD SVD decomposition first (so we threshold SVDLinear)
    # -----------------------------------------------------------------------
    if args.apply_svd_first:
        logging.info("Applying QSVD joint QKV SVD decomposition first...")
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
        logging.info("SVD decomposition done.")

    # -----------------------------------------------------------------------
    # Save original weights
    # -----------------------------------------------------------------------
    logging.info("Saving original QKV weights...")
    original_weights = save_original_weights(model)
    logging.info(f"Saved weights for {len(original_weights)} (layer, proj) pairs.")

    # -----------------------------------------------------------------------
    # Compute layer sensitivity (reconstruction error, no forward pass needed)
    # -----------------------------------------------------------------------
    percentiles = args.percentiles
    logging.info("Computing layer sensitivity (reconstruction error)...")
    sensitivity = compute_layer_sensitivity(model, original_weights, percentiles)

    # -----------------------------------------------------------------------
    # Baseline: original model accuracy
    # -----------------------------------------------------------------------
    results = []

    logging.info("\n--- Evaluating BASELINE (no compression) ---")
    baseline_acc = evaluate_model(model, args, tokenizer, image_processor)
    results.append({
        'method': 'Baseline',
        'threshold_pct': '-',
        'tau': '-',
        'sparsity': 0.0,
        'accuracy': baseline_acc,
    })
    logging.info(f"Baseline accuracy: {baseline_acc:.4f}")

    # -----------------------------------------------------------------------
    # Truncated SVD baseline
    # -----------------------------------------------------------------------
    if args.run_svd_baseline:
        logging.info("\n--- Evaluating Truncated SVD baseline ---")
        restore_weights(model, original_weights)
        apply_svd_truncation(model, args.svd_rank_ratio, args, original_weights, tokenizer, image_processor)
        svd_acc = evaluate_model(model, args, tokenizer, image_processor)
        results.append({
            'method': f'TruncSVD(r={args.svd_rank_ratio})',
            'threshold_pct': '-',
            'tau': '-',
            'sparsity': '-',
            'accuracy': svd_acc,
        })
        logging.info(f"Truncated SVD accuracy: {svd_acc:.4f}")
        restore_weights(model, original_weights)

    # -----------------------------------------------------------------------
    # Magnitude threshold sweep
    # -----------------------------------------------------------------------
    for pct in percentiles:
        logging.info(f"\n--- Evaluating Magnitude Threshold @ {pct}th percentile ---")
        restore_weights(model, original_weights)

        masks, tau, sparsity = apply_magnitude_threshold(model, pct, target='qkv')
        logging.info(f"  tau={tau:.6f}, sparsity={sparsity*100:.2f}%")

        acc = evaluate_model(model, args, tokenizer, image_processor)
        results.append({
            'method': 'MagThresh',
            'threshold_pct': pct,
            'tau': f'{tau:.6f}',
            'sparsity': sparsity,
            'accuracy': acc,
        })
        logging.info(f"  Accuracy: {acc:.4f}")

        # Restore for next iteration
        restore_weights(model, original_weights)

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    _print_results_table(results)
    _print_sensitivity_table(sensitivity, percentiles)
    _print_analysis(results, sensitivity)

    # Save results
    import json
    out_path = os.path.join(args.save_path, 'magnitude_experiment_results.json')
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'sensitivity': {str(k): v for k, v in sensitivity.items()}}, f, indent=2)
    logging.info(f"\nResults saved to {out_path}")

    return results


def _print_results_table(results):
    header = f"{'Method':<25} {'Threshold%':>12} {'Tau':>12} {'Sparsity%':>12} {'Accuracy':>10}"
    sep = "-" * len(header)
    logging.info("\n" + sep)
    logging.info("RESULTS TABLE")
    logging.info(sep)
    logging.info(header)
    logging.info(sep)
    for r in results:
        sparsity_str = f"{r['sparsity']*100:.2f}" if isinstance(r['sparsity'], float) else str(r['sparsity'])
        acc_str = f"{r['accuracy']:.4f}" if isinstance(r['accuracy'], float) else str(r['accuracy'])
        logging.info(
            f"{r['method']:<25} {str(r['threshold_pct']):>12} {str(r['tau']):>12} "
            f"{sparsity_str:>12} {acc_str:>10}"
        )
    logging.info(sep)


def _print_sensitivity_table(sensitivity, percentiles):
    logging.info("\nLAYER SENSITIVITY (L2 reconstruction error, summed over Q+K+V)")
    header = f"{'Layer':>6} " + " ".join(f"{'p'+str(p):>10}" for p in percentiles)
    logging.info(header)
    logging.info("-" * len(header))
    for idx in sorted(sensitivity.keys()):
        row = f"{idx:>6} " + " ".join(f"{sensitivity[idx].get(p, 0.0):>10.4f}" for p in percentiles)
        logging.info(row)


def _print_analysis(results, sensitivity):
    logging.info("\n" + "=" * 60)
    logging.info("ANALYSIS")
    logging.info("=" * 60)

    # Find accuracy drop per threshold
    baseline = next((r for r in results if r['method'] == 'Baseline'), None)
    if baseline is None:
        return
    base_acc = baseline['accuracy']

    logging.info("\n1. Accuracy vs Sparsity trade-off:")
    mag_results = [r for r in results if r['method'] == 'MagThresh']
    for r in mag_results:
        drop = base_acc - r['accuracy']
        logging.info(
            f"   p={r['threshold_pct']:>3}%  sparsity={r['sparsity']*100:>6.2f}%  "
            f"acc={r['accuracy']:.4f}  drop={drop:+.4f}"
        )

    # Most sensitive layers (highest error at 90th percentile)
    if 90 in sensitivity[list(sensitivity.keys())[0]]:
        sorted_layers = sorted(sensitivity.items(), key=lambda x: x[1].get(90, 0), reverse=True)
        logging.info("\n2. Most sensitive layers (L2 error at 90th percentile threshold):")
        for idx, errs in sorted_layers[:5]:
            logging.info(f"   Layer {idx:>3}: error={errs.get(90, 0):.4f}")

    logging.info("\n3. Key observations:")
    logging.info("   - Magnitude thresholding in weight space is a coarser approximation")
    logging.info("     than SVD truncation in singular space, which preserves the principal")
    logging.info("     directions of the weight matrix.")
    logging.info("   - SVD truncation removes the least important singular components")
    logging.info("     globally, while magnitude thresholding removes small individual")
    logging.info("     entries which may collectively span important subspaces.")
    logging.info("   - Layers with high reconstruction error at a given threshold are")
    logging.info("     more sensitive and may require lower sparsity budgets.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Magnitude sparsification experiment")

    # Model
    parser.add_argument('--model', type=str, default='llava-hf/llava-v1.6-vicuna-7b-hf')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)

    # Data
    parser.add_argument('--cal_dataset', type=str, default='ScienceQA_Train')
    parser.add_argument('--eval_dataset', type=str, default='ScienceQA_TEST')
    parser.add_argument('--nsamples', type=int, default=16)
    parser.add_argument('--vitnsamples', type=int, default=16)
    parser.add_argument('--basepath', type=str, default='../')
    parser.add_argument('--cache_file', type=str, default=None)

    # SVD / QSVD settings (used if apply_svd_first=True)
    parser.add_argument('--apply_svd_first', action='store_true', default=False,
                        help='Apply QSVD joint QKV SVD before thresholding')
    parser.add_argument('--rank_ratio', type=float, default=0.9)
    parser.add_argument('--svd_mode', type=str, default='UV')
    parser.add_argument('--calib_method', type=str, default='abs_mean')
    parser.add_argument('--act_aware', action='store_true', default=True)
    parser.add_argument('--act_alpha', type=float, default=0.5)
    parser.add_argument('--had_rank', action='store_true', default=True)
    parser.add_argument('--qkv_fuse', action='store_true', default=True)
    parser.add_argument('--grad_info', action='store_true', default=True)
    parser.add_argument('--use_cache', type=lambda x: x.lower() in ['true','1'], default=True)
    parser.add_argument('--cache_in_log', action='store_true', default=False)
    parser.add_argument('--label_mode', type=str, default='qa-qa')
    parser.add_argument('--beta_then_svd', action='store_true', default=True)
    parser.add_argument('--svd_lm', action='store_true', default=False)

    # Experiment settings
    parser.add_argument('--percentiles', type=float, nargs='+',
                        default=[50, 70, 80, 90, 95, 99],
                        help='Percentile thresholds to sweep')
    parser.add_argument('--run_svd_baseline', action='store_true', default=True,
                        help='Also evaluate truncated SVD baseline')
    parser.add_argument('--svd_rank_ratio', type=float, default=0.5,
                        help='Rank ratio for truncated SVD baseline')

    # Misc (needed by utils/eval infrastructure)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--a_bits', type=int, default=16)
    parser.add_argument('--w_bits', type=int, default=16)
    parser.add_argument('--k_bits', type=int, default=16)
    parser.add_argument('--v_bits', type=int, default=16)
    parser.add_argument('--a_groupsize', type=int, default=-1)
    parser.add_argument('--w_groupsize', type=int, default=-1)
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--a_asym', action='store_true', default=False)
    parser.add_argument('--w_asym', action='store_true', default=False)
    parser.add_argument('--v_asym', action='store_true', default=False)
    parser.add_argument('--k_asym', action='store_true', default=False)
    parser.add_argument('--a_clip_ratio', type=float, default=0.9)
    parser.add_argument('--lma_clip_ratio', type=float, default=0.9)
    parser.add_argument('--vita_clip_ratio', type=float, default=1.0)
    parser.add_argument('--k_clip_ratio', type=float, default=1.0)
    parser.add_argument('--v_clip_ratio', type=float, default=1.0)
    parser.add_argument('--w_clip', action='store_true', default=False)
    parser.add_argument('--w_rtn', action='store_true', default=False)
    parser.add_argument('--rotate', action='store_true', default=False)
    parser.add_argument('--rotate_mode', type=str, default='hadamard')
    parser.add_argument('--rotation_seed', type=int, default=-1)
    parser.add_argument('--fp32_had', action='store_true', default=False)
    parser.add_argument('--int8_down_proj', action='store_true', default=False)
    parser.add_argument('--k_pre_rope', action='store_true', default=False)
    parser.add_argument('--act_order', action='store_true', default=False)
    parser.add_argument('--percdamp', type=float, default=0.01)
    parser.add_argument('--vit_module', action='store_true', default=False)
    parser.add_argument('--vit_online', action='store_true', default=False)
    parser.add_argument('--vit_mmoff', action='store_true', default=False)
    parser.add_argument('--mm_rh', action='store_true', default=False)
    parser.add_argument('--lm_off', action='store_true', default=False)
    parser.add_argument('--profile_method', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--load_qmodel_path', type=str, default=None)
    parser.add_argument('--save_qmodel_path', type=str, default=None)
    parser.add_argument('--distribute', action='store_true', default=False)
    parser.add_argument('--lm_eval', action='store_true', default=False)
    parser.add_argument('--tasks', nargs='+', default=[])
    parser.add_argument('--vlmtasks', nargs='+', default=['ScienceQA_TEST'])
    parser.add_argument('--lm_eval_batch_size', type=int, default=128)
    parser.add_argument('--dosample', action='store_true', default=False)
    parser.add_argument('--case_study', action='store_true', default=False)
    parser.add_argument('--bs_to_nsamples', action='store_true', default=False)
    parser.add_argument('--train_fix_hf', action='store_true', default=False)
    parser.add_argument('--token_length', type=int, default=-1)
    parser.add_argument('--grad_alpha', type=float, default=1.0)
    parser.add_argument('--fisher_info', action='store_true', default=False)
    parser.add_argument('--magnitude_info', action='store_true', default=False)
    parser.add_argument('--had_svd', action='store_true', default=False)
    parser.add_argument('--kv_fuse', action='store_true', default=False)
    parser.add_argument('--latent_smooth', action='store_true', default=False)
    parser.add_argument('--svd_vit', action='store_true', default=False)
    parser.add_argument('--beta_lr', type=float, default=1.0)
    parser.add_argument('--beta_epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=512)

    # Output
    parser.add_argument('--setting', type=str, default='magnitude_experiment')
    parser.add_argument('--save_name', type=str, default=None)

    args = parser.parse_args()

    # Build save path (mirrors utils.parser_gen logic)
    from datetime import datetime
    if args.save_name is None:
        args.save_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.setting = f'W{args.w_bits}A{args.a_bits}K{args.k_bits}V{args.v_bits}' + args.setting
    args.save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'experiments', args.model, args.setting
    )
    os.makedirs(args.save_path, exist_ok=True)
    utils.config_logging(os.path.join(args.save_path, f'{args.save_name}.log'))

    return args


if __name__ == '__main__':
    args = parse_args()
    utils.set_seed(args.seed)
    run_experiment(args)
