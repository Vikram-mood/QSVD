import torch
import torch.nn as nn
import utils
import model_utils
import data_utils
import svd_utils
import grad_info_utils
import eval_utilsdistllava
import logging
import os
import pandas as pd
from tqdm import tqdm
import traceback
import numpy as np
import gptq_utils
import quant_utils
import rotation_utils


def run_sweep():
    # Load environment and args
    args = utils.parser_gen()
    utils.set_seed(args.seed)

    # Load model
    logging.info(f"Loading model: {args.model}")
    model, tokenizer, image_processor = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    device = utils.get_dev()
    model.to(device)

    # Step 1: Gradient calibration (computes qkv_svd_info and S_grad_info per layer)
    logging.info("Step 1: Computing gradient information...")
    trainloader = data_utils.get_loaders(
        args.cal_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False,
        args=args
    )

    grad_info_utils.calib_grad_info(
        model=model,
        dataloader=trainloader[0],
        tokenizer=tokenizer,
        image_processor=image_processor,
        args=args,
        use_cache=True
    )

    # Step 2: Backup original (FP16) weights before any quantization or thresholding
    logging.info("Step 2: Backing up original FP16 weights...")
    transformer_layers = model_utils.get_transformer_layers(model, model_utils.get_model_type(model))
    original_weights_backup = {}
    for idx, layer in enumerate(transformer_layers):
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj = getattr(layer.self_attn, name, None)
            if proj is not None and hasattr(proj, 'weight'):
                original_weights_backup[(idx, 'self_attn', name)] = proj.weight.data.clone().cpu()
        if hasattr(layer, 'mlp'):
            for name in ['up_proj', 'gate_proj', 'down_proj']:
                proj = getattr(layer.mlp, name, None)
                if proj is not None and hasattr(proj, 'weight'):
                    original_weights_backup[(idx, 'mlp', name)] = proj.weight.data.clone().cpu()

    # Step 3: Get eval data
    logging.info("Step 3: Loading evaluation dataset...")
    testloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True,
        args=args
    )

    # Step 4: Sweep over percentiles (Threshold -> GPTQ -> Eval)
    percentiles = [0, 10, 15, 20, 25, 30, 40, 50]  # 0 is baseline
    results = []
    original_save_path = args.save_path

    for p in percentiles:
        logging.info(f"--- Evaluating Gradient-Informed Threshold Percentile: {p}% ---")

        # Update save_path per percentile to avoid collisions
        args.save_path = os.path.join(original_save_path, f"p{p}")
        os.makedirs(args.save_path, exist_ok=True)

        # 4.1 Restore original FP16 weights to reset state for this percentile
        for idx, layer in enumerate(transformer_layers):
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(layer.self_attn, name, None)
                key = (idx, 'self_attn', name)
                if proj is not None and key in original_weights_backup:
                    proj.weight.data.copy_(original_weights_backup[key].to(device))
            if hasattr(layer, 'mlp'):
                for name in ['up_proj', 'gate_proj', 'down_proj']:
                    proj = getattr(layer.mlp, name, None)
                    key = (idx, 'mlp', name)
                    if proj is not None and key in original_weights_backup:
                        proj.weight.data.copy_(original_weights_backup[key].to(device))

        # 4.2 Apply gradient-informed thresholding to QKV attention layers BEFORE GPTQ
        if p > 0:
            for idx, layer in enumerate(transformer_layers):
                svd_info = getattr(layer.self_attn, 'qkv_svd_info', None)
                grad_info = getattr(layer.self_attn, 'S_grad_info', None)

                for j, name in enumerate(['q_proj', 'k_proj', 'v_proj']):
                    proj = getattr(layer.self_attn, name, None)
                    if proj is None:
                        continue

                    current_svd_info = None
                    if svd_info is not None:
                        out_features = proj.out_features
                        U_slice = svd_info['U'][j * out_features:(j + 1) * out_features, :]
                        current_svd_info = {
                            'U': U_slice,
                            'V': svd_info['V'],
                            'S': svd_info['S']
                        }

                    svd_utils.apply_weight_threshold(
                        proj, p, grad_info=grad_info, svd_info=current_svd_info
                    )

        # 4.3 Apply Quantization over the thresholded weights (restores performance!)
        if args.w_bits < 16:
            if args.w_rtn:
                logging.info(f"Applying RTN {args.w_bits}-bit weight quantization...")
                gptq_utils.rtn_fwrd(model, device, args)
            else:
                logging.info(f"Applying GPTQ {args.w_bits}-bit weight quantization...")
                gptq_utils.gptq_fwrdllava(model, trainloader, device, args, tokenizer, image_processor)

        if args.a_bits < 16 or args.v_bits < 16:
            logging.info(f"Configuring {args.a_bits}-bit activation and {args.v_bits}-bit V-cache quantization...")
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            down_proj_groupsize = -1
            if args.a_groupsize > 0 and "llama" in args.model:
                down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
            for name in qlayers:
                layer_input_bits = args.a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not (args.a_asym)
                layer_a_clip = min(args.a_clip_ratio, args.lma_clip_ratio)

                if 'v_proj' in name and args.v_bits < 16:
                    qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                                          groupsize=args.v_groupsize,
                                                          sym=not (args.v_asym),
                                                          clip_ratio=args.v_clip_ratio)
                if 'lm_head' in name:
                    layer_input_bits = 16
                if 'down_proj' in name:
                    layer_groupsize = down_proj_groupsize
                qlayers[name].quantizer.configure(bits=layer_input_bits,
                                                  groupsize=layer_groupsize,
                                                  sym=layer_a_sym,
                                                  clip_ratio=layer_a_clip)

        if args.k_bits < 16:
            logging.info(f"Configuring {args.k_bits}-bit K-cache quantization...")
            rope_function_name = model_utils.get_rope_function_name(model)
            k_quant_config = {'k_bits': args.k_bits, "k_groupsize": args.k_groupsize,
                              "k_sym": not (args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in transformer_layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config.text_config,
                    **k_quant_config)

        # 4.4 Calculate global sparsity
        sparsity = svd_utils.calculate_model_sparsity(model)
        logging.info(f"Global Model Sparsity: {sparsity:.2f}%")

        # 4.5 Evaluate
        logging.info(f"Evaluating accuracy on {args.eval_dataset}...")
        try:
            acc_results = eval_utilsdistllava.evaluator(
                model, testloader, device, args, tokenizer, image_processor
            )
            if isinstance(acc_results, pd.DataFrame):
                if 'Overall' in acc_results.index:
                    acc = acc_results.loc['Overall'].values[0]
                else:
                    acc = acc_results.iloc[0, 0]
            elif isinstance(acc_results, dict):
                acc = acc_results.get('Overall', 0.0)
            else:
                acc = float(acc_results) if acc_results is not None else 0.0
        except Exception as e:
            logging.error(f"Evaluation failed at {p}%: {e}")
            logging.error(traceback.format_exc())
            acc = 0.0

        logging.info(f"Accuracy at {p}%: {acc}")

        results.append({
            'Percentile': p,
            'Sparsity': f"{sparsity:.2f}%",
            'Accuracy': acc
        })

        # Save incrementally
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(original_save_path, 'threshold_sweep_results.csv'), index=False)

    logging.info("Sweep completed. Final Results:")
    print(df.to_markdown())


if __name__ == "__main__":
    run_sweep()
