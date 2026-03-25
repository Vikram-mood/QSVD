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
    
    # Calibration for gradients (Step 1)
    logging.info("Step 1: Computing gradient information...")
    trainloader = data_utils.get_loaders(
        args.cal_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False,
        args=args
    )
    dataloader, _ = trainloader
    
    # We need qkv_svd_info and S_grad_info for analysis
    grad_info_utils.calib_grad_info(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        image_processor=image_processor,
        args=args,
        use_cache=True
    )
    
    # Define percentiles to test
    percentiles = [0, 50, 70, 80, 90, 95, 99] # 0 is baseline
    results = []
    
    # Get test loader
    logging.info("Loading test loader for evaluation...")
    testloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True,
        args=args
    )

    # Backup the original weights of attention layers so we can reset for each threshold
    logging.info("Backing up attention layer weights...")
    attention_weights_backup = {}
    layers = model_utils.get_transformer_layers(model, model_utils.get_model_type(model))
    for idx, layer in enumerate(layers):
        for name in ['q_proj', 'k_proj', 'v_proj']:
            proj = getattr(layer.self_attn, name)
            attention_weights_backup[(idx, name)] = proj.weight.data.clone().cpu()

    original_save_path = args.save_path
    for p in percentiles:
        logging.info(f"--- Evaluating Threshold Percentile: {p}% ---")
        
        # Update save_path to avoid result collision between runs
        args.save_path = os.path.join(original_save_path, f"p{p}")
        os.makedirs(args.save_path, exist_ok=True)
        
        # Reset and Apply thresholding to attention layers
        for idx, layer in enumerate(layers):
            for name in ['q_proj', 'k_proj', 'v_proj']:
                proj = getattr(layer.self_attn, name)
                # Restore original weight
                proj.weight.data.copy_(attention_weights_backup[(idx, name)].to(device))
                if p > 0:
                    svd_utils.apply_weight_threshold(proj, p)
        
        # Calculate sparsity
        sparsity = svd_utils.calculate_model_sparsity(model)
        logging.info(f"Global Model Sparsity: {sparsity:.2f}%")
        
        # Evaluate Accuracy
        logging.info(f"Evaluating accuracy on {args.eval_dataset}...")
        try:
            acc_results = eval_utilsdistllava.evaluator(model, testloader, device, args, tokenizer, image_processor)
            # Extract accuracy from results
            if isinstance(acc_results, pd.DataFrame):
                # Look for 'Overall' or similar
                if 'Overall' in acc_results.index:
                    acc = acc_results.loc['Overall'].values[0]
                else:
                    acc = acc_results.iloc[0, 0] # Fallback
            elif isinstance(acc_results, dict):
                acc = acc_results.get('Overall', 0.0)
            else:
                acc = float(acc_results) if acc_results is not None else 0.0
        except Exception as e:
            import traceback
            logging.error(f"Evaluation failed at {p}%: {e}")
            logging.error(traceback.format_exc())
            acc = 0.0
            
        logging.info(f"Accuracy at {p}%: {acc}")
        
        results.append({
            'ThresholdP': p,
            'Sparsity': sparsity,
            'Accuracy': acc
        })
        
        # Save results to the original base path
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(original_save_path, 'threshold_sweep_results.csv'), index=False)

    logging.info("Sweep completed. Final Results:")
    print(df.to_markdown())

if __name__ == "__main__":
    run_sweep()
