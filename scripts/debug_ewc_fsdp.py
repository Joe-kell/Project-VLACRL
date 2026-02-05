#!/usr/bin/env python3
"""
Quick debug script to verify EWC data integrity.

This script analyzes ewc_data.pt files to:
1. Check Fisher information for NaN/Inf/suspicious values
2. (For legacy files) Check if old_params have correct shapes by comparing
   them against the saved LoRA checkpoint

Note: New EWC files only contain Fisher information (no old_params).
The old_params reference is captured from loaded checkpoint weights at training start.

Usage:
    python scripts/debug_ewc_fsdp.py \
        --ewc_data ./logs/naive_lora_ewc/task_0_seed1234/ewc_data.pt \
        --checkpoint ./logs/naive_lora_ewc/task_0_seed1234/checkpoints/global_step_10/actor

This runs in seconds without GPU - just loads and compares saved files.
"""

import argparse
import os
import torch


def load_lora_checkpoint(checkpoint_path: str) -> dict:
    """Load LoRA adapter weights from checkpoint."""
    # Try different possible file names
    possible_files = [
        os.path.join(checkpoint_path, "adapter_model.bin"),
        os.path.join(checkpoint_path, "adapter_model.safetensors"),
        os.path.join(checkpoint_path, "pytorch_model.bin"),
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            print(f"Loading checkpoint from: {filepath}")
            if filepath.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(filepath)
            else:
                return torch.load(filepath, map_location="cpu")
    
    raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug EWC FSDP sharding issue")
    parser.add_argument("--ewc_data", required=True, help="Path to ewc_data.pt file")
    parser.add_argument("--checkpoint", default=None, help="Path to LoRA checkpoint directory (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all parameters")
    args = parser.parse_args()

    print("=" * 80)
    print("EWC DATA DEBUG SCRIPT")
    print("=" * 80)

    # Load EWC data
    print(f"\n1. Loading EWC data from: {args.ewc_data}")
    if not os.path.exists(args.ewc_data):
        print(f"   ERROR: EWC data file not found!")
        return 1
    
    ewc_data = torch.load(args.ewc_data, map_location="cpu")
    fisher_dict = ewc_data.get("fisher_dict", {})
    old_params = ewc_data.get("old_params", {})  # May be empty for new format
    
    print(f"   ✓ Loaded successfully")
    print(f"   Fisher dict: {len(fisher_dict)} parameters")
    if old_params:
        print(f"   Old params: {len(old_params)} parameters (legacy format)")
    else:
        print(f"   Old params: None (new format - captured from loaded checkpoint at runtime)")

    # Check for suspicious patterns in old_params (only for legacy format)
    if old_params:
        print(f"\n2. Analyzing old_params shapes and values (legacy format)...")
        
        suspicious_params = []
        total_params = 0
        total_zeros = 0
        total_nans = 0
        
        for name, param in old_params.items():
            total_params += param.numel()
            zeros = (param == 0).sum().item()
            nans = torch.isnan(param).sum().item()
            infs = torch.isinf(param).sum().item()
            total_zeros += zeros
            total_nans += nans
            
            zero_ratio = zeros / param.numel() if param.numel() > 0 else 0
            
            if args.verbose or zero_ratio > 0.5 or nans > 0 or infs > 0:
                print(f"   {name}:")
                print(f"      Shape: {param.shape}")
                print(f"      Dtype: {param.dtype}")
                print(f"      Min: {param.min().item():.6e}, Max: {param.max().item():.6e}")
                print(f"      Mean: {param.mean().item():.6e}, Std: {param.std().item():.6e}")
                print(f"      Zeros: {zeros}/{param.numel()} ({zero_ratio*100:.1f}%)")
                if nans > 0:
                    print(f"      NaNs: {nans}")
                if infs > 0:
                    print(f"      Infs: {infs}")
            
            if zero_ratio > 0.9:
                suspicious_params.append((name, "mostly zeros", zero_ratio))
            if nans > 0:
                suspicious_params.append((name, "contains NaN", nans))
            if infs > 0:
                suspicious_params.append((name, "contains Inf", infs))
        
        overall_zero_ratio = total_zeros / total_params if total_params > 0 else 0
        print(f"\n   Summary:")
        print(f"      Total parameters: {total_params:,}")
        print(f"      Total zeros: {total_zeros:,} ({overall_zero_ratio*100:.1f}%)")
        print(f"      Total NaNs: {total_nans:,}")
        
        if suspicious_params:
            print(f"\n   ⚠ SUSPICIOUS PARAMETERS FOUND:")
            for name, issue, value in suspicious_params[:10]:
                print(f"      - {name}: {issue} ({value})")
            if len(suspicious_params) > 10:
                print(f"      ... and {len(suspicious_params) - 10} more")
        else:
            print(f"\n   ✓ No suspicious patterns in old_params")
    else:
        print(f"\n2. Skipping old_params analysis (new format without old_params)")

    # Check Fisher values
    print(f"\n3. Analyzing Fisher information...")
    
    fisher_stats = []
    for name, fisher in fisher_dict.items():
        mean_val = fisher.mean().item()
        max_val = fisher.max().item()
        min_val = fisher.min().item()
        fisher_stats.append((name, mean_val, max_val, min_val))
    
    if fisher_stats:
        mean_fisher = sum(s[1] for s in fisher_stats) / len(fisher_stats)
        max_fisher = max(s[2] for s in fisher_stats)
        min_fisher = min(s[3] for s in fisher_stats)
        
        print(f"   Mean Fisher (across params): {mean_fisher:.6e}")
        print(f"   Max Fisher value: {max_fisher:.6e}")
        print(f"   Min Fisher value: {min_fisher:.6e}")
        
        if mean_fisher < 1e-10:
            print(f"   ⚠ WARNING: Fisher values are extremely small! EWC will be ineffective.")
        elif mean_fisher > 1e6:
            print(f"   ⚠ WARNING: Fisher values are very large. May cause numerical issues.")
        else:
            print(f"   ✓ Fisher values look reasonable")

    # Compare with checkpoint if provided (only useful for legacy format with old_params)
    if args.checkpoint and old_params:
        print(f"\n4. Comparing old_params with checkpoint: {args.checkpoint}")
        
        try:
            ckpt_params = load_lora_checkpoint(args.checkpoint)
            print(f"   ✓ Checkpoint loaded: {len(ckpt_params)} parameters")
            
            # Compare parameter names
            ewc_names = set(old_params.keys())
            ckpt_names = set(ckpt_params.keys())
            
            def normalize_name(name):
                """
                Normalize parameter names to handle FSDP wrapper differences.
                
                EWC names look like:
                  base_model.model.language_model.model.layers.27._fsdp_wrapped_module.mlp.down_proj.lora_A.default._fsdp_wrapped_module.weight
                
                Checkpoint names look like:
                  base_model.model.language_model.model.layers.13.mlp.gate_proj.lora_B.weight
                
                Need to remove:
                  - _fsdp_wrapped_module (appears in middle of name)
                  - .default (PEFT adapter name, appears before weight)
                """
                import re
                # Remove _fsdp_wrapped_module. (with the dot)
                name = name.replace("._fsdp_wrapped_module.", ".")
                name = name.replace("_fsdp_wrapped_module.", "")
                # Remove .default (PEFT adapter name)
                name = name.replace(".default.", ".")
                # Clean up any double dots
                while ".." in name:
                    name = name.replace("..", ".")
                return name
            
            ewc_normalized = {normalize_name(n): n for n in ewc_names}
            ckpt_normalized = {normalize_name(n): n for n in ckpt_names}
            
            common_normalized = set(ewc_normalized.keys()) & set(ckpt_normalized.keys())
            
            print(f"\n   Name comparison:")
            print(f"      EWC old_params: {len(ewc_names)} params")
            print(f"      Checkpoint: {len(ckpt_names)} params")
            print(f"      Common (normalized): {len(common_normalized)} params")
            
            # Show sample normalized names for debugging
            if args.verbose:
                sample_ewc = list(ewc_names)[:2]
                for name in sample_ewc:
                    print(f"      EWC: {name}")
                    print(f"        -> {normalize_name(name)}")
                sample_ckpt = list(ckpt_names)[:2]
                for name in sample_ckpt:
                    print(f"      Ckpt: {name}")
                    print(f"        -> {normalize_name(name)}")
            
            if len(common_normalized) == 0:
                print(f"\n   ⚠ WARNING: No common parameters found after normalization!")
                print(f"      Sample EWC names (normalized):")
                for name in list(ewc_names)[:3]:
                    print(f"        {name}")
                    print(f"        -> {normalize_name(name)}")
                print(f"      Sample ckpt names (normalized):")
                for name in list(ckpt_names)[:3]:
                    print(f"        {name}")
                    print(f"        -> {normalize_name(name)}")
            else:
                # Compare shapes and values for common parameters
                print(f"\n   Comparing {len(common_normalized)} common parameters...")
                
                shape_mismatches = 0
                value_diffs = []
                large_diffs = []
                
                for norm_name in common_normalized:
                    ewc_name = ewc_normalized[norm_name]
                    ckpt_name = ckpt_normalized[norm_name]
                    
                    ewc_param = old_params[ewc_name]
                    ckpt_param = ckpt_params[ckpt_name]
                    
                    if ewc_param.shape != ckpt_param.shape:
                        shape_mismatches += 1
                        print(f"\n   ⚠ SHAPE MISMATCH: {norm_name}")
                        print(f"      EWC shape: {ewc_param.shape}")
                        print(f"      Ckpt shape: {ckpt_param.shape}")
                        print(f"      THIS CONFIRMS FSDP SHARDING BUG!")
                    else:
                        # Convert to float32 for comparison (handle bfloat16)
                        ewc_f = ewc_param.float()
                        ckpt_f = ckpt_param.float()
                        
                        diff = (ewc_f - ckpt_f).abs()
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        value_diffs.append((norm_name, max_diff, mean_diff, ewc_param.shape))
                        
                        if max_diff > 1e-3:  # Significant difference
                            large_diffs.append((norm_name, max_diff, mean_diff, 
                                               ewc_f.mean().item(), ckpt_f.mean().item()))
                
                # Summary
                print(f"\n   Results:")
                print(f"      Shape mismatches: {shape_mismatches}/{len(common_normalized)}")
                print(f"      Value comparisons: {len(value_diffs)}")
                
                if shape_mismatches > 0:
                    print(f"\n   ✗ FSDP SHARDING BUG CONFIRMED!")
                    print(f"      {shape_mismatches} parameters have shape mismatches")
                    print(f"      old_params contains sharded (partial) values!")
                elif value_diffs:
                    avg_max_diff = sum(d[1] for d in value_diffs) / len(value_diffs)
                    max_max_diff = max(d[1] for d in value_diffs)
                    
                    print(f"      Avg max diff: {avg_max_diff:.6e}")
                    print(f"      Worst max diff: {max_max_diff:.6e}")
                    
                    if large_diffs:
                        print(f"\n   ⚠ {len(large_diffs)} parameters have large value differences (>1e-3):")
                        for name, maxd, meand, ewc_mean, ckpt_mean in large_diffs[:10]:
                            print(f"      {name}:")
                            print(f"         max_diff={maxd:.6e}, ewc_mean={ewc_mean:.6e}, ckpt_mean={ckpt_mean:.6e}")
                        if len(large_diffs) > 10:
                            print(f"      ... and {len(large_diffs) - 10} more")
                        print(f"\n   This indicates old_params DO NOT match the checkpoint!")
                        print(f"   The EWC reference weights are WRONG.")
                    else:
                        print(f"\n   ✓ old_params matches checkpoint!")
                        print(f"      Values are consistent (all diffs < 1e-3)")
                        
        except Exception as e:
            print(f"   ✗ Failed to load checkpoint: {e}")
    elif args.checkpoint and not old_params:
        print(f"\n4. Skipping checkpoint comparison (new format without old_params)")
        print(f"   With the new format, old_params is captured from loaded checkpoint at runtime.")
        print(f"   This avoids the FSDP sharding bug entirely.")
    else:
        print(f"\n4. Skipping checkpoint comparison (--checkpoint not provided)")
        print(f"   Tip: For legacy ewc_data.pt files, provide checkpoint path to detect FSDP sharding issues")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
