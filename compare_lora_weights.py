import torch
from safetensors.torch import load_file
from collections import defaultdict
import numpy as np

run1_path = "/scratch/users/ntu/song0304/loraedit_exp/processed_data/V7_hair/lora/20260228_22-18-28/epoch100/adapter_model.safetensors"
run3_path = "/scratch/users/ntu/song0304/loraedit_exp/processed_data/V7_hair/lora/20260306_01-36-01/epoch100/adapter_model.safetensors"

def analyze_lora(path: str, label: str) -> dict[str, torch.Tensor]:
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*80}")
    
    state_dict = load_file(path)
    
    # Basic counts
    all_keys = list(state_dict.keys())
    lora_a_keys = [k for k in all_keys if "lora_A" in k]
    lora_b_keys = [k for k in all_keys if "lora_B" in k]
    other_keys = [k for k in all_keys if "lora_A" not in k and "lora_B" not in k]
    
    print(f"\nTotal keys: {len(all_keys)}")
    print(f"  lora_A keys: {len(lora_a_keys)}")
    print(f"  lora_B keys: {len(lora_b_keys)}")
    print(f"  other keys:  {len(other_keys)}")
    if other_keys:
        print(f"  other key names: {other_keys[:5]}...")
    
    # Global statistics
    all_params = torch.cat([v.flatten().float() for v in state_dict.values()])
    print(f"\nGlobal weight stats:")
    print(f"  Total params: {all_params.numel():,}")
    print(f"  Mean:  {all_params.mean().item():.6e}")
    print(f"  Std:   {all_params.std().item():.6e}")
    print(f"  Min:   {all_params.min().item():.6e}")
    print(f"  Max:   {all_params.max().item():.6e}")
    print(f"  L2 norm: {all_params.norm().item():.6e}")
    
    # lora_A stats
    a_params = torch.cat([state_dict[k].flatten().float() for k in lora_a_keys])
    print(f"\nlora_A aggregate stats:")
    print(f"  Params: {a_params.numel():,}")
    print(f"  Mean:  {a_params.mean().item():.6e}")
    print(f"  Std:   {a_params.std().item():.6e}")
    print(f"  Min:   {a_params.min().item():.6e}")
    print(f"  Max:   {a_params.max().item():.6e}")
    
    # lora_B stats
    b_params = torch.cat([state_dict[k].flatten().float() for k in lora_b_keys])
    print(f"\nlora_B aggregate stats:")
    print(f"  Params: {b_params.numel():,}")
    print(f"  Mean:  {b_params.mean().item():.6e}")
    print(f"  Std:   {b_params.std().item():.6e}")
    print(f"  Min:   {b_params.min().item():.6e}")
    print(f"  Max:   {b_params.max().item():.6e}")
    
    # Check for all-zero lora_B tensors
    zero_b_keys = []
    for k in lora_b_keys:
        if state_dict[k].abs().max().item() == 0.0:
            zero_b_keys.append(k)
    
    print(f"\nAll-zero lora_B tensors: {len(zero_b_keys)} / {len(lora_b_keys)}")
    if zero_b_keys:
        for k in zero_b_keys[:10]:
            print(f"  ZERO: {k}  shape={list(state_dict[k].shape)}")
        if len(zero_b_keys) > 10:
            print(f"  ... and {len(zero_b_keys) - 10} more")
    
    # Show sample tensors (first 3 lora_A and lora_B)
    print(f"\nSample lora_A tensors:")
    for k in sorted(lora_a_keys)[:3]:
        t = state_dict[k].float()
        print(f"  {k}")
        print(f"    shape={list(t.shape)}  mean={t.mean().item():.6e}  std={t.std().item():.6e}  min={t.min().item():.6e}  max={t.max().item():.6e}")
    
    print(f"\nSample lora_B tensors:")
    for k in sorted(lora_b_keys)[:3]:
        t = state_dict[k].float()
        print(f"  {k}")
        print(f"    shape={list(t.shape)}  mean={t.mean().item():.6e}  std={t.std().item():.6e}  min={t.min().item():.6e}  max={t.max().item():.6e}")
    
    # Magnitude distribution (percentiles)
    abs_all = all_params.abs()
    percentiles = [25, 50, 75, 90, 95, 99, 99.9]
    print(f"\nWeight magnitude percentiles (|w|):")
    for p in percentiles:
        val = torch.quantile(abs_all, p / 100.0).item()
        print(f"  P{p:>5}: {val:.6e}")
    
    return state_dict


# Analyze both
sd1 = analyze_lora(run1_path, "Run1 (20260228_22-18-28) epoch100")
sd3 = analyze_lora(run3_path, "Run3 (20260306_01-36-01) epoch100")

# Direct comparison
print(f"\n{'='*80}")
print(f"  DIRECT COMPARISON: Run1 vs Run3")
print(f"{'='*80}")

# Key overlap
keys1 = set(sd1.keys())
keys3 = set(sd3.keys())
shared = keys1 & keys3
only1 = keys1 - keys3
only3 = keys3 - keys1
print(f"\nKey overlap:")
print(f"  Shared: {len(shared)}")
print(f"  Only in Run1: {len(only1)}")
print(f"  Only in Run3: {len(only3)}")
if only1:
    print(f"    Run1 unique: {list(only1)[:5]}")
if only3:
    print(f"    Run3 unique: {list(only3)[:5]}")

# Per-key difference on shared keys
print(f"\nPer-key comparison on shared keys:")
diffs = []
cosine_sims = []
relative_diffs = []
for k in sorted(shared):
    t1 = sd1[k].float()
    t3 = sd3[k].float()
    diff = (t1 - t3).abs()
    diffs.append(diff.mean().item())
    
    # Cosine similarity
    if t1.norm() > 0 and t3.norm() > 0:
        cos = torch.nn.functional.cosine_similarity(t1.flatten().unsqueeze(0), t3.flatten().unsqueeze(0)).item()
        cosine_sims.append((k, cos))
    
    # Relative diff
    denom = max(t1.abs().mean().item(), t3.abs().mean().item(), 1e-10)
    rel = diff.mean().item() / denom
    relative_diffs.append((k, rel))

# Summary stats of differences
diffs_arr = np.array(diffs)
print(f"  Mean absolute diff across keys: {diffs_arr.mean():.6e}")
print(f"  Max absolute diff across keys:  {diffs_arr.max():.6e}")

# Top 5 most different keys (by relative diff)
relative_diffs.sort(key=lambda x: -x[1])
print(f"\nTop 10 most different keys (by relative diff):")
for k, rel in relative_diffs[:10]:
    t1 = sd1[k].float()
    t3 = sd3[k].float()
    diff = (t1 - t3).abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(t1.flatten().unsqueeze(0), t3.flatten().unsqueeze(0)).item() if t1.norm() > 0 and t3.norm() > 0 else float('nan')
    print(f"  {k}")
    print(f"    rel_diff={rel:.4f}  abs_diff={diff:.6e}  cosine_sim={cos:.4f}")
    print(f"    run1: mean={t1.mean().item():.6e} std={t1.std().item():.6e}")
    print(f"    run3: mean={t3.mean().item():.6e} std={t3.std().item():.6e}")

# Top 5 most similar
print(f"\nTop 5 most similar keys (by relative diff):")
for k, rel in relative_diffs[-5:]:
    t1 = sd1[k].float()
    t3 = sd3[k].float()
    cos = torch.nn.functional.cosine_similarity(t1.flatten().unsqueeze(0), t3.flatten().unsqueeze(0)).item() if t1.norm() > 0 and t3.norm() > 0 else float('nan')
    print(f"  {k}  rel_diff={rel:.6f}  cosine_sim={cos:.4f}")

# Overall cosine similarity of flattened all weights
all1 = torch.cat([sd1[k].flatten().float() for k in sorted(shared)])
all3 = torch.cat([sd3[k].flatten().float() for k in sorted(shared)])
overall_cos = torch.nn.functional.cosine_similarity(all1.unsqueeze(0), all3.unsqueeze(0)).item()
overall_l2 = (all1 - all3).norm().item()
print(f"\nOverall (all shared params concatenated):")
print(f"  Cosine similarity: {overall_cos:.6f}")
print(f"  L2 distance:       {overall_l2:.6e}")
print(f"  Run1 L2 norm:      {all1.norm().item():.6e}")
print(f"  Run3 L2 norm:      {all3.norm().item():.6e}")
print(f"  Norm ratio (run3/run1): {all3.norm().item() / all1.norm().item():.4f}")

