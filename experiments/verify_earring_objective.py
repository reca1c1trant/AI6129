"""Objective pixel-level comparison across all earring experiments.

Compares: Hair (baseline working), Earring original, Earring expanded mask.
For each: measures mask-region diff between original and edited inference outputs.
"""
import numpy as np
import imageio.v3 as iio
import os
from pathlib import Path


def load_video(path: str) -> np.ndarray:
    """Load video as (T, H, W, C) uint8 array."""
    frames = iio.imread(path, plugin="pyav")
    return frames


def load_mask(mask_dir: str, num_frames: int) -> np.ndarray:
    """Load mask frames as (T, H, W) bool array."""
    exts = ('.png', '.jpg', '.jpeg')
    files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(exts) and f[0].isdigit()])
    masks = []
    for f in files[:num_frames]:
        m = iio.imread(os.path.join(mask_dir, f))
        if m.ndim == 3:
            m = m[..., 0]
        masks.append(m > 127)
    return np.stack(masks)


def analyze_condition(name: str, original_path: str, edited_path: str, mask_dir: str) -> dict:
    """Analyze one experimental condition."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    original = load_video(original_path)
    edited = load_video(edited_path)
    num_frames = min(len(original), len(edited))
    masks = load_mask(mask_dir, num_frames)

    print(f"  Frames: {num_frames}, Shape: {original.shape[1:]}")
    print(f"  Mask coverage: {masks.mean()*100:.1f}%")

    # Pixel diff between original and edited
    diff = np.abs(original[:num_frames].astype(float) - edited[:num_frames].astype(float))
    mean_diff = diff.mean(axis=-1)  # (T, H, W)

    results = {
        "name": name,
        "num_frames": num_frames,
        "mask_coverage": masks.mean() * 100,
    }

    # Per-frame analysis
    fg_diffs = []
    bg_diffs = []
    for t in range(num_frames):
        fg_mask = masks[t]
        bg_mask = ~fg_mask
        if fg_mask.sum() > 0:
            fg_diff = mean_diff[t][fg_mask].mean()
        else:
            fg_diff = 0.0
        bg_diff = mean_diff[t][bg_mask].mean()
        fg_diffs.append(fg_diff)
        bg_diffs.append(bg_diff)

    fg_diffs = np.array(fg_diffs)
    bg_diffs = np.array(bg_diffs)

    # Frame 0 (directly from edited input)
    print(f"\n  Frame 0:")
    print(f"    Mask region diff: {fg_diffs[0]:.2f}")
    print(f"    Background diff:  {bg_diffs[0]:.2f}")
    print(f"    Ratio (mask/bg):  {fg_diffs[0]/max(bg_diffs[0],1e-6):.2f}x")

    # Frames 1+ (model-generated)
    print(f"\n  Frames 1-{num_frames-1} (model generated):")
    print(f"    Mask region diff: mean={fg_diffs[1:].mean():.2f}, std={fg_diffs[1:].std():.2f}")
    print(f"    Background diff:  mean={bg_diffs[1:].mean():.2f}, std={bg_diffs[1:].std():.2f}")
    ratio = fg_diffs[1:].mean() / max(bg_diffs[1:].mean(), 1e-6)
    print(f"    Ratio (mask/bg):  {ratio:.2f}x")

    # Edit persistence: how much of frame 0's edit signal survives in later frames
    if fg_diffs[0] > 0:
        persistence = fg_diffs[1:].mean() / fg_diffs[0]
        print(f"    Edit persistence:  {persistence:.2f}x of frame 0 signal")
    else:
        persistence = 0.0
        print(f"    Edit persistence:  N/A (no frame 0 signal)")

    results["frame0_fg"] = fg_diffs[0]
    results["frame0_bg"] = bg_diffs[0]
    results["gen_fg_mean"] = fg_diffs[1:].mean()
    results["gen_fg_std"] = fg_diffs[1:].std()
    results["gen_bg_mean"] = bg_diffs[1:].mean()
    results["gen_bg_std"] = bg_diffs[1:].std()
    results["fg_bg_ratio"] = ratio
    results["edit_persistence"] = persistence

    return results


def main():
    base = "/scratch/users/ntu/song0304/loraedit_exp/processed_data"

    conditions = []

    # 1. Hair baseline (working reference)
    hair_orig = f"{base}/V7_hair/output_original_run4_ep100.mp4"
    hair_edit = f"{base}/V7_hair/output_edited_run4_ep100.mp4"
    hair_mask = f"{base}/V7_hair/source_masks"
    if os.path.exists(hair_orig) and os.path.exists(hair_edit):
        conditions.append(("Hair (Run4 ep100) — WORKING REFERENCE", hair_orig, hair_edit, hair_mask))

    # 2. Earring original mask
    ear_orig = f"{base}/V7_earring/output_original_run4_ep100.mp4"
    ear_edit = f"{base}/V7_earring/output_edited_run4_ep100.mp4"
    ear_mask = f"{base}/V7_earring/source_masks"
    if os.path.exists(ear_orig) and os.path.exists(ear_edit):
        conditions.append(("Earring Original Mask (Run4 ep100)", ear_orig, ear_edit, ear_mask))

    # 3. Earring expanded mask
    ear_exp_orig = f"{base}/V7_earring_expanded/output_original_expanded_ep100.mp4"
    ear_exp_edit = f"{base}/V7_earring_expanded/output_edited_expanded_ep100.mp4"
    ear_exp_mask = f"{base}/V7_earring_expanded/source_masks"
    if os.path.exists(ear_exp_orig) and os.path.exists(ear_exp_edit):
        conditions.append(("Earring Expanded Mask (+33%, ep100)", ear_exp_orig, ear_exp_edit, ear_exp_mask))

    if not conditions:
        print("ERROR: No conditions found!")
        return

    print("=" * 60)
    print("  OBJECTIVE EARRING VERIFICATION")
    print("  Comparing original vs edited inference outputs")
    print("=" * 60)

    all_results = []
    for name, orig, edit, mask_dir in conditions:
        result = analyze_condition(name, orig, edit, mask_dir)
        all_results.append(result)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Condition':<40} {'Mask%':>6} {'FG diff':>8} {'BG diff':>8} {'FG/BG':>6} {'Persist':>8}")
    print(f"{'-'*40} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for r in all_results:
        print(f"{r['name']:<40} {r['mask_coverage']:>5.1f}% {r['gen_fg_mean']:>8.2f} {r['gen_bg_mean']:>8.2f} {r['fg_bg_ratio']:>5.1f}x {r['edit_persistence']:>7.2f}x")

    print(f"\n{'='*80}")
    print("Key metrics:")
    print("  FG diff  = Mean pixel diff in mask region between original & edited outputs (frames 1+)")
    print("  BG diff  = Mean pixel diff in background between original & edited outputs (frames 1+)")
    print("  FG/BG    = Ratio: how much more the mask region differs vs background")
    print("  Persist  = Edit persistence: FG diff (frames 1+) / FG diff (frame 0)")
    print("             >1.0 = model amplifies edit signal, ~1.0 = preserves, <1.0 = loses")


if __name__ == "__main__":
    main()
