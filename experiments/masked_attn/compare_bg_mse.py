"""Compare background-region MSE between different LoRA reconstructions and GT.

Usage:
    python compare_bg_mse.py \
        --data_dir /path/to/V7_hair \
        --videos output_original_run4_ep100.mp4 output_original_masked_attn_ep100.mp4
"""
import argparse
import cv2
import numpy as np
import os
from typing import Optional


def extract_video_frames(video_path: str) -> np.ndarray:
    """Extract all frames from a video file as RGB uint8 array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def load_gt_frames(source_frames_dir: str) -> np.ndarray:
    """Load GT frames from source_frames/ directory."""
    exts = ('.png', '.jpg', '.jpeg')
    all_files = [f for f in os.listdir(source_frames_dir) if f.lower().endswith(exts)]
    # Deduplicate by stem (prefer jpg over png if both exist)
    stem_to_file: dict[str, str] = {}
    for f in sorted(all_files):
        stem = os.path.splitext(f)[0]
        if stem not in stem_to_file or f.lower().endswith('.jpg'):
            stem_to_file[stem] = f
    files = [stem_to_file[s] for s in sorted(stem_to_file)]
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(source_frames_dir, f))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def load_masks(mask_dir: str) -> np.ndarray:
    """Load mask frames, return boolean array (True = foreground/masked region)."""
    files = sorted(f for f in os.listdir(mask_dir) if f.endswith('.png'))
    masks = []
    for f in files:
        img = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        masks.append(img > 127)
    return np.stack(masks)


def compute_region_mse(
    gt_frames: np.ndarray,
    gen_frames: np.ndarray,
    mask_frames: np.ndarray,
    region: str = "background"
) -> dict:
    """Compute per-frame and average MSE for background or foreground region.

    Args:
        gt_frames: [F, H, W, 3] uint8 GT frames
        gen_frames: [F, H, W, 3] uint8 generated frames
        mask_frames: [F, H, W] bool, True = foreground
        region: "background" or "foreground"

    Returns:
        dict with mean_mse, psnr, per_frame_mse
    """
    n_frames = min(len(gt_frames), len(gen_frames), len(mask_frames))
    gt = gt_frames[:n_frames].astype(np.float64)
    gen = gen_frames[:n_frames].astype(np.float64)
    masks = mask_frames[:n_frames]

    # Resize generated frames to match GT if needed
    if gt.shape[1:3] != gen.shape[1:3]:
        resized = []
        for frame in gen:
            r = cv2.resize(frame.astype(np.uint8), (gt.shape[2], gt.shape[1]),
                           interpolation=cv2.INTER_LINEAR)
            resized.append(r.astype(np.float64))
        gen = np.stack(resized)

    per_frame_mse = []
    for i in range(n_frames):
        if region == "background":
            pixel_mask = ~masks[i]  # background = NOT foreground
        else:
            pixel_mask = masks[i]

        n_pixels = pixel_mask.sum()
        if n_pixels == 0:
            continue

        # Expand mask to cover RGB channels
        mask_3c = np.stack([pixel_mask] * 3, axis=-1)
        diff_sq = (gt[i] - gen[i]) ** 2
        mse = diff_sq[mask_3c].mean()
        per_frame_mse.append(mse)

    mean_mse = float(np.mean(per_frame_mse))
    psnr = 10 * np.log10(255**2 / mean_mse) if mean_mse > 0 else float('inf')

    return {
        'mean_mse': mean_mse,
        'psnr': psnr,
        'n_frames': len(per_frame_mse),
        'per_frame_mse': per_frame_mse,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare background MSE of reconstructions vs GT")
    parser.add_argument("--data_dir", required=True, help="Processed data directory (contains source_frames/, source_masks/)")
    parser.add_argument("--videos", nargs="+", required=True, help="Video filenames to compare (relative to data_dir)")
    parser.add_argument("--region", default="background", choices=["background", "foreground", "both"],
                        help="Which region to compute MSE for")
    args = parser.parse_args()

    # Load GT and masks
    gt_dir = os.path.join(args.data_dir, "source_frames")
    mask_dir = os.path.join(args.data_dir, "source_masks")

    print("Loading GT frames...")
    gt_frames = load_gt_frames(gt_dir)
    print(f"  GT: {gt_frames.shape}")

    print("Loading masks...")
    mask_frames = load_masks(mask_dir)
    fg_ratio = mask_frames.mean()
    print(f"  Masks: {mask_frames.shape}, foreground ratio: {fg_ratio:.3f}")

    regions = ["background", "foreground"] if args.region == "both" else [args.region]

    # Compare each video
    print(f"\n{'='*70}")
    for region in regions:
        print(f"\n  Region: {region.upper()}")
        print(f"  {'Video':<45} {'MSE':>8} {'PSNR':>8}")
        print(f"  {'-'*65}")

        for video_name in args.videos:
            video_path = os.path.join(args.data_dir, video_name)
            if not os.path.exists(video_path):
                print(f"  {video_name:<45} {'NOT FOUND':>8}")
                continue

            gen_frames = extract_video_frames(video_path)
            result = compute_region_mse(gt_frames, gen_frames, mask_frames, region)
            print(f"  {video_name:<45} {result['mean_mse']:8.2f} {result['psnr']:7.2f}dB")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
