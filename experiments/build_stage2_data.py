"""Build Stage 2 training data for earring reference frame experiment.

Stage 2 (paper 3.3.2): Continue from Stage 1 LoRA with edited reference frame
to teach the model earring appearance.

Training video modifications (K=81, total=244 frames):
  - Frame 0 (CLIP): Edited first frame (with earring)
  - Frame 1 (condition pos 0): Edited first frame (full, no gray mask)
  - Frame 41 (condition pos 40): Original frame 40, grayed (SAME as Stage 1)
  - Frame 82 (target pos 0): Edited first frame (with earring)
  - Frame 122 (target pos 40): Edited frame 40 (Gemini-generated)
  - Frame 163 (mask pos 0): All-black (SAME)
  - Frame 203 (mask pos 40): Spatial mask (SAME)
  - Everything else: SAME as Stage 1
"""

import os
import shutil
import numpy as np
import imageio.v3 as iio
from pathlib import Path


def apply_gray_to_mask(image: np.ndarray, bbox_mask: np.ndarray) -> np.ndarray:
    """Set mask (white) regions to gray (128) in image."""
    output = image.copy()
    output[bbox_mask > 200] = 128
    return output


def build_stage2_training_video(
    source_frames_dir: str,
    source_masks_dir: str,
    edited_first_frame_path: str,
    edited_frame40_path: str,
    output_path: str,
    num_frames: int = 81,
    ref_frame_idx: int = 40,
) -> None:
    """Build the 1+3K training video for Stage 2."""

    # Load source frames
    frame_files = sorted([
        f for f in os.listdir(source_frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f[0].isdigit()
    ])
    # Deduplicate by stem (prefer .jpg over .png)
    stem_to_file: dict[str, str] = {}
    for f in frame_files:
        stem = os.path.splitext(f)[0]
        if stem not in stem_to_file or f.lower().endswith('.jpg'):
            stem_to_file[stem] = f
    frame_files = [stem_to_file[s] for s in sorted(stem_to_file)][:num_frames]

    source_frames = []
    for f in frame_files:
        img = iio.imread(os.path.join(source_frames_dir, f))[:, :, :3]
        source_frames.append(img)
    print(f"Loaded {len(source_frames)} source frames, shape={source_frames[0].shape}")

    # Load masks
    mask_files = sorted([
        f for f in os.listdir(source_masks_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f[0].isdigit()
    ])[:num_frames]

    bbox_masks = []
    for f in mask_files:
        m = iio.imread(os.path.join(source_masks_dir, f))
        if m.ndim == 3:
            m = m[:, :, 0]
        bbox_masks.append(m)
    print(f"Loaded {len(bbox_masks)} masks")

    # Load edited frames
    edited_first = iio.imread(edited_first_frame_path)[:, :, :3]
    edited_ref = iio.imread(edited_frame40_path)[:, :, :3]
    print(f"Edited first frame: {edited_first.shape}")
    print(f"Edited ref frame (idx {ref_frame_idx}): {edited_ref.shape}")

    h, w = source_frames[0].shape[:2]
    all_frames = []

    # === Frame 0 (CLIP): Edited first frame ===
    all_frames.append(edited_first)

    # === Frames 1-K (Condition): pseudo-video ===
    # Position 0: Edited first frame (full, no gray mask applied)
    all_frames.append(edited_first)
    # Positions 1 to K-1: Original frames with gray mask applied (same as Stage 1)
    for i in range(1, num_frames):
        grayed = apply_gray_to_mask(source_frames[i], bbox_masks[i])
        all_frames.append(grayed)

    # === Frames K+1 to 2K (Target): training targets ===
    for i in range(num_frames):
        if i == 0:
            # Position 0: Edited first frame (target matches condition)
            all_frames.append(edited_first)
        elif i == ref_frame_idx:
            # Position ref_frame_idx: Edited reference frame
            all_frames.append(edited_ref)
        else:
            # Other positions: Original frames (same as Stage 1)
            all_frames.append(source_frames[i])

    # === Frames 2K+1 to 3K (Mask): spatial binary masks ===
    black_mask = np.zeros((h, w, 3), dtype=np.uint8)
    all_frames.append(black_mask)  # Frame 0 mask: all-black (fully known)
    for i in range(1, num_frames):
        mask_3ch = np.stack([bbox_masks[i]] * 3, axis=-1)
        all_frames.append(mask_3ch)

    total = len(all_frames)
    expected = 1 + 3 * num_frames
    assert total == expected, f"Expected {expected} frames, got {total}"
    print(f"Built training video: {total} frames (1+3*{num_frames})")

    # Highlight changes
    print(f"\nStage 2 modifications:")
    print(f"  Frame 0 (CLIP):         edited_image.png (was: original first frame)")
    print(f"  Frame 1 (cond pos 0):   edited_image.png full (was: original first frame)")
    print(f"  Frame {num_frames+1+0} (target pos 0):  edited_image.png (was: original first frame)")
    print(f"  Frame {num_frames+1+ref_frame_idx} (target pos {ref_frame_idx}): edited_frame40.png (was: original frame {ref_frame_idx})")

    # Encode to video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import imageio
    writer = imageio.get_writer(output_path, fps=5, codec="libx264",
                                quality=None, pixelformat="yuv420p",
                                output_params=["-crf", "18"])
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved: {output_path}")


def main():
    base = "/scratch/users/ntu/song0304/loraedit_exp/processed_data"
    stage2_dir = f"{base}/V7_earring_stage2"
    expanded_dir = f"{base}/V7_earring_expanded"

    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(f"{stage2_dir}/traindata", exist_ok=True)

    # Symlink shared files
    for name in ["source_frames", "source_masks", "prefix.txt", "video_meta.json",
                  "inference_rgb.mp4", "inference_mask.mp4", "edited_image.png"]:
        src = os.path.join(expanded_dir, name)
        dst = os.path.join(stage2_dir, name)
        if os.path.exists(dst):
            if os.path.islink(dst):
                os.unlink(dst)
            else:
                continue
        if os.path.exists(src):
            os.symlink(src, dst)
            print(f"Symlinked: {name}")

    # Copy edited_frame40.png
    shutil.copy2(f"{expanded_dir}/edited_frame40.png", f"{stage2_dir}/edited_frame40.png")

    # Copy caption
    caption_src = f"{expanded_dir}/traindata/sequence_all_frames_81.txt"
    caption_dst = f"{stage2_dir}/traindata/sequence_all_frames_81.txt"
    if os.path.exists(caption_src):
        shutil.copy2(caption_src, caption_dst)
        print(f"Copied caption")

    # Build Stage 2 training video
    build_stage2_training_video(
        source_frames_dir=f"{expanded_dir}/source_frames",
        source_masks_dir=f"{expanded_dir}/source_masks",
        edited_first_frame_path=f"{expanded_dir}/edited_image.png",
        edited_frame40_path=f"{expanded_dir}/edited_frame40.png",
        output_path=f"{stage2_dir}/traindata/sequence_all_frames_81.mp4",
        num_frames=81,
        ref_frame_idx=40,
    )

    print(f"\n=== Stage 2 data directory: {stage2_dir} ===")
    print("Ready for training!")


if __name__ == "__main__":
    main()
