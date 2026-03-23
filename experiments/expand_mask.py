"""Expand earring mask downward by a fraction of its height.

Creates expanded masks in source_masks_expanded/ and a comparison visualization.
"""
import cv2
import numpy as np
import os
import argparse


def expand_mask_down(mask: np.ndarray, expand_pixels: int) -> np.ndarray:
    """Expand mask downward: for each column, extend the bottom edge down."""
    expanded = mask.copy()
    h, w = mask.shape

    for col in range(w):
        col_data = mask[:, col]
        fg_rows = np.where(col_data > 0)[0]
        if len(fg_rows) == 0:
            continue
        bottom = fg_rows[-1]
        new_bottom = min(bottom + expand_pixels, h - 1)
        expanded[bottom:new_bottom + 1, col] = 255

    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--expand_ratio", type=float, default=0.25,
                        help="Expand downward by this fraction of mask height")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir for expanded masks (default: data_dir/source_masks_expanded)")
    args = parser.parse_args()

    mask_dir = os.path.join(args.data_dir, "source_masks")
    output_dir = args.output_dir or os.path.join(args.data_dir, "source_masks_expanded")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(mask_dir) if f.endswith('.png'))
    print(f"Found {len(files)} mask frames")

    # Compute expansion from first mask's bounding box
    first_mask = cv2.imread(os.path.join(mask_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    fg_rows = np.where(first_mask.max(axis=1) > 127)[0]
    mask_height = fg_rows[-1] - fg_rows[0] + 1
    expand_pixels = int(mask_height * args.expand_ratio)
    print(f"Mask bounding box height: {mask_height}px")
    print(f"Expanding downward by: {expand_pixels}px ({args.expand_ratio:.0%})")

    # Process all frames
    for f in files:
        mask = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 127).astype(np.uint8) * 255
        expanded = expand_mask_down(mask_bin, expand_pixels)
        cv2.imwrite(os.path.join(output_dir, f), expanded)

    # Coverage comparison
    orig_mask = cv2.imread(os.path.join(mask_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    exp_mask = cv2.imread(os.path.join(output_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    orig_ratio = (orig_mask > 127).mean()
    exp_ratio = (exp_mask > 127).mean()
    print(f"\nCoverage: {orig_ratio:.3f} -> {exp_ratio:.3f}")

    # Save side-by-side visualization (original mask | expanded mask | overlay on first frame)
    orig_frame_path = os.path.join(args.data_dir, "source_frames", "00000.jpg")
    if not os.path.exists(orig_frame_path):
        orig_frame_path = os.path.join(args.data_dir, "source_frames", "00000.png")
    orig_frame = cv2.imread(orig_frame_path)

    # Panel 1: original mask on frame
    panel1 = orig_frame.copy()
    overlay1 = np.zeros_like(panel1)
    overlay1[orig_mask > 127] = [0, 0, 255]  # red
    panel1 = cv2.addWeighted(panel1, 0.7, overlay1, 0.3, 0)
    cv2.putText(panel1, f"Original ({orig_ratio:.1%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Panel 2: expanded mask on frame
    panel2 = orig_frame.copy()
    overlay2 = np.zeros_like(panel2)
    overlay2[exp_mask > 127] = [0, 255, 0]  # green = expanded
    overlay2[orig_mask > 127] = [0, 0, 255]  # red = original
    panel2 = cv2.addWeighted(panel2, 0.7, overlay2, 0.3, 0)
    cv2.putText(panel2, f"Expanded ({exp_ratio:.1%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Panel 3: expanded mask + edited image overlay
    edited_path = os.path.join(args.data_dir, "edited_image.png")
    edited = cv2.imread(edited_path)
    if edited.shape[:2] != orig_frame.shape[:2]:
        edited = cv2.resize(edited, (orig_frame.shape[1], orig_frame.shape[0]))
    diff = np.abs(orig_frame.astype(float) - edited.astype(float)).mean(axis=2)
    edit_pixels = diff > 10
    panel3 = orig_frame.copy()
    overlay3 = np.zeros_like(panel3)
    overlay3[exp_mask > 127] = [0, 255, 0]   # green = mask
    overlay3[edit_pixels & (exp_mask <= 127)] = [0, 0, 255]  # red = edit outside mask
    panel3 = cv2.addWeighted(panel3, 0.7, overlay3, 0.3, 0)
    n_outside = (edit_pixels & (exp_mask <= 127)).sum()
    n_edit = edit_pixels.sum()
    leak = n_outside / max(n_edit, 1) * 100
    cv2.putText(panel3, f"Edit leak: {leak:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    vis = np.hstack([panel1, panel2, panel3])
    vis_path = os.path.join(output_dir, "comparison.png")
    cv2.imwrite(vis_path, vis)
    print(f"Visualization: {vis_path}")


if __name__ == "__main__":
    main()
