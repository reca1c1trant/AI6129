"""Check if the edit in the first frame extends beyond the mask boundary.

Compares edited_image.png vs original first frame, overlays the difference
with the mask to see if edits leak outside the mask.
"""
import cv2
import numpy as np
import os
import sys


def check_overlap(data_dir: str, output_dir: str | None = None) -> None:
    if output_dir is None:
        output_dir = data_dir

    # Load images
    orig_path = os.path.join(data_dir, "source_frames", "00000.jpg")
    if not os.path.exists(orig_path):
        orig_path = os.path.join(data_dir, "source_frames", "00000.png")
    edited_path = os.path.join(data_dir, "edited_image.png")
    mask_path = os.path.join(data_dir, "source_masks", "00000.png")

    orig = cv2.imread(orig_path)
    edited = cv2.imread(edited_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    print(f"Original: {orig.shape}")
    print(f"Edited:   {edited.shape}")
    print(f"Mask:     {mask.shape}")

    # Resize edited to match original if needed
    if orig.shape[:2] != edited.shape[:2]:
        edited = cv2.resize(edited, (orig.shape[1], orig.shape[0]))
        print(f"Resized edited to {edited.shape}")

    # Compute pixel difference
    diff = np.abs(orig.astype(float) - edited.astype(float))
    diff_gray = diff.mean(axis=2)  # average across RGB

    mask_bool = mask > 127
    fg_ratio = mask_bool.mean()

    # Threshold: pixels with diff > 10 are considered "edited"
    threshold = 10
    edited_pixels = diff_gray > threshold

    # Compute overlap stats
    n_edited = edited_pixels.sum()
    n_edited_inside = (edited_pixels & mask_bool).sum()
    n_edited_outside = (edited_pixels & ~mask_bool).sum()

    print(f"\nMask coverage: {fg_ratio:.3f} ({mask_bool.sum()} pixels)")
    print(f"Edited pixels (diff>{threshold}): {n_edited}")
    print(f"  Inside mask:  {n_edited_inside} ({n_edited_inside/max(n_edited,1)*100:.1f}%)")
    print(f"  Outside mask: {n_edited_outside} ({n_edited_outside/max(n_edited,1)*100:.1f}%)")

    if n_edited_outside > 0:
        # Average diff inside vs outside mask
        avg_diff_inside = diff_gray[mask_bool].mean()
        avg_diff_outside_edited = diff_gray[edited_pixels & ~mask_bool].mean() if n_edited_outside > 0 else 0
        print(f"\n  Avg diff inside mask: {avg_diff_inside:.1f}")
        print(f"  Avg diff outside mask (edited pixels only): {avg_diff_outside_edited:.1f}")
        print(f"\n  >>> EDIT LEAKS OUTSIDE MASK! <<<")
    else:
        print(f"\n  Edit is fully contained within the mask.")

    # Save visualization
    vis = orig.copy()
    # Red overlay for mask boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

    # Green overlay for edit pixels outside mask
    outside_mask_vis = np.zeros_like(orig)
    outside_mask_vis[edited_pixels & ~mask_bool] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 1.0, outside_mask_vis, 0.5, 0)

    # Blue overlay for edit pixels inside mask
    inside_mask_vis = np.zeros_like(orig)
    inside_mask_vis[edited_pixels & mask_bool] = [255, 0, 0]
    vis = cv2.addWeighted(vis, 1.0, inside_mask_vis, 0.3, 0)

    vis_path = os.path.join(output_dir, "edit_mask_overlap.png")
    cv2.imwrite(vis_path, vis)
    print(f"\nVisualization saved: {vis_path}")
    print("  Red contour = mask boundary")
    print("  Green = edit OUTSIDE mask")
    print("  Blue = edit INSIDE mask")

    # Also save the diff heatmap
    diff_norm = (diff_gray / diff_gray.max() * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    # overlay mask contour
    cv2.drawContours(diff_color, contours, -1, (255, 255, 255), 2)
    diff_path = os.path.join(output_dir, "edit_diff_heatmap.png")
    cv2.imwrite(diff_path, diff_color)
    print(f"Diff heatmap saved: {diff_path}")


if __name__ == "__main__":
    data_dirs = {
        "V7_hair": "/scratch/users/ntu/song0304/loraedit_exp/processed_data/V7_hair",
        "V7_earring": "/scratch/users/ntu/song0304/loraedit_exp/processed_data/V7_earring",
    }
    for name, data_dir in data_dirs.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        out_dir = f"/scratch/users/ntu/song0304/loraedit_exp/experiments/{name}_overlap"
        os.makedirs(out_dir, exist_ok=True)
        check_overlap(data_dir, out_dir)
