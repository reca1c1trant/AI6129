"""
First-frame editing tool for LoRAEdit experiments.
Supports color-based edits (hair color, clothes color, etc.) via mask + HSV manipulation.

Usage:
    # 颜色替换（最常用）
    python tools/edit_first_frame.py \
        --input frame_0000.png \
        --mask mask.png \
        --output edited_frame.png \
        --mode color \
        --target_color 255 215 0   # RGB, e.g. golden

    # 亮度/饱和度调整
    python tools/edit_first_frame.py \
        --input frame_0000.png \
        --mask mask.png \
        --output edited_frame.png \
        --mode adjust \
        --brightness 1.2 \
        --saturation 1.5

    # 预览 mask 覆盖区域（debug 用）
    python tools/edit_first_frame.py \
        --input frame_0000.png \
        --mask mask.png \
        --output preview.png \
        --mode preview
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


def load_mask(mask_path: str, size: tuple[int, int]) -> np.ndarray:
    """Load mask and normalize to [0, 1]. White=edit region."""
    mask = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
    mask_np = np.array(mask).astype(np.float32) / 255.0
    return mask_np


def color_replace(
    image: np.ndarray,
    mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    blend_strength: float = 0.85,
) -> np.ndarray:
    """Replace color in masked region while preserving luminance structure."""
    import colorsys

    target_h, target_s, _ = colorsys.rgb_to_hsv(
        target_rgb[0] / 255.0, target_rgb[1] / 255.0, target_rgb[2] / 255.0
    )

    result = image.copy().astype(np.float32)
    # Convert to HSV
    img_pil = Image.fromarray(image)
    hsv = np.array(img_pil.convert("HSV")).astype(np.float32)

    # Replace H and S in masked region, keep V (luminance)
    new_h = np.full_like(hsv[:, :, 0], target_h * 255)
    new_s = np.full_like(hsv[:, :, 1], target_s * 255)

    mask_3d = mask[:, :, np.newaxis]
    hsv[:, :, 0] = hsv[:, :, 0] * (1 - mask * blend_strength) + new_h * mask * blend_strength
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - mask * blend_strength) + new_s * mask * blend_strength

    # Convert back to RGB
    hsv_img = Image.fromarray(hsv.astype(np.uint8), mode="HSV")
    edited = np.array(hsv_img.convert("RGB"))

    # Blend with original using mask
    result = image * (1 - mask_3d) + edited * mask_3d
    return result.astype(np.uint8)


def adjust_region(
    image: np.ndarray,
    mask: np.ndarray,
    brightness: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Adjust brightness/saturation in masked region."""
    img_pil = Image.fromarray(image)

    adjusted = img_pil.copy()
    if brightness != 1.0:
        adjusted = ImageEnhance.Brightness(adjusted).enhance(brightness)
    if saturation != 1.0:
        adjusted = ImageEnhance.Color(adjusted).enhance(saturation)

    adjusted_np = np.array(adjusted).astype(np.float32)
    original_np = image.astype(np.float32)
    mask_3d = mask[:, :, np.newaxis]

    result = original_np * (1 - mask_3d) + adjusted_np * mask_3d
    return result.astype(np.uint8)


def preview_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask on image for visualization (red tint on masked region)."""
    result = image.copy().astype(np.float32)
    red_overlay = np.zeros_like(result)
    red_overlay[:, :, 0] = 255  # Red channel

    mask_3d = mask[:, :, np.newaxis]
    result = result * (1 - mask_3d * 0.4) + red_overlay * mask_3d * 0.4
    return result.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit first frame for LoRAEdit")
    parser.add_argument("--input", type=str, required=True, help="Input frame path")
    parser.add_argument("--mask", type=str, required=True, help="Mask path (white=edit)")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["color", "adjust", "preview"],
        default="color",
    )
    # color mode args
    parser.add_argument(
        "--target_color",
        type=int,
        nargs=3,
        default=[255, 215, 0],
        help="Target RGB color, e.g. 255 215 0 for golden",
    )
    parser.add_argument(
        "--blend_strength",
        type=float,
        default=0.85,
        help="Color blend strength [0-1]",
    )
    # adjust mode args
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--saturation", type=float, default=1.0)

    args = parser.parse_args()

    image = np.array(Image.open(args.input).convert("RGB"))
    h, w = image.shape[:2]
    mask = load_mask(args.mask, (w, h))

    if args.mode == "color":
        result = color_replace(
            image, mask, tuple(args.target_color), args.blend_strength
        )
        print(f"Color replaced to RGB({args.target_color}) in masked region")
    elif args.mode == "adjust":
        result = adjust_region(image, mask, args.brightness, args.saturation)
        print(f"Adjusted brightness={args.brightness}, saturation={args.saturation}")
    elif args.mode == "preview":
        result = preview_mask(image, mask)
        print("Preview: red tint shows masked region")

    Image.fromarray(result).save(args.output)
    print(f"Saved to {args.output}")


# 常用颜色参考
COLOR_PRESETS = {
    "golden": (255, 215, 0),
    "red": (220, 20, 60),
    "blue": (30, 80, 220),
    "white": (245, 245, 245),
    "black": (20, 20, 20),
    "brown": (139, 69, 19),
    "silver": (192, 192, 192),
    "green": (34, 139, 34),
    "pink": (255, 105, 180),
    "purple": (128, 0, 128),
}

if __name__ == "__main__":
    main()
