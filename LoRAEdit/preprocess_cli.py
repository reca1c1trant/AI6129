"""
CLI preprocessing script for LoRAEdit — replaces the Gradio UI (predata_app.py).
No GUI needed, suitable for HPC/remote servers.

Workflow:
  Step 1: Extract frames from video
    python preprocess_cli.py extract --video input.mp4 --output_dir ./processed_data/my_video

  Step 2: Look at the first frame, determine point coordinates for SAM2
    (open source_frames/00000.png, note the (x, y) of the object to edit)

  Step 3: Run SAM2 tracking + generate all LoRAEdit training data
    python preprocess_cli.py process \
      --data_dir ./processed_data/my_video \
      --points 320,240 \
      --ckpt_path /scratch/users/ntu/song0304/checkpoints/Wan2.1-I2V-14B-480P \
      --sam_checkpoint models_sam/sam2_hiera_large.pt

  Step 4: Edit the first frame and save as edited_image.png in data_dir
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
from PIL import Image


def extract_frames(
    video_path: str,
    output_dir: str,
    target_frames: int = 49,
    target_width: int = 832,
    target_height: int = 480,
    consecutive: bool = False,
) -> str:
    """Extract frames from video, resize/crop to target resolution."""
    source_dir = os.path.join(output_dir, "source_frames")
    os.makedirs(source_dir, exist_ok=True)

    # Read video
    reader = iio.get_reader(video_path)
    meta = reader.get_meta_data()
    total_frames = reader.count_frames()
    print(f"Video: {video_path}")
    print(f"  Total frames: {total_frames}, FPS: {meta.get('fps', 'unknown')}")
    print(f"  Target: {target_frames} frames, {target_width}x{target_height}")
    print(f"  Mode: {'consecutive (first N)' if consecutive else 'uniform sampling'}")

    if consecutive:
        # Take first N consecutive frames
        indices = list(range(min(target_frames, total_frames)))
    elif total_frames <= target_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / target_frames) for i in range(target_frames)]

    for out_idx, frame_idx in enumerate(indices):
        frame = reader.get_data(frame_idx)
        frame = resize_and_crop(frame, target_width, target_height)
        img = Image.fromarray(frame)
        # SAM2 only accepts .jpg files
        img.save(os.path.join(source_dir, f"{out_idx:05d}.jpg"), quality=95)
        # Also save first frame as PNG for editing reference
        if out_idx == 0:
            img.save(os.path.join(source_dir, f"{out_idx:05d}.png"))

    reader.close()
    print(f"Extracted {len(indices)} frames to {source_dir}")

    # Save video metadata for correct fps calculation later
    original_fps = meta.get('fps', 25)
    original_duration = total_frames / original_fps if original_fps > 0 else total_frames / 25.0
    if consecutive:
        # Consecutive frames: fps stays the same as original
        effective_fps = original_fps
    else:
        # Uniform sampling: N frames spread over full duration
        effective_fps = len(indices) / original_duration
    meta_path = os.path.join(output_dir, "video_meta.json")
    import json
    with open(meta_path, "w") as f:
        json.dump({
            "original_fps": original_fps,
            "original_total_frames": total_frames,
            "original_duration": original_duration,
            "extracted_frames": len(indices),
            "effective_fps": round(effective_fps, 2),
        }, f, indent=2)
    print(f"  Effective fps for extracted frames: {effective_fps:.2f}")

    return source_dir


def resize_and_crop(
    frame: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """Resize and center-crop frame to target dimensions."""
    h, w = frame.shape[:2]
    if w == target_width and h == target_height:
        return frame

    # Scale to cover target, then center crop
    scale = max(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h))

    start_x = (new_w - target_width) // 2
    start_y = (new_h - target_height) // 2
    frame = frame[start_y : start_y + target_height, start_x : start_x + target_width]
    return frame


def create_bbox_mask(
    binary_mask: np.ndarray, min_area: int = 100, expand_ratio: float = 0.1
) -> np.ndarray:
    """Create bounding box mask from binary mask (replicates predata_app.py logic)."""
    if binary_mask.dtype == bool:
        mask_img = (binary_mask * 255).astype(np.uint8)
    else:
        mask_img = (binary_mask * 255).astype(np.uint8)

    height, width = mask_img.shape
    bbox_mask = np.zeros_like(mask_img)
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        expand_w = int(w * expand_ratio)
        expand_h = int(h * expand_ratio)
        new_x = max(0, x - expand_w // 2)
        new_y = max(0, y - expand_h // 2)
        new_w = min(width - new_x, w + expand_w)
        new_h = min(height - new_y, h + expand_h)
        bbox_mask[new_y : new_y + new_h, new_x : new_x + new_w] = 255

    return bbox_mask


def apply_gray_to_mask(image: np.ndarray, bbox_mask: np.ndarray) -> np.ndarray:
    """Set mask (white) regions to gray (128) in image."""
    output = image.copy()
    output[bbox_mask > 200] = 128
    return output


def run_sam2_tracking(
    source_dir: str,
    sam_checkpoint: str,
    points: list[tuple[int, int]],
    labels: list[int] | None = None,
    sam_cfg: str = "sam2_hiera_l.yaml",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run SAM2 video tracking with point prompts."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if labels is None:
        labels = [1] * len(points)

    print(f"Loading SAM2 from {sam_checkpoint}...")
    predictor = build_sam2_video_predictor(sam_cfg, sam_checkpoint)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_path=source_dir)
        predictor.reset_state(inference_state)

        # Add point prompts on frame 0
        pts = np.array(points, dtype=np.float32)
        lbls = np.array(labels, dtype=np.int32)
        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=0,
            points=pts,
            labels=lbls,
        )

        # Propagate to all frames
        masks_all = []
        bbox_masks_all = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=0
        ):
            mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()
            masks_all.append(mask)
            bbox_mask = create_bbox_mask(mask, min_area=50, expand_ratio=0.1)
            bbox_masks_all.append(bbox_mask)

    print(f"SAM2 tracking done: {len(masks_all)} frames")
    return masks_all, bbox_masks_all


def generate_training_video(
    img_paths: list[str],
    bbox_masks_all: list[np.ndarray],
    output_dir: str,
) -> str:
    """Generate the 4-stage training video (replicates predata_app.py logic)."""
    train_dir = os.path.join(output_dir, "traindata")
    os.makedirs(train_dir, exist_ok=True)

    temp_dir = os.path.join(output_dir, "temp_train_seq")
    os.makedirs(temp_dir, exist_ok=True)

    images = [iio.imread(p)[:, :, :3] for p in img_paths]
    frame_idx = 0

    # Stage 0: First frame alone
    iio.imwrite(os.path.join(temp_dir, f"frame_{frame_idx:03d}.jpg"), images[0])
    frame_idx += 1

    # Stage 1: First frame copy + remaining frames gray-processed
    iio.imwrite(os.path.join(temp_dir, f"frame_{frame_idx:03d}.jpg"), images[0])
    frame_idx += 1
    for i in range(1, len(images)):
        grayed = apply_gray_to_mask(images[i], bbox_masks_all[i])
        iio.imwrite(os.path.join(temp_dir, f"frame_{frame_idx:03d}.jpg"), grayed)
        frame_idx += 1

    # Stage 2: All original frames
    for img in images:
        iio.imwrite(os.path.join(temp_dir, f"frame_{frame_idx:03d}.jpg"), img)
        frame_idx += 1

    # Stage 3: Masks (first frame all-black, rest use bbox masks)
    # Convert single-channel masks to 3-channel for video encoding
    h, w = images[0].shape[:2]
    black_mask = np.zeros((h, w, 3), dtype=np.uint8)
    iio.imwrite(os.path.join(temp_dir, f"frame_{frame_idx:03d}.jpg"), black_mask)
    frame_idx += 1
    for bbox_mask in bbox_masks_all[1:]:
        mask_3ch = np.stack([bbox_mask] * 3, axis=-1) if bbox_mask.ndim == 2 else bbox_mask
        iio.imwrite(os.path.join(temp_dir, f"frame_{frame_idx:03d}.jpg"), mask_3ch)
        frame_idx += 1

    # Encode to video using imageio
    total_frames = len(images)
    video_name = f"sequence_all_frames_{total_frames}.mp4"
    output_video = os.path.join(train_dir, video_name)

    all_frames = []
    for i in range(frame_idx):
        fpath = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
        all_frames.append(iio.imread(fpath))
    iio.mimwrite(output_video, all_frames, fps=5, quality=8)

    shutil.rmtree(temp_dir)
    print(f"Training video: {output_video}")
    return video_name


def _read_effective_fps(output_dir: str) -> float:
    """Read effective fps from saved video metadata, fallback to 5."""
    import json
    meta_path = os.path.join(output_dir, "video_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get("effective_fps", 5.0)
    return 5.0


def generate_inference_videos(
    img_paths: list[str],
    bbox_masks_all: list[np.ndarray],
    output_dir: str,
) -> None:
    """Generate inference_rgb.mp4 and inference_mask.mp4."""
    temp_rgb = os.path.join(output_dir, "temp_rgb")
    temp_mask = os.path.join(output_dir, "temp_mask")
    os.makedirs(temp_rgb, exist_ok=True)
    os.makedirs(temp_mask, exist_ok=True)

    for i, (img_path, bbox_mask) in enumerate(zip(img_paths, bbox_masks_all)):
        img = iio.imread(img_path)[:, :, :3]
        grayed = apply_gray_to_mask(img, bbox_mask)
        iio.imwrite(os.path.join(temp_rgb, f"frame_{i:04d}.jpg"), grayed)
        mask_3ch = np.stack([bbox_mask] * 3, axis=-1) if bbox_mask.ndim == 2 else bbox_mask
        iio.imwrite(os.path.join(temp_mask, f"frame_{i:04d}.jpg"), mask_3ch)

    effective_fps = _read_effective_fps(output_dir)
    print(f"Using effective fps for inference videos: {effective_fps}")

    for name, temp in [("inference_rgb.mp4", temp_rgb), ("inference_mask.mp4", temp_mask)]:
        out_path = os.path.join(output_dir, name)
        frame_files = sorted(
            [os.path.join(temp, f) for f in os.listdir(temp) if f.endswith(".jpg")]
        )
        frames = [iio.imread(f) for f in frame_files]
        iio.mimwrite(out_path, frames, fps=effective_fps, quality=8)
        shutil.rmtree(temp)
        print(f"Generated: {out_path}")


def save_masks(
    bbox_masks_all: list[np.ndarray], output_dir: str
) -> None:
    """Save bbox masks to source_masks directory."""
    masks_dir = os.path.join(output_dir, "source_masks")
    os.makedirs(masks_dir, exist_ok=True)
    for i, mask in enumerate(bbox_masks_all):
        iio.imwrite(os.path.join(masks_dir, f"{i:05d}.png"), mask)
    print(f"Masks saved to {masks_dir}")


def create_configs(
    output_dir: str,
    ckpt_path: str,
    video_name: str,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    save_every_n_epochs: int = 50,
    blocks_to_swap: int = 32,
    lora_rank: int = 16,
) -> None:
    """Create dataset.toml and training.toml."""
    configs_dir = os.path.join(output_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    train_data_path = os.path.join(output_dir, "traindata").replace(os.sep, "/")
    dataset_config = f"""enable_ar_bucket = false

[[directory]]
path = '{train_data_path}'
num_repeats = 1
"""
    dataset_path = os.path.join(configs_dir, "dataset.toml")
    with open(dataset_path, "w") as f:
        f.write(dataset_config)

    lora_output = os.path.join(output_dir, "lora").replace(os.sep, "/")
    training_config = f"""output_dir = '{lora_output}'
dataset = '{dataset_path.replace(os.sep, "/")}'

epochs = {epochs}
micro_batch_size_per_gpu = 1
pipeline_stages = 1
gradient_accumulation_steps = 1
gradient_clipping = 1
warmup_steps = 0

eval_every_n_epochs = 1000000
eval_before_first_step = true
eval_micro_batch_size_per_gpu = 1
eval_gradient_accumulation_steps = 1

save_every_n_epochs = {save_every_n_epochs}
checkpoint_every_n_minutes = 1000000000
activation_checkpointing = 'unsloth'
partition_method = 'parameters'
save_dtype = 'bfloat16'
caching_batch_size = 1
steps_per_print = 1
video_clip_mode = 'single_beginning'
blocks_to_swap = {blocks_to_swap}

[model]
type = 'wan'
ckpt_path = '{ckpt_path.replace(os.sep, "/")}'
dtype = 'bfloat16'
transformer_dtype = 'float8'
timestep_sample_method = 'uniform'

[adapter]
type = 'lora'
rank = {lora_rank}
dtype = 'bfloat16'
exclude_linear_modules = ["k_img", "v_img"]

[optimizer]
type = 'AdamW8bitKahan'
lr = {learning_rate}
betas = [0.9, 0.99]
weight_decay = 0.01
stabilize = true
"""
    training_path = os.path.join(configs_dir, "training.toml")
    with open(training_path, "w") as f:
        f.write(training_config)

    print(f"Configs saved to {configs_dir}")
    print(f"  dataset:  {dataset_path}")
    print(f"  training: {training_path}")


def create_caption(
    img_paths: list[str], output_dir: str, video_name: str, concept_prefix: str = ""
) -> None:
    """Generate caption for first frame using Florence-2."""
    try:
        from inference import init_florence_model, generate_caption
        init_florence_model()
        caption = generate_caption(img_paths[0], concept_prefix=concept_prefix)
    except Exception:
        # Fallback: simple caption
        caption = f"{concept_prefix} a video scene" if concept_prefix else "a video scene"
        print(f"Warning: Florence-2 not available, using fallback caption: {caption}")

    txt_path = os.path.join(output_dir, "traindata", f"{video_name.replace('.mp4', '.txt')}")
    with open(txt_path, "w") as f:
        f.write(caption)
    print(f"Caption saved: {txt_path}")
    print(f"  Content: {caption}")


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract frames from video."""
    extract_frames(
        args.video,
        args.output_dir,
        target_frames=args.num_frames,
        target_width=args.width,
        target_height=args.height,
        consecutive=args.consecutive,
    )
    first_frame = os.path.join(args.output_dir, "source_frames", "00000.png")
    print(f"\n{'='*60}")
    print(f"Next step: Look at {first_frame}")
    print(f"Note the (x, y) coordinates of the object you want to edit.")
    print(f"Then run:")
    print(f"  python preprocess_cli.py process \\")
    print(f"    --data_dir {args.output_dir} \\")
    print(f"    --points X,Y \\")
    print(f"    --ckpt_path /path/to/Wan2.1-I2V-14B-480P \\")
    print(f"    --sam_checkpoint models_sam/sam2_hiera_large.pt")
    print(f"{'='*60}")


def cmd_process(args: argparse.Namespace) -> None:
    """Run SAM2 tracking and generate all LoRAEdit training data."""
    source_dir = os.path.join(args.data_dir, "source_frames")
    if not os.path.exists(source_dir):
        print(f"Error: source_frames not found at {source_dir}")
        print(f"Run 'extract' first.")
        sys.exit(1)

    img_paths = sorted(
        [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(".jpg")]
    )
    print(f"Found {len(img_paths)} frames in {source_dir}")

    # Parse points
    points = []
    for pt_str in args.points:
        x, y = map(int, pt_str.split(","))
        points.append((x, y))
    labels = [1] * len(points)
    # Parse negative points if provided
    if args.neg_points:
        for pt_str in args.neg_points:
            x, y = map(int, pt_str.split(","))
            points.append((x, y))
            labels.append(0)

    print(f"Point prompts: {points}, labels: {labels}")

    # Run SAM2
    masks_all, bbox_masks_all = run_sam2_tracking(
        source_dir, args.sam_checkpoint, points, labels, args.sam_cfg
    )

    # Save masks
    save_masks(bbox_masks_all, args.data_dir)

    # Generate training video
    video_name = generate_training_video(img_paths, bbox_masks_all, args.data_dir)

    # Generate inference videos
    generate_inference_videos(img_paths, bbox_masks_all, args.data_dir)

    # Create configs
    create_configs(
        args.data_dir,
        args.ckpt_path,
        video_name,
        learning_rate=args.lr,
        epochs=args.epochs,
        save_every_n_epochs=args.save_every,
        blocks_to_swap=args.blocks_to_swap,
        lora_rank=args.lora_rank,
    )

    # Save prefix
    prefix_path = os.path.join(args.data_dir, "prefix.txt")
    with open(prefix_path, "w") as f:
        f.write(args.concept_prefix)
    print(f"Prefix saved: {prefix_path} -> '{args.concept_prefix}'")

    # Generate caption
    create_caption(img_paths, args.data_dir, video_name, args.concept_prefix)

    # Summary
    print(f"\n{'='*60}")
    print(f"Preprocessing complete! Output: {args.data_dir}")
    print(f"\nNext steps:")
    print(f"  1. Edit the first frame:")
    print(f"     cp {os.path.join(args.data_dir, 'source_frames', '00000.png')} /tmp/first_frame.png")
    print(f"     python tools/edit_first_frame.py --input /tmp/first_frame.png \\")
    print(f"       --mask {os.path.join(args.data_dir, 'source_masks', '00000.png')} \\")
    print(f"       --output {os.path.join(args.data_dir, 'edited_image.png')} \\")
    print(f"       --mode color --target_color 255 215 0")
    print(f"  2. Train LoRA:")
    print(f"     NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 train.py \\")
    print(f"       --deepspeed --config {os.path.join(args.data_dir, 'configs', 'training.toml')}")
    print(f"  3. Inference:")
    print(f"     python inference.py --model_root_dir {args.ckpt_path} --data_dir {args.data_dir}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRAEdit CLI Preprocessor")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Extract command
    p_extract = subparsers.add_parser("extract", help="Extract frames from video")
    p_extract.add_argument("--video", type=str, required=True, help="Input video path")
    p_extract.add_argument("--output_dir", type=str, required=True, help="Output directory")
    p_extract.add_argument("--num_frames", type=int, default=49, help="Number of frames to extract")
    p_extract.add_argument("--width", type=int, default=832, help="Target width")
    p_extract.add_argument("--height", type=int, default=480, help="Target height")
    p_extract.add_argument("--consecutive", action="store_true", help="Take first N consecutive frames instead of uniform sampling")

    # Process command
    p_process = subparsers.add_parser("process", help="Run SAM2 + generate training data")
    p_process.add_argument("--data_dir", type=str, required=True, help="Data directory (with source_frames/)")
    p_process.add_argument("--points", type=str, nargs="+", required=True, help="SAM2 point prompts as x,y (e.g. 320,240)")
    p_process.add_argument("--neg_points", type=str, nargs="*", help="Negative point prompts as x,y")
    p_process.add_argument("--sam_checkpoint", type=str, default="models_sam/sam2_hiera_large.pt")
    p_process.add_argument("--sam_cfg", type=str, default="sam2_hiera_l.yaml")
    p_process.add_argument("--ckpt_path", type=str, required=True, help="Wan2.1-I2V model path")
    p_process.add_argument("--concept_prefix", type=str, default="", help="Concept prefix for caption")
    p_process.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_process.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p_process.add_argument("--save_every", type=int, default=50, help="Save every N epochs")
    p_process.add_argument("--blocks_to_swap", type=int, default=32, help="Blocks to swap for VRAM saving")
    p_process.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")

    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "process":
        cmd_process(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
