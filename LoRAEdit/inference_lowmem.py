"""Memory-efficient inference script for LoRAEdit.

Bypasses diffsynth's ModelManager for the large DiT model (7 shards, ~61GB)
to avoid OOM on nodes with <= 64GB RAM. Loads shards one at a time instead
of accumulating all into a single dict.
"""
import torch
import gc
import os
import glob
import argparse
import random

import safetensors
from PIL import Image
from diffsynth.models import ModelManager
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.utils import init_weights_on_device
from custom_wan_pipe import WanVideoPipeline

# Reuse helpers from inference.py
from inference import (
    find_max_epoch_lora, find_input_image, validate_paths
)

# Known config for Wan2.1-I2V-14B (hash: 6bfcfb3b342cb286ce886889d519a77e)
WAN_I2V_14B_CONFIG: dict = {
    "has_image_input": True,
    "patch_size": [1, 2, 2],
    "in_dim": 36,
    "dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 40,
    "num_layers": 40,
    "eps": 1e-6,
}

# Keys to skip (same filter as WanModelStateDictConverter.from_civitai)
SKIP_PREFIXES = ("vace",)
SKIP_FIRST_COMPONENTS = {"pose_patch_embedding", "face_adapter", "face_encoder", "motion_encoder"}


def load_dit_shard_by_shard(
    model_root_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> WanModel:
    """Create WanModel on meta device, then load weights shard-by-shard.
    Uses GPU for fast fp32->bf16 dtype conversion."""
    import time as _time
    from safetensors.torch import load_file as st_load_file

    pattern = os.path.join(model_root_dir, "diffusion_pytorch_model*.safetensors")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No diffusion model shards found: {pattern}")

    print(f"Creating WanModel on meta device...")
    with init_weights_on_device():
        dit = WanModel(**WAN_I2V_14B_CONFIG)
    dit.eval()

    param_names = {name for name, _ in dit.named_parameters()}
    t_all = _time.time()
    for shard_idx, shard_path in enumerate(shards):
        t0 = _time.time()
        print(f"  Loading shard {shard_idx + 1}/{len(shards)}: {os.path.basename(shard_path)}")
        raw = st_load_file(shard_path, device='cpu')
        converted: dict[str, torch.Tensor] = {}
        for key, tensor in raw.items():
            if any(key.startswith(p) for p in SKIP_PREFIXES):
                continue
            first_comp = key.split(".")[0]
            if first_comp in SKIP_FIRST_COMPONENTS:
                continue
            param_key = key[len("model."):] if key.startswith("model.") else key
            if param_key in param_names:
                if tensor.dtype != torch_dtype:
                    converted[param_key] = tensor.cuda().to(torch_dtype).cpu()
                else:
                    converted[param_key] = tensor
        del raw
        dit.load_state_dict(converted, strict=False, assign=True)
        del converted
        gc.collect()
        torch.cuda.empty_cache()
        print(f"    shard {shard_idx+1}: {_time.time()-t0:.1f}s")

    print(f"  DiT loaded in {_time.time()-t_all:.1f}s")
    return dit


def save_video_imageio(frames: list, output_path: str, fps: int = 30) -> None:
    """Save video frames using imageio (no ffmpeg binary needed)."""
    import imageio
    import numpy as np

    writer = imageio.get_writer(output_path, fps=fps, quality=5)
    for frame in frames:
        if isinstance(frame, Image.Image):
            writer.append_data(np.array(frame))
        else:
            writer.append_data(frame)
    writer.close()


def run_inference_once(pipe, generated_prompt: str, input_image: Image.Image,
                       pseudo_video_path: str, mask_video_path: str) -> list:
    """Run a single inference pass, return list of PIL frames."""
    random_seed = random.randint(0, 2**32 - 1)
    print(f"Seed: {random_seed}")
    video = pipe(
        prompt=generated_prompt,
        negative_prompt=(
            "Overexposure, static, blurred details, subtitles, paintings, pictures, "
            "still, overall gray, worst quality, low quality, JPEG compression residue, "
            "ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, "
            "deformed, disfigured, deformed limbs, fused fingers, cluttered background, "
            "three legs, a lot of people in the background, upside down"
        ),
        input_image=input_image,
        pseudo_video_path=pseudo_video_path,
        mask_video_path=mask_video_path,
        num_inference_steps=30,
        seed=random_seed,
        tiled=True,
        tea_cache_l1_thresh=0.275,
        tea_cache_model_id="Wan2.1-I2V-14B-480P",
    )
    return video


def save_comparison_video(frames_original: list, frames_edited: list,
                          output_path: str, fps: float) -> None:
    """Save side-by-side comparison video (original left, edited right)."""
    import imageio
    import numpy as np

    writer = imageio.get_writer(output_path, fps=fps, quality=5)
    for fo, fe in zip(frames_original, frames_edited):
        arr_o = np.array(fo) if isinstance(fo, Image.Image) else fo
        arr_e = np.array(fe) if isinstance(fe, Image.Image) else fe
        combined = np.concatenate([arr_o, arr_e], axis=1)
        writer.append_data(combined)
    writer.close()
    print(f"Comparison video saved to: {output_path}")


def find_specific_epoch_lora(data_dir: str, epoch: int, use_additional: bool = False) -> str:
    """Find LoRA checkpoint for a specific epoch in the latest training run."""
    lora_dir_name = "lora_additional" if use_additional else "lora"
    lora_base_dir = os.path.join(data_dir, lora_dir_name)

    date_dirs = sorted(d for d in os.listdir(lora_base_dir)
                       if os.path.isdir(os.path.join(lora_base_dir, d)))
    if not date_dirs:
        raise FileNotFoundError(f"No training directories found in: {lora_base_dir}")

    latest_date_dir = date_dirs[-1]
    epoch_path = os.path.join(lora_base_dir, latest_date_dir, f"epoch{epoch}", "adapter_model.safetensors")
    if not os.path.exists(epoch_path):
        available = [d for d in os.listdir(os.path.join(lora_base_dir, latest_date_dir))
                     if d.startswith("epoch")]
        raise FileNotFoundError(
            f"epoch{epoch} not found in {latest_date_dir}. Available: {sorted(available)}")

    print(f"Using specified LoRA: epoch{epoch} - {epoch_path}")
    return epoch_path


def main(model_root_dir: str, data_dir: str, prompt: str = "",
         use_additional: bool = False, output_name: str = "edited_video.mp4",
         compare: bool = False, epoch: int | None = None,
         num_persistent_params: int | None = None) -> None:
    # Validate paths
    print("Validating paths...")
    validate_paths(model_root_dir, data_dir)

    # Infer paths
    if epoch is not None:
        lora_path = find_specific_epoch_lora(data_dir, epoch, use_additional=use_additional)
    else:
        lora_path = find_max_epoch_lora(data_dir, use_additional=use_additional)
    input_image_path = find_input_image(data_dir)
    pseudo_video_path = os.path.join(data_dir, "inference_rgb.mp4")
    mask_video_path = os.path.join(data_dir, "inference_mask.mp4")

    for path in [pseudo_video_path, mask_video_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    print(f"Paths:")
    print(f"  Model root: {model_root_dir}")
    print(f"  LoRA:       {lora_path}")
    print(f"  Edited img: {input_image_path}")
    print(f"  Pseudo vid: {pseudo_video_path}")
    print(f"  Mask vid:   {mask_video_path}")

    # === Step 1: Create empty ModelManager ===
    model_manager = ModelManager(device="cpu", torch_dtype=torch.bfloat16)

    # === Step 2: Load small models normally (each is a single file) ===
    print("\n=== Loading CLIP encoder ===")
    model_manager.load_model(
        os.path.join(model_root_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        torch_dtype=torch.float32,
    )

    print("\n=== Loading T5 text encoder ===")
    model_manager.load_model(
        os.path.join(model_root_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        torch_dtype=torch.bfloat16,
    )
    gc.collect()

    print("\n=== Loading VAE ===")
    model_manager.load_model(
        os.path.join(model_root_dir, "Wan2.1_VAE.pth"),
        torch_dtype=torch.bfloat16,
    )
    gc.collect()

    # === Step 3: Load DiT shard-by-shard (memory-efficient) ===
    print("\n=== Loading DiT (shard-by-shard) ===")
    dit = load_dit_shard_by_shard(model_root_dir, torch_dtype=torch.bfloat16)

    # Inject DiT into ModelManager so LoRA loading and pipeline creation work
    diffusion_files = sorted(glob.glob(
        os.path.join(model_root_dir, "diffusion_pytorch_model*.safetensors")
    ))
    model_manager.model.append(dit)
    model_manager.model_path.append(diffusion_files)
    model_manager.model_name.append("wan_video_dit")
    gc.collect()

    # === Step 4: Load LoRA ===
    print(f"\n=== Loading LoRA ===")
    model_manager.load_lora(lora_path, lora_alpha=1.0)

    # === Step 5: Create pipeline ===
    print("\n=== Creating pipeline ===")
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )
    # num_persistent_param_in_dit: how many DiT params stay on GPU.
    # 0 = all on CPU (very slow, each step transfers ~28GB).
    # Higher = faster inference. 10B params ≈ 20GB VRAM for DiT weights.
    persist = num_persistent_params if num_persistent_params is not None else 0
    print(f"VRAM management: num_persistent_param_in_dit={persist}")
    pipe.enable_vram_management(num_persistent_param_in_dit=persist)

    # === Step 6: Prompt ===
    input_image = Image.open(input_image_path)

    if prompt:
        generated_prompt = prompt
        print(f"\n=== Using provided prompt ===")
        print(f"Prompt: {generated_prompt}")
    else:
        # Try Florence model for auto-captioning
        print("\n=== Generating caption with Florence ===")
        from inference import init_florence_model, generate_caption
        init_florence_model()

        prefix_file = os.path.join(data_dir, "prefix.txt")
        if os.path.exists(prefix_file):
            with open(prefix_file, "r", encoding="utf-8") as f:
                concept_prefix = f.read().strip()
            print(f"Concept prefix: {concept_prefix}")
        else:
            concept_prefix = "p3rs0n,"

        generated_prompt = generate_caption(input_image, concept_prefix=concept_prefix)
        print(f"Generated prompt: {generated_prompt}")

    # Read effective fps
    import json
    meta_path = os.path.join(data_dir, "video_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            source_fps = json.load(f).get("effective_fps", 5.0)
    else:
        import cv2
        cap = cv2.VideoCapture(pseudo_video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if source_fps <= 0:
            source_fps = 5.0
    print(f"Output video fps: {source_fps}")

    if compare:
        # === Compare mode: run original + edited, save both + comparison ===
        # Derive output prefix from output_name (e.g., "comparison_run3_ep100.mp4" -> "run3_ep100")
        stem = os.path.splitext(output_name)[0]
        if stem.startswith("comparison_"):
            suffix = "_" + stem[len("comparison_"):]
        elif stem == "comparison" or stem == "edited_video":
            suffix = ""
        else:
            suffix = "_" + stem

        original_frame_path = os.path.join(data_dir, "source_frames", "00000.png")
        original_image = Image.open(original_frame_path)
        print(f"\n=== Inference 1/2: original first frame ===")
        video_original = run_inference_once(pipe, generated_prompt, original_image,
                                            pseudo_video_path, mask_video_path)
        orig_name = f"output_original{suffix}.mp4"
        save_video_imageio(video_original, os.path.join(data_dir, orig_name), fps=source_fps)
        print(f"Saved: {orig_name}")

        print(f"\n=== Inference 2/2: edited first frame ===")
        video_edited = run_inference_once(pipe, generated_prompt, input_image,
                                          pseudo_video_path, mask_video_path)
        edited_name = f"output_edited{suffix}.mp4"
        save_video_imageio(video_edited, os.path.join(data_dir, edited_name), fps=source_fps)
        print(f"Saved: {edited_name}")

        comp_name = f"comparison{suffix}.mp4"
        save_comparison_video(video_original, video_edited,
                              os.path.join(data_dir, comp_name), fps=source_fps)
    else:
        # === Single inference ===
        print("\n=== Starting inference ===")
        video = run_inference_once(pipe, generated_prompt, input_image,
                                   pseudo_video_path, mask_video_path)
        output_path = os.path.join(data_dir, output_name)
        save_video_imageio(video, output_path, fps=source_fps)
        print(f"\nVideo saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-efficient LoRAEdit inference")
    parser.add_argument("--model_root_dir", required=True, help="Wan2.1 model directory")
    parser.add_argument("--data_dir", required=True, help="Processed data directory")
    parser.add_argument("--prompt", type=str, default="",
                        help="Manual prompt (skips Florence model if provided)")
    parser.add_argument("--additional", action="store_true",
                        help="Use additional LoRA from lora_additional/")
    parser.add_argument("--output_name", type=str, default="edited_video.mp4",
                        help="Output video filename")
    parser.add_argument("--compare", action="store_true",
                        help="Run both original and edited frames, output comparison video")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specify LoRA epoch checkpoint (default: max epoch)")
    parser.add_argument("--num_persistent_params", type=int, default=None,
                        help="Number of DiT params to keep on GPU (default: 0). "
                             "Higher = faster inference but more VRAM. Try 10000000000 for A100-40GB.")
    args = parser.parse_args()
    main(args.model_root_dir, args.data_dir, prompt=args.prompt,
         use_additional=args.additional, output_name=args.output_name,
         compare=args.compare, epoch=args.epoch,
         num_persistent_params=args.num_persistent_params)
