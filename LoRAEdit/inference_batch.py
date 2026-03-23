"""Batch inference for multiple LoRA epochs.

Optimizations vs inference_lowmem.py:
1. Models loaded ONCE (not per-epoch) — saves ~1min × (N-1) epochs
2. LoRA hot-swapped via AutoWrappedLinear native slots (no weight merge)
3. All epochs in single process — no Python startup/import overhead

Usage:
    python inference_batch.py \
        --model_root_dir /path/to/Wan2.1-I2V-14B-480P \
        --data_dir /path/to/processed_data/V7_hair \
        --epochs 50,100,150,200 \
        --prompt "a video scene" \
        --compare
"""
import torch
import gc
import os
import glob
import argparse
import random
import time
import json

from PIL import Image
from safetensors.torch import load_file as st_load_file

from diffsynth.models import ModelManager
from diffsynth.lora import GeneralLoRALoader
from custom_wan_pipe import WanVideoPipeline

from inference import find_input_image, validate_paths
from inference_lowmem import (
    load_dit_shard_by_shard,
    save_video_imageio,
    run_inference_once,
    save_comparison_video,
    find_specific_epoch_lora,
)


def set_native_lora(dit: torch.nn.Module, lora_path: str,
                    alpha: float = 1.0,
                    dtype: torch.dtype = torch.bfloat16) -> int:
    """Load LoRA into AutoWrappedLinear native slots (no weight merge).

    Math: F.linear gives out = x @ W^T.
    Native LoRA adds: out += x @ A^T @ B^T = x @ (BA)^T
    Equivalent to merged W' = W + alpha * B @ A when we scale B by alpha.
    """
    from diffsynth.vram_management.layers import AutoWrappedLinear

    lora_sd = st_load_file(lora_path, device='cpu')
    loader = GeneralLoRALoader(device='cpu', torch_dtype=dtype)
    name_map = loader.get_name_dict(lora_sd)  # target_name → (B_key, A_key)

    updated = 0
    for name, module in dit.named_modules():
        if name in name_map and isinstance(module, AutoWrappedLinear):
            b_key, a_key = name_map[name]
            lora_a = lora_sd[a_key].to(device='cuda', dtype=dtype)
            lora_b = lora_sd[b_key].to(device='cuda', dtype=dtype)
            if alpha != 1.0:
                lora_b = lora_b * alpha
            module.lora_A_weights = [lora_a]
            module.lora_B_weights = [lora_b]
            updated += 1

    del lora_sd
    gc.collect()
    return updated


def clear_native_lora(dit: torch.nn.Module) -> int:
    """Clear LoRA from all AutoWrappedLinear modules."""
    from diffsynth.vram_management.layers import AutoWrappedLinear
    cleared = 0
    for _, module in dit.named_modules():
        if isinstance(module, AutoWrappedLinear) and module.lora_A_weights:
            module.lora_A_weights = []
            module.lora_B_weights = []
            cleared += 1
    torch.cuda.empty_cache()
    return cleared


def main(model_root_dir: str, data_dir: str, epochs: list[int],
         prompt: str = "", use_additional: bool = False,
         compare: bool = False) -> None:

    t_total = time.time()

    # === Validate paths ===
    validate_paths(model_root_dir, data_dir)
    input_image_path = find_input_image(data_dir)
    pseudo_video_path = os.path.join(data_dir, "inference_rgb.mp4")
    mask_video_path = os.path.join(data_dir, "inference_mask.mp4")
    for p in [pseudo_video_path, mask_video_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")

    # ================================================================
    # Phase 1: Load base models ONCE
    # ================================================================
    t_phase = time.time()
    print("=" * 60)
    print("  Phase 1: Loading base models (once for all epochs)")
    print("=" * 60)

    model_manager = ModelManager(device="cpu", torch_dtype=torch.bfloat16)

    print("\n  Loading CLIP encoder...")
    model_manager.load_model(
        os.path.join(model_root_dir,
                     "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        torch_dtype=torch.float32,
    )

    print("  Loading T5 text encoder...")
    model_manager.load_model(
        os.path.join(model_root_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        torch_dtype=torch.bfloat16,
    )
    gc.collect()

    print("  Loading VAE...")
    model_manager.load_model(
        os.path.join(model_root_dir, "Wan2.1_VAE.pth"),
        torch_dtype=torch.bfloat16,
    )
    gc.collect()

    print("  Loading DiT (GPU shard-by-shard)...")
    dit = load_dit_shard_by_shard(model_root_dir, torch_dtype=torch.bfloat16)

    # Inject DiT into ModelManager
    diffusion_files = sorted(glob.glob(
        os.path.join(model_root_dir, "diffusion_pytorch_model*.safetensors")
    ))
    model_manager.model.append(dit)
    model_manager.model_path.append(diffusion_files)
    model_manager.model_name.append("wan_video_dit")
    gc.collect()

    print(f"\n  [TIMING] Base models loaded: {time.time()-t_phase:.1f}s")

    # ================================================================
    # Phase 2: Create pipeline ONCE (no LoRA merge needed)
    # ================================================================
    t_phase = time.time()
    print("\n" + "=" * 60)
    print("  Phase 2: Creating pipeline + VRAM management")
    print("=" * 60)

    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=0)

    print(f"  [TIMING] Pipeline created: {time.time()-t_phase:.1f}s")

    # ================================================================
    # Phase 3: Prepare prompt & images ONCE
    # ================================================================
    input_image = Image.open(input_image_path)

    if prompt:
        generated_prompt = prompt
        print(f"\n  Prompt: {generated_prompt}")
    else:
        from inference import init_florence_model, generate_caption
        init_florence_model()
        prefix_file = os.path.join(data_dir, "prefix.txt")
        if os.path.exists(prefix_file):
            with open(prefix_file, "r", encoding="utf-8") as f:
                concept_prefix = f.read().strip()
        else:
            concept_prefix = "p3rs0n,"
        generated_prompt = generate_caption(input_image, concept_prefix=concept_prefix)
        print(f"\n  Generated prompt: {generated_prompt}")

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
    print(f"  Output fps: {source_fps}")

    if compare:
        original_frame_path = os.path.join(data_dir, "source_frames", "00000.png")
        original_image = Image.open(original_frame_path)

    setup_time = time.time() - t_total
    print(f"\n  [TIMING] Total setup: {setup_time:.1f}s")

    # ================================================================
    # Phase 4: Loop over epochs with LoRA hot-swap
    # ================================================================
    for i, epoch in enumerate(epochs):
        print(f"\n{'=' * 60}")
        print(f"  Epoch {epoch} ({i+1}/{len(epochs)})")
        print(f"{'=' * 60}")
        t_epoch = time.time()

        # Find LoRA checkpoint
        lora_path = find_specific_epoch_lora(
            data_dir, epoch, use_additional=use_additional
        )

        # Set LoRA (native, no weight merge)
        t0 = time.time()
        n_updated = set_native_lora(pipe.dit, lora_path, alpha=1.0)
        print(f"  LoRA loaded: {n_updated} layers, {time.time()-t0:.1f}s")

        # Output naming
        suffix = f"_run4_ep{epoch}"

        if compare:
            # --- Inference 1/2: original first frame ---
            print(f"\n  --- Inference 1/2: original first frame ---")
            t0 = time.time()
            video_original = run_inference_once(
                pipe, generated_prompt, original_image,
                pseudo_video_path, mask_video_path,
            )
            t_infer1 = time.time() - t0
            print(f"  [TIMING] Inference 1: {t_infer1:.1f}s")

            orig_name = f"output_original{suffix}.mp4"
            save_video_imageio(
                video_original, os.path.join(data_dir, orig_name), fps=source_fps
            )
            print(f"  Saved: {orig_name}")

            # --- Inference 2/2: edited first frame ---
            print(f"\n  --- Inference 2/2: edited first frame ---")
            t0 = time.time()
            video_edited = run_inference_once(
                pipe, generated_prompt, input_image,
                pseudo_video_path, mask_video_path,
            )
            t_infer2 = time.time() - t0
            print(f"  [TIMING] Inference 2: {t_infer2:.1f}s")

            edited_name = f"output_edited{suffix}.mp4"
            save_video_imageio(
                video_edited, os.path.join(data_dir, edited_name), fps=source_fps
            )
            print(f"  Saved: {edited_name}")

            comp_name = f"comparison{suffix}.mp4"
            save_comparison_video(
                video_original, video_edited,
                os.path.join(data_dir, comp_name), fps=source_fps,
            )
        else:
            print(f"\n  --- Single inference ---")
            t0 = time.time()
            video = run_inference_once(
                pipe, generated_prompt, input_image,
                pseudo_video_path, mask_video_path,
            )
            t_infer = time.time() - t0
            print(f"  [TIMING] Inference: {t_infer:.1f}s")

            output_name = f"output_edited{suffix}.mp4"
            save_video_imageio(
                video, os.path.join(data_dir, output_name), fps=source_fps
            )
            print(f"  Saved: {output_name}")

        # Clear LoRA for next epoch
        n_cleared = clear_native_lora(pipe.dit)
        gc.collect()
        torch.cuda.empty_cache()

        t_epoch_total = time.time() - t_epoch
        print(f"\n  [TIMING] Epoch {epoch}: {t_epoch_total:.1f}s "
              f"(LoRA cleared: {n_cleared} layers)")

    # ================================================================
    # Summary
    # ================================================================
    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  All {len(epochs)} epochs done in {total_time:.1f}s")
    print(f"  Setup: {setup_time:.1f}s | "
          f"Inference: {total_time - setup_time:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch LoRAEdit inference (multi-epoch, single model load)"
    )
    parser.add_argument("--model_root_dir", required=True,
                        help="Wan2.1 model directory")
    parser.add_argument("--data_dir", required=True,
                        help="Processed data directory")
    parser.add_argument("--epochs", required=True,
                        help="Comma-separated epoch numbers (e.g., 50,100,150,200)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Manual prompt (skips Florence if provided)")
    parser.add_argument("--additional", action="store_true",
                        help="Use lora_additional/ directory")
    parser.add_argument("--compare", action="store_true",
                        help="Run original + edited, save comparison video")
    args = parser.parse_args()

    epoch_list = [int(e.strip()) for e in args.epochs.split(",")]
    main(
        model_root_dir=args.model_root_dir,
        data_dir=args.data_dir,
        epochs=epoch_list,
        prompt=args.prompt,
        use_additional=args.additional,
        compare=args.compare,
    )
