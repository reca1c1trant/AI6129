"""Profile model loading times to find bottlenecks in LoRA training startup."""
import time
import sys
import os
import gc

# Add project paths
sys.path.insert(0, '/home/users/ntu/song0304/code/LoRAEdit')
sys.path.insert(0, '/home/users/ntu/song0304/code/LoRAEdit/submodules/Wan2_1')

def timed(label: str):
    """Context manager to time a block of code."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"\n>>> START: {label}", flush=True)
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f">>> END:   {label} — {elapsed:.1f}s", flush=True)
    return Timer()

def main():
    total_start = time.time()

    ckpt_path = '/scratch/users/ntu/song0304/checkpoints/Wan2.1-I2V-14B-480P'
    config_path = '/scratch/users/ntu/song0304/loraedit_exp/processed_data/V7_hair/configs/training.toml'

    # === Stage 0: Imports ===
    with timed("Python imports"):
        import torch
        import toml
        import json
        from pathlib import Path
        import safetensors
        from accelerate.utils import set_module_tensor_to_device
        from accelerate import init_empty_weights

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # === Stage 1: Load config ===
    with timed("Load TOML config"):
        config = toml.load(config_path)

    # === Stage 2: T5 Text Encoder (from models/wan.py T5EncoderModel) ===
    with timed("Load T5 text encoder"):
        from models.wan import T5EncoderModel
        t5_model_path = os.path.join(ckpt_path, 'models_t5_umt5-xxl-enc-bf16.pth')
        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device='cpu',
            checkpoint_path=t5_model_path,
            tokenizer_path=os.path.join(ckpt_path, 'google/umt5-xxl'),
            shard_fn=None,
        )
    print(f"  T5 params: {sum(p.numel() for p in text_encoder.model.parameters()) / 1e9:.2f}B")

    # === Stage 3: VAE ===
    with timed("Load VAE"):
        from wan.modules.vae import WanVAE
        vae = WanVAE(
            vae_pth=os.path.join(ckpt_path, 'Wan2.1_VAE.pth'),
            device='cpu',
        )

    # === Stage 4: CLIP ===
    with timed("Load CLIP model"):
        from wan.modules.clip import CLIPModel
        clip = CLIPModel(
            dtype=torch.bfloat16,
            device='cpu',
            checkpoint_path=os.path.join(ckpt_path, 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
            tokenizer_path=os.path.join(ckpt_path, 'xlm-roberta-large'),
        )

    # === Stage 5: DiT (init empty weights) ===
    with timed("DiT init_empty_weights"):
        from wan.modules.model import WanModel
        with init_empty_weights():
            transformer = WanModel.from_config(os.path.join(ckpt_path, 'config.json'))

    # === Stage 6: DiT shard loading (our custom code) ===
    KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'time_', 'time_embedding', 'head']
    dtype = torch.bfloat16
    transformer_dtype_str = config['model'].get('transformer_dtype', 'bfloat16')
    transformer_dtype = torch.float8_e4m3fn if transformer_dtype_str == 'float8' else torch.bfloat16

    param_names = {name for name, _ in transformer.named_parameters()}
    shards = sorted(Path(ckpt_path).glob('diffusion_pytorch_model*.safetensors'))
    print(f"\n  Found {len(shards)} shards, {len(param_names)} parameters")
    print(f"  transformer_dtype: {transformer_dtype}")

    with timed("DiT load ALL shards"):
        for shard_idx, shard in enumerate(shards):
            t0 = time.time()
            count = 0
            with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                keys = f.keys()
                for key in keys:
                    if key in param_names:
                        dtype_to_use = dtype if any(kw in key for kw in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                        set_module_tensor_to_device(transformer, key, device='cpu', dtype=dtype_to_use, value=f.get_tensor(key))
                        count += 1
            elapsed = time.time() - t0
            print(f"  Shard {shard_idx+1}/{len(shards)}: {shard.name} — {elapsed:.1f}s ({count}/{len(keys)} matched)")
            gc.collect()

    # Check memory usage
    import resource
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  Peak RSS: {mem_mb:.0f} MB")

    # === Stage 7: LoRA setup (PEFT) ===
    with timed("PEFT LoRA configure_adapter"):
        import peft
        from torch import nn

        adapter_target_modules = ['WanAttentionBlock', 'WanTimeTextImageBlock']
        exclude_linear_modules = config['adapter'].get('exclude_linear_modules', [])

        target_linear_modules = set()
        for name, module in transformer.named_modules():
            if module.__class__.__name__ not in adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    should_exclude = any(exc in full_submodule_name for exc in exclude_linear_modules)
                    if not should_exclude:
                        target_linear_modules.add(full_submodule_name)

        print(f"  Target modules: {len(target_linear_modules)}")

        torch.manual_seed(42)
        peft_config = peft.LoraConfig(
            r=config['adapter']['rank'],
            lora_alpha=config['adapter']['rank'],
            lora_dropout=0.0,
            bias='none',
            target_modules=list(target_linear_modules),
        )
        lora_model = peft.get_peft_model(transformer, peft_config)
        lora_model.print_trainable_parameters()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
