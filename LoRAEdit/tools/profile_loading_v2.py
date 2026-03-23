"""Profile model loading V2 — compare set_module_tensor_to_device vs load_state_dict(assign=True)."""
import time
import sys
import os
import gc

sys.path.insert(0, '/home/users/ntu/song0304/code/LoRAEdit')
sys.path.insert(0, '/home/users/ntu/song0304/code/LoRAEdit/submodules/Wan2_1')

KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'time_', 'time_embedding', 'head']

def timed(label: str):
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

    import torch
    import toml
    from pathlib import Path
    from safetensors.torch import load_file
    import safetensors
    from accelerate import init_empty_weights

    config = toml.load(config_path)
    dtype = torch.bfloat16
    transformer_dtype_str = config['model'].get('transformer_dtype', 'bfloat16')
    transformer_dtype = torch.float8_e4m3fn if transformer_dtype_str == 'float8' else torch.bfloat16
    print(f"dtype={dtype}, transformer_dtype={transformer_dtype}")

    shards = sorted(Path(ckpt_path).glob('diffusion_pytorch_model*.safetensors'))

    # ============================================================
    # Method 1: Current approach (set_module_tensor_to_device)
    # Only test first shard to save time
    # ============================================================
    print("\n" + "=" * 60)
    print("METHOD 1: set_module_tensor_to_device (per-tensor)")
    print("=" * 60)

    from accelerate.utils import set_module_tensor_to_device
    from wan.modules.model import WanModel

    with init_empty_weights():
        model1 = WanModel.from_config(os.path.join(ckpt_path, 'config.json'))
    param_names = {name for name, _ in model1.named_parameters()}

    with timed("Method1 — Shard 1 only"):
        shard = shards[0]
        count = 0
        with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in param_names:
                    dtype_to_use = dtype if any(kw in key for kw in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                    set_module_tensor_to_device(model1, key, device='cpu', dtype=dtype_to_use, value=f.get_tensor(key))
                    count += 1
        print(f"  Loaded {count} tensors")

    del model1
    gc.collect()

    # ============================================================
    # Method 2: load_state_dict with assign=True (batch)
    # ============================================================
    print("\n" + "=" * 60)
    print("METHOD 2: load_state_dict(assign=True) (batch per shard)")
    print("=" * 60)

    with init_empty_weights():
        model2 = WanModel.from_config(os.path.join(ckpt_path, 'config.json'))
    param_names2 = {name for name, _ in model2.named_parameters()}

    with timed("Method2 — Shard 1 only"):
        shard = shards[0]

        t0 = time.time()
        shard_dict = load_file(str(shard), device='cpu')
        t_load = time.time() - t0
        print(f"  load_file: {t_load:.1f}s ({len(shard_dict)} keys)")

        t0 = time.time()
        converted = {}
        for key, tensor in shard_dict.items():
            if key in param_names2:
                dtype_to_use = dtype if any(kw in key for kw in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                converted[key] = tensor.to(dtype_to_use)
        t_convert = time.time() - t0
        print(f"  dtype convert: {t_convert:.1f}s ({len(converted)} tensors)")

        del shard_dict
        gc.collect()

        t0 = time.time()
        model2.load_state_dict(converted, strict=False, assign=True)
        t_assign = time.time() - t0
        print(f"  load_state_dict: {t_assign:.1f}s")

        del converted
        gc.collect()

    del model2
    gc.collect()

    # ============================================================
    # Method 3: load_file + assign for ALL shards
    # ============================================================
    print("\n" + "=" * 60)
    print("METHOD 3: load_state_dict(assign=True) — ALL 7 shards")
    print("=" * 60)

    with init_empty_weights():
        model3 = WanModel.from_config(os.path.join(ckpt_path, 'config.json'))
    param_names3 = {name for name, _ in model3.named_parameters()}

    with timed("Method3 — All shards"):
        for shard_idx, shard in enumerate(shards):
            t0 = time.time()
            shard_dict = load_file(str(shard), device='cpu')
            converted = {}
            for key, tensor in shard_dict.items():
                if key in param_names3:
                    dtype_to_use = dtype if any(kw in key for kw in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                    converted[key] = tensor.to(dtype_to_use)
            del shard_dict
            model3.load_state_dict(converted, strict=False, assign=True)
            del converted
            gc.collect()
            elapsed = time.time() - t0
            print(f"  Shard {shard_idx+1}/{len(shards)}: {elapsed:.1f}s")

    import resource
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  Peak RSS: {mem_mb:.0f} MB")

    # Quick sanity check: first param should not be on meta
    first_param = next(model3.parameters())
    print(f"  First param device: {first_param.device}, dtype: {first_param.dtype}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
