"""Profile V3 — fine-grained timing per shard per step."""
import time
import sys
import os
import gc

sys.path.insert(0, '/home/users/ntu/song0304/code/LoRAEdit')
sys.path.insert(0, '/home/users/ntu/song0304/code/LoRAEdit/submodules/Wan2_1')

KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'time_', 'time_embedding', 'head']

def main():
    import torch
    import toml
    from pathlib import Path
    from safetensors.torch import load_file
    from accelerate import init_empty_weights
    from wan.modules.model import WanModel
    import resource

    ckpt_path = '/scratch/users/ntu/song0304/checkpoints/Wan2.1-I2V-14B-480P'
    config_path = '/scratch/users/ntu/song0304/loraedit_exp/processed_data/V7_hair/configs/training.toml'
    config = toml.load(config_path)

    dtype = torch.bfloat16
    transformer_dtype_str = config['model'].get('transformer_dtype', 'bfloat16')
    transformer_dtype = torch.float8_e4m3fn if transformer_dtype_str == 'float8' else torch.bfloat16

    shards = sorted(Path(ckpt_path).glob('diffusion_pytorch_model*.safetensors'))

    with init_empty_weights():
        model = WanModel.from_config(os.path.join(ckpt_path, 'config.json'))
    param_names = {name for name, _ in model.named_parameters()}

    print(f"Model: {len(param_names)} params, {len(shards)} shards")
    print(f"transformer_dtype={transformer_dtype}")
    print()

    total_start = time.time()
    for shard_idx, shard in enumerate(shards):
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"=== Shard {shard_idx+1}/{len(shards)}: {shard.name} (RSS={mem_before:.0f}MB) ===", flush=True)

        # Step 1: load_file (mmap)
        t0 = time.time()
        shard_dict = load_file(str(shard), device='cpu')
        t_load = time.time() - t0
        print(f"  1. load_file:       {t_load:.2f}s ({len(shard_dict)} keys)", flush=True)

        # Step 2: dtype convert
        t0 = time.time()
        converted = {}
        for key, tensor in shard_dict.items():
            if key in param_names:
                dtype_to_use = dtype if any(kw in key for kw in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                converted[key] = tensor.to(dtype_to_use)
        t_convert = time.time() - t0
        print(f"  2. dtype convert:   {t_convert:.2f}s ({len(converted)} tensors)", flush=True)

        # Step 3: delete shard_dict
        t0 = time.time()
        del shard_dict
        t_del = time.time() - t0
        print(f"  3. del shard_dict:  {t_del:.2f}s", flush=True)

        # Step 4: load_state_dict
        t0 = time.time()
        model.load_state_dict(converted, strict=False, assign=True)
        t_assign = time.time() - t0
        print(f"  4. load_state_dict: {t_assign:.2f}s", flush=True)

        # Step 5: cleanup
        t0 = time.time()
        del converted
        gc.collect()
        t_gc = time.time() - t0
        print(f"  5. del+gc.collect:  {t_gc:.2f}s", flush=True)

        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"  RSS after: {mem_after:.0f}MB", flush=True)
        print(flush=True)

    total = time.time() - total_start
    print(f"TOTAL: {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
