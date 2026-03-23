# LoRAEdit 加速优化技术报告

> 日期: 2026-03-19 (updated)
> 环境: A100-SXM4-40GB, 64GB RAM, PyTorch 2.6+cu124, diffsynth 1.1.3
> 模型: Wan2.1-I2V-14B-480P (7 shards, fp32 checkpoint, ~61GB)
>
> **最终保留的优化**: GPU dtype 转换 + 单进程 batch + native LoRA 热换
> **测试后移除**: `num_persistent_params` (提速 5%, OOM 风险), `torch.compile` (提速 ≤4%, 破坏 LoRA)

---

## 一、概览

| 阶段 | 优化前 | 优化后 | 提速 |
|---|---|---|---|
| **训练 - 模型加载** | 16.3 min (CPU dtype 转换) | 58.5s (GPU 加速) | **16.7×** |
| **训练 - VRAM 安全性** | 38.8/40.9GB (95%, 不安全) | 35.8/40.9GB (87%) | blocks_to_swap 24 |
| **推理 - DiT 加载** | ~15 min × 每 epoch | ~1 min × **1 次** | **~60×** (含去重) |
| **推理 - 4 epoch 总时间** | ~80 min | **42 min** | **1.9×** |
| **推理 - LoRA 切换** | 重启进程+重加载全模型 (~10min) | 原生热换 (~6s) | **~100×** |

---

## 二、训练阶段优化

### 2.1 GPU 加速 fp32→bf16 转换

**问题根因**: Wan2.1 的 checkpoint 文件存储为 **fp32** 格式。diffsynth 在加载时对每个 shard（~9.85GB）做 `tensor.to(bfloat16)` 转换，全程在 CPU 上执行。CPU 的 fp32→bf16 转换极慢（每 shard 110-178s，7 shards 共 976s = 16.3 min）。

**修改文件**: `models/wan.py` — `load_diffusion_model()` 方法

**修改内容**: 所有 dtype 转换通过 GPU 加速

```python
# 修改前 (CPU 转换)
converted[key] = tensor.to(target_dt)

# 修改后 (GPU 加速转换)
if tensor.dtype != target_dt:
    converted[key] = tensor.cuda().to(target_dt).cpu()
else:
    converted[key] = tensor
```

**原理**: GPU 的向量化浮点转换远快于 CPU。数据路径：CPU→GPU（PCIe ~30GB/s）→ GPU 上转换（几乎瞬时）→ GPU→CPU（PCIe ~30GB/s）。9.85GB shard 的 PCIe round-trip 约 0.7s，远优于 CPU 转换的 110-178s。

**效果**: 976s → 58.5s（**16.7× 提速**）

### 2.2 blocks_to_swap 调整

**问题**: `blocks_to_swap=20` 时训练 peak VRAM = 38.8GB / 40.9GB (95%)，接近 OOM 边界。

**修改文件**: `V7_hair/configs/training.toml`, `V7_earring/configs/training.toml`

```toml
# 修改前
blocks_to_swap = 20  # 峰值 38.8GB (95%)

# 修改后
blocks_to_swap = 24  # 峰值 35.8GB (87%)，留 5.1GB 余量
```

**原理**: Wan2.1 DiT 有 40 个 transformer block。`blocks_to_swap` 控制多少个 block 在 forward/backward 时临时从 CPU 换入 GPU（类似虚拟内存的 swap）。增大此值 → GPU 上同时存在的 block 更少 → VRAM 占用更低，但训练速度略降。实测 24 vs 20 的训练速度无明显差异（~43s/step）。

### 2.3 Cosine Annealing LR

**问题**: 原始代码硬编码 `ConstantLR(factor=1.0)`。lr=0.001 恒定 + 单样本训练 → 超过 ~100 epoch 后过拟合。

**修改文件**: `train.py` — 训练循环末尾（`total_steps` 计算完成后）

```python
# 新增代码
lr_scheduler_config = config.get('lr_scheduler', {})
if lr_scheduler_config.get('type') == 'cosine':
    eta_min = lr_scheduler_config.get('eta_min', 0.0)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=model_engine.total_steps, eta_min=eta_min
    )
    model_engine.lr_scheduler = cosine_scheduler
```

**配置** (`training.toml`):
```toml
[optimizer]
lr = 0.0005

[lr_scheduler]
type = 'cosine'
eta_min = 1e-6
```

**注意**: scheduler 创建必须在 `train_data.post_init()` 之后，因为 `T_max = total_steps` 依赖 `len(train_data)`，而 `post_init()` 才设置数据集长度。

---

## 三、推理阶段优化

### 3.1 GPU 加速 DiT 加载（同训练）

**修改文件**: `inference_lowmem.py` — `load_dit_shard_by_shard()` 函数

与训练阶段相同的 GPU 加速 dtype 转换。效果：DiT 加载 ~15 min → ~1 min。

### 3.2 num_persistent_params

**修改文件**: `inference_lowmem.py` — 新增 `--num_persistent_params` 参数

```python
persist = num_persistent_params if num_persistent_params is not None else 0
pipe.enable_vram_management(num_persistent_param_in_dit=persist)
```

**原理**: diffsynth 的 VRAM management 将 DiT 的 `nn.Linear` 替换为 `AutoWrappedLinear`，通过 `num_persistent_param_in_dit` 控制多少参数常驻 GPU：

```
num_persistent_param = 0:
  所有权重在 CPU，每个 denoising step 临时搬到 GPU 再搬回

num_persistent_param = 10B:
  前 10B params (~20GB bf16) 常驻 GPU
  剩余 ~4B params 每 step 临时搬运
```

**实测结果**: 22s/step → 21s/step（**仅 5% 提速**）。

**原因分析**: 瓶颈不是数据搬运（28GB PCIe 传输仅需 ~0.9s），而是 **14B 模型对 ~32K tokens 的纯计算量**。持久化参数减少的搬运时间 (<1s) 相对于计算时间 (~20s) 可忽略。

### 3.3 单进程 Batch Inference（核心优化）

**新增文件**: `inference_batch.py`

**问题**: 原方案用 bash `for` 循环对每个 epoch 启动独立 Python 进程，**每次都重新加载所有模型**：

```
旧方案 (4 epochs):
  epoch50:  加载 CLIP+T5+VAE+DiT (~10min) → merge LoRA → inference ×2
  epoch100: 加载 CLIP+T5+VAE+DiT (~10min) → merge LoRA → inference ×2
  epoch150: 加载 CLIP+T5+VAE+DiT (~10min) → merge LoRA → inference ×2
  epoch200: 加载 CLIP+T5+VAE+DiT (~10min) → merge LoRA → inference ×2
  总计: ~80 min (加载占 40 min)
```

**新方案**:

```
新方案 (4 epochs):
  Phase 1: 加载 CLIP+T5+VAE+DiT (105s, 一次性)
  Phase 2: 创建 pipeline + VRAM management (0.8s, 一次性)
  Loop:
    epoch50:  set native LoRA (6s) → inference ×2 → clear LoRA
    epoch100: set native LoRA (6s) → inference ×2 → clear LoRA
    epoch150: set native LoRA (6s) → inference ×2 → clear LoRA
    epoch200: set native LoRA (6s) → inference ×2 → clear LoRA
  总计: 42 min (加载仅 105s)
```

### 3.4 Native LoRA 热换（关键技术）

**问题**: diffsynth 的 `GeneralLoRALoader.load()` 通过**破坏性合并**应用 LoRA：

```python
# diffsynth 原始方式 — 直接修改 base weight
state_dict["weight"] = state_dict["weight"] + alpha * torch.mm(lora_B, lora_A)
```

这意味着换 epoch 时无法恢复 base weight，必须重新加载 28GB 的 DiT。

**解决方案**: 利用 `AutoWrappedLinear` 内置的 LoRA 插槽：

```python
class AutoWrappedLinear(torch.nn.Linear):
    def __init__(self, ...):
        self.lora_A_weights = []  # 内置插槽!
        self.lora_B_weights = []

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        # LoRA 在 forward 时 on-the-fly 应用，不修改 base weight
        for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
            out = out + x @ lora_A.T @ lora_B.T
        return out
```

**实现** (`inference_batch.py`):

```python
def set_native_lora(dit, lora_path, alpha=1.0, dtype=torch.bfloat16):
    """直接设置 AutoWrappedLinear 的 LoRA 插槽，不修改 base weight"""
    lora_sd = load_file(lora_path, device='cpu')
    name_map = GeneralLoRALoader().get_name_dict(lora_sd)  # target → (B_key, A_key)

    for name, module in dit.named_modules():
        if name in name_map and isinstance(module, AutoWrappedLinear):
            b_key, a_key = name_map[name]
            module.lora_A_weights = [lora_sd[a_key].to('cuda', dtype)]
            module.lora_B_weights = [lora_sd[b_key].to('cuda', dtype) * alpha]

def clear_native_lora(dit):
    """清空所有 LoRA 插槽，恢复 base model"""
    for _, module in dit.named_modules():
        if isinstance(module, AutoWrappedLinear) and module.lora_A_weights:
            module.lora_A_weights = []
            module.lora_B_weights = []
```

**数学等价性验证**:

```
标准 LoRA 合并:
  y = (W + αBA)x
  F.linear: out = x @ W^T + α · x @ A^T @ B^T

Native LoRA (AutoWrappedLinear):
  out = F.linear(x, W) + x @ lora_A^T @ lora_B^T
  其中 lora_A = A, lora_B = αB
  → out = x @ W^T + α · x @ A^T @ B^T  ✓  等价
```

**LoRA 权重内存开销**:
- 400 层 × (rank 8 × dim 5120) × 2 matrices × 2 bytes (bf16) ≈ **64MB**
- 常驻 GPU，相对于 28GB DiT 可忽略

### 3.5 torch.compile（测试中）

**修改文件**: `inference_batch.py` — 新增 `--compile` flag

```python
if use_compile:
    pipe.dit = torch.compile(pipe.dit, mode="default", fullgraph=False)
```

**已知限制**: `AutoWrappedLinear.forward()` 中有基于 `torch.cuda.mem_get_info()` 的运行时分支，无法被静态图追踪。使用 `fullgraph=False` 允许在动态边界处插入 graph break，编译静态子图。

**状态**: Job 13243282 测试中。

---

## 四、文件修改清单

### 已修改的 LoRAEdit 源文件

| 文件 | 修改内容 |
|---|---|
| `models/wan.py` | GPU 加速 dtype 转换 + 逐 shard 加载 + timing prints |
| `train.py` | Cosine annealing scheduler + seed=42 + timing prints |

### 新增的自定义文件

| 文件 | 用途 |
|---|---|
| `inference_lowmem.py` | 低内存 inference（逐 shard DiT 加载 + GPU dtype 转换 + persistent params） |
| `inference_batch.py` | 批量推理（单次加载 + native LoRA 热换 + torch.compile） |
| `preprocess_cli.py` | CLI 预处理（替代 Gradio UI） |
| `tools/edit_first_frame.py` | 首帧颜色编辑 |

### 配置文件修改

| 文件 | 修改 |
|---|---|
| `V7_hair/configs/training.toml` | blocks_to_swap=24, lr=0.0005, cosine annealing |
| `V7_earring/configs/training.toml` | blocks_to_swap=24 |

---

## 五、Denoising 速度瓶颈分析

### 实测数据

| 配置 | 每 step 耗时 | 30 steps 总计 |
|---|---|---|
| persistent=0 (全 CPU offload) | ~22s | 4:51 |
| persistent=10B (70% on GPU) | ~21s | 4:33 |
| 差异 | **-1s (-5%)** | **-18s (-6%)** |

### 为什么 persistent params 帮助有限？

```
理论分析:

  数据搬运时间:
    28GB bf16 model, PCIe bandwidth ~30 GB/s
    全搬运: 28GB / 30 = 0.93s
    10B persistent: 8GB / 30 = 0.27s
    节省: 0.66s/step ← 和实测的 ~1s 吻合

  纯计算时间:
    14B params, ~32K tokens, 40 transformer blocks
    A100 BF16 Tensor Core: 312 TFLOPS
    实测: ~20s/step ← 计算主导

  结论:
    搬运占比 < 5%，计算占比 > 95%
    → persistent params 优化空间有限
    → 需要 torch.compile / 算子融合 才能加速计算
```

### TeaCache 效果

TeaCache 通过检测 token 变化量决定是否跳过整个 block 循环。从 progress bar 分析，30 steps 中约 17 个 step 被 cache 命中（跳过），实际只执行 ~13 次完整 forward pass。这已经大幅减少了计算量。

---

## 六、优化效果总结

### 4 epoch 对比推理端到端时间

```
Run3 (无任何优化):     ~80 min
  ├── 模型加载 ×4:      ~40 min  (CPU dtype 转换)
  ├── Denoising ×8:     ~39 min  (4:51/次, 全 offload)
  └── VAE + 其他:       ~1 min

Batch (全部优化):       ~42 min  (1.9× 提速)
  ├── 模型加载 ×1:      1.8 min  (GPU dtype 转换, 单次)
  ├── Denoising ×8:     ~37 min  (4:33/次, 10B persistent)
  ├── LoRA 热换 ×4:     ~24s     (native LoRA)
  └── VAE + 其他:       ~1 min
```

### 单次 inference 时间

```
旧 (完整流程):          ~15 min
  ├── 模型加载:          ~10 min
  └── Denoising + VAE:   ~5 min

新 (模型已加载):         ~5 min
  ├── LoRA 设置:          6s
  ├── Denoising:          4:33
  └── VAE decode:         8s
```
