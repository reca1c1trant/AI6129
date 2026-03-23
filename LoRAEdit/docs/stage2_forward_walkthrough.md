# Stage 2 Forward Pass Walkthrough — Earring Case

## 训练视频结构 (1+3K = 244 frames, K=81)

```
Index 0        : CLIP frame     → edited_image.png (首帧，带耳环)
Index 1-81     : Condition      → [edited_image.png, frame01_grayed, ..., frame80_grayed]
Index 82-162   : Target         → [edited_image.png, frame01, ..., frame39, edited_frame40.png, frame41, ..., frame80]
Index 163-243  : Spatial Mask   → [all_black, mask01, ..., mask80]  (expanded +33%)
```

与 Stage 1 的**区别**（只改了 4 帧）：

| 位置 | Stage 1 | Stage 2 |
|------|---------|---------|
| Frame 0 (CLIP) | 原始首帧 | **编辑首帧（带耳环）** |
| Frame 1 (condition[0]) | 原始首帧 | **编辑首帧（带耳环，完整无灰化）** |
| Frame 82 (target[0]) | 原始首帧 | **编辑首帧（带耳环）** |
| Frame 122 (target[40]) | 原始 frame40 | **Gemini 编辑的 frame40（带耳环）** |

## Forward Pass 详解

### Step 1: 数据拆分 (`get_call_vae_fn`)

```
输入: tensor (bs=1, C=3, T=244, H=480, W=832)

拆分:
  first_frame     = tensor[:,:, 0:1]       → (1, 3, 1, 480, 832)   = 编辑首帧
  condition_frames = tensor[:,:, 1:82]      → (1, 3, 81, 480, 832)  = 伪视频
  target_frames    = tensor[:,:, 82:163]    → (1, 3, 81, 480, 832)  = 目标视频
  mask_frames      = tensor[:,:, 163:244]   → (1, 3, 81, 480, 832)  = 空间 mask
```

### Step 2: 编码

```
clip_context = CLIP_ViT(first_frame)
             → (1, 257, 1280)
             → 编辑首帧的全局语义特征（包含"耳环"概念）

y = VAE_encode(condition_frames)
  → (1, 16, 21, 60, 104)
  → 81帧 → VAE temporal compress 4x → 21 latent frames
  → condition[0] = 编辑首帧的 latent（带耳环信息）
  → condition[1-80] = 灰化区域的 latent（耳环区域被灰色覆盖）

latents = VAE_encode(target_frames)
        → (1, 16, 21, 60, 104)
        → target[0] = 编辑首帧的 latent
        → target[10] ≈ 编辑 frame40 的 latent（带耳环）  ← 4x 压缩后第40帧 ≈ 第10个latent frame
        → 其他 = 原始视频的 latent（无耳环）
```

### Step 3: 加噪 (`prepare_inputs`)

```
x_1 = latents                    # 干净的 target latent
x_0 = randn_like(x_1)           # 纯噪声
t   ~ Uniform(0, 1)             # 随机时间步

x_t = (1-t) * x_1 + t * x_0    # 加噪后的 latent
target = x_0 - x_1              # 模型需要预测的 flow (noise direction)
```

### Step 4: InitialLayer — 构建 36 通道输入

```
x_t: (1, 16, 21, 60, 104)      # 加噪 latent, 16 channels

mask 构建 (from mask_frames):
  mask_frames → mean(dim=1) → interpolate → binarize
  → (1, 4, 21, 60, 104)        # 4 channels

  mask[0] (frame 0):  全 1（all black mask → 已知）
  mask[1-20]:         耳环区域=0（需要生成），背景=1（已知）

y: (1, 16, 21, 60, 104)        # condition latent, 16 channels

拼接: [mask, y] → (1, 20, 21, 60, 104)
再拼接: [x_t; mask_y] → (1, 36, 21, 60, 104)
```

**36 通道含义：**
```
Ch  0-15:  x_t (加噪 latent, 模型要 denoise 的目标)
Ch 16-19:  spatial mask (哪些区域已知/需要生成)
Ch 20-35:  condition latent (伪视频的 VAE 编码)
           → Ch 20-35 的 frame 0 位置 = 编辑首帧 latent（带耳环！）
```

### Step 5: Patch Embedding + Attention

```
patch_embedding: (1, 36, 21, 60, 104) → patches
  patch_size = (1, 2, 2)
  → 21 × 30 × 52 = 32,760 tokens per sample
  → 每个 token: 36 × 1 × 2 × 2 = 144 → linear → dim=5120

3D RoPE 位置编码: 每个 token 知道自己在 (t, h, w) 的位置

经过 40 个 Transformer blocks:
  每个 block:
    1. Self-Attention (全局): 32,760 tokens 互相 attend
       → frame 0 的耳环 token 可以影响所有帧的耳环区域 token
       → frame 40 位置的 target 也带耳环信息
    2. Cross-Attention (text): 与文本 prompt 交互
    3. Cross-Attention (image): 与 CLIP 特征交互
       → clip_context 包含编辑首帧的语义（有耳环概念）
    4. FFN
```

### Step 6: Loss 计算

```
prediction = DiT(x_t, condition)    # 模型预测的 flow
target = x_0 - x_1                   # 真实 flow

loss = MSE(prediction, target)       # 全像素等权

关键：
  - frame 0 target = 编辑首帧 → 模型学习重建带耳环的首帧
  - frame 40 target = 编辑 frame40 → 模型学习重建带耳环的第 40 帧
  - 其他 frames target = 原始视频 → 模型保持运动一致性
```

## 推理时的行为

推理时用标准 pipeline（只有 frame 0 编辑首帧作为 condition）：

```
condition[0] = 编辑首帧 latent (带耳环)
condition[1-80] = 灰化原始帧
mask = 空间 mask (expanded)
clip_context = CLIP(编辑首帧) → 包含"耳环"语义

模型从纯噪声 denoise：
  - LoRA 权重已通过 Stage 2 学到"mask 区域 → 耳环"的映射
  - 首帧 condition 提供耳环的像素级参考
  - CLIP 提供耳环的语义级参考
  - Self-attention 让首帧耳环信息传播到后续帧
```

## Stage 2 的核心作用

Stage 1 的问题：模型从未见过"耳环"→ 不知道 mask 区域该生成什么。

Stage 2 的解决：
1. **CLIP 输入** = 编辑首帧 → 模型通过 cross-attention 获得"耳环"的全局语义
2. **Target frame 40** = 编辑 frame40 → 模型通过 MSE loss 直接学习"mask 区域的正确输出是耳环"
3. **Condition frame 0** = 编辑首帧 → 模拟推理时的真实输入，缩小 train/inference gap
