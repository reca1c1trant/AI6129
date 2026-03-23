# Wan2.1 I2V DiT 架构分析

> 模型: Wan2.1-I2V-14B-480P
> 参数量: ~14B
> 基于: Wan2.1 官方代码 + diffsynth 1.1.3 重写

---

## 1. 模型总体参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_type` | `'i2v'` | Image-to-Video |
| `patch_size` | `(1, 2, 2)` | 时间不下采样，空间 2x |
| `in_dim` | **36** | 16 noise + 4 mask + 16 condition |
| `dim` | 5120 | hidden dimension |
| `ffn_dim` | 13824 | FFN 中间层 |
| `freq_dim` | 256 | timestep embedding |
| `text_dim` | 4096 | UMT5-XXL 输出维度 |
| `out_dim` | 16 | 输出通道（预测噪声） |
| `num_heads` | 40 | attention heads |
| `head_dim` | 128 | = 5120 / 40 |
| `num_layers` | **40** | transformer blocks |
| `window_size` | `(-1, -1)` | 全局 attention |
| `text_len` | 512 | text tokens 最大长度 |

关键差异：I2V 的 `in_dim=36`（T2V 为 16）。多出的 20 通道来自 mask(4) + condition latent(16)。

---

## 2. 输入

DiT 接收 5 个输入：

| 输入 | Shape | 来源 |
|------|-------|------|
| `x` (noise latent) | `(B, 16, F_lat, H_lat, W_lat)` | noisy latent |
| `y` (condition) | `(B, 20, F_lat, H_lat, W_lat)` | 4-ch mask + 16-ch VAE latent |
| `timestep` | `(B,)` | diffusion timestep [0, 1000] |
| `context` (text) | `(B, L_text, 4096)` | UMT5-XXL 编码 |
| `clip_feature` (image) | `(B, 257, 1280)` | CLIP ViT-H/14 前 31 层 |

对于 81 帧 480×832 视频：
- VAE 下采样：时间 4x，空间 8x → `F_lat=21, H_lat=60, W_lat=104`
- Patchify 后：空间再 2x → `f=21, h=30, w=52`
- **序列长度** = 21 × 30 × 52 = **32,760 tokens**

---

## 3. 首帧信息的三条注入路径

### 路径 1: Channel Concatenation（最直接的信号）

```
x: (B, 16, F, H, W)  ← noisy latent
y: (B, 20, F, H, W)  ← [4-ch mask | 16-ch VAE latent of condition]
                         首帧 mask=1, 其余=0; 首帧有真实 VAE latent

concat → (B, 36, F, H, W)
Conv3d(36, 5120, kernel=(1,2,2), stride=(1,2,2)) → (B, 5120, f, h, w)
rearrange → (B, f*h*w, 5120)
```

mask 告诉模型"首帧是已知的"，VAE latent 提供首帧的像素级信息。

### 路径 2: CLIP Cross-Attention（全局语义信号）

```
首帧 → resize 224×224 → CLIP ViT-H/14 (前31层) → (B, 257, 1280)
     → MLPProj(1280 → 5120) → (B, 257, 5120)
     → cat with text → context: (B, 257+L_text, 5120)
```

在每个 DiTBlock 的 cross-attention 中，用独立的 `k_img, v_img` 做图像 cross-attention。

### 路径 3: Self-Attention（帧间信息传播）

全局 self-attention (`window_size=(-1,-1)`)，所有帧的 tokens 互相 attend。
首帧的 32,760/21 = 1,560 个 tokens 可以与后续帧的所有 tokens 交互。
通过 3D RoPE 编码位置，模型知道哪些 tokens 来自首帧。

---

## 4. Forward Pass 详细流程

### Step 1: Time Embedding

```python
t = Linear(256→5120) → SiLU → Linear(5120→5120)     # (B, 5120)
t_mod = SiLU → Linear(5120→30720) → unflatten(6, 5120)  # (B, 6, 5120)
# 6 组调制向量: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
```

### Step 2: Text Embedding

```python
context = Linear(4096→5120) → GELU → Linear(5120→5120)  # (B, L_text, 5120)
```

### Step 3: Image Injection (I2V 特有)

```python
x = cat([x, y], dim=1)           # channel concat: (B, 36, F, H, W)
clip_emb = MLPProj(clip_feature)  # (B, 257, 5120)
context = cat([clip_emb, context], dim=1)  # (B, 257+L_text, 5120)
```

### Step 4: Patchify

```python
x = Conv3d(36→5120, kernel=(1,2,2), stride=(1,2,2))  # (B, 5120, f, h, w)
x = rearrange('b c f h w → b (f h w) c')              # (B, 32760, 5120)
```

### Step 5: 3D RoPE

```python
# head_dim=128 拆分为 3 段:
#   temporal: 44 维
#   height:   42 维
#   width:    42 维
# 总计: 44+42+42 = 128
freqs = cat([
    freqs_t[:f].expand(f,h,w,-1),  # (f,h,w,44)
    freqs_h[:h].expand(f,h,w,-1),  # (f,h,w,42)
    freqs_w[:w].expand(f,h,w,-1),  # (f,h,w,42)
], dim=-1).reshape(f*h*w, 1, -1)   # (32760, 1, 128)
```

### Step 6: 40 个 Transformer Blocks

```python
for block in self.blocks:  # 40 次
    x = block(x, context, t_mod, freqs)
```

### Step 7: Head + Unpatchify

```python
x = LayerNorm(x)
x = modulate(x, shift, scale)  # AdaLN with timestep
x = Linear(5120 → 64)          # 64 = 16 × 1 × 2 × 2
x = rearrange('b (f h w) (c pt ph pw) → b c (f*pt) (h*ph) (w*pw)')
# 输出: (B, 16, F_lat, H_lat, W_lat)
```

---

## 5. DiTBlock 内部结构

每个 block 包含三个子层：

```
┌──────────────────────────────────────────────────────┐
│                    DiTBlock                           │
│                                                      │
│  1. Self-Attention (AdaLN modulated + gated)         │
│     norm1(x) → modulate(shift_msa, scale_msa)       │
│     → Q, K, V projections → QK-Norm (RMSNorm)       │
│     → 3D RoPE on Q, K                               │
│     → flash_attention (全局, 32760 tokens)           │
│     → output projection                             │
│     x = x + gate_msa * self_attn_out                │
│                                                      │
│  2. Cross-Attention (双路: text + image)              │
│     norm3(x) → Q projection → QK-Norm               │
│                                                      │
│     Text path:                                       │
│       K = norm_k(Linear_k(text_ctx))                 │
│       V = Linear_v(text_ctx)                         │
│       text_out = flash_attn(Q, K, V)                 │
│                                                      │
│     Image path (独立权重 k_img, v_img):               │
│       K_img = norm_k_img(Linear_k_img(clip_ctx))     │
│       V_img = Linear_v_img(clip_ctx)                 │
│       img_out = flash_attn(Q, K_img, V_img)          │
│                                                      │
│     cross_out = output_proj(text_out + img_out)      │
│     x = x + cross_out  (无 gate)                     │
│                                                      │
│  3. FFN (AdaLN modulated + gated)                    │
│     norm2(x) → modulate(shift_mlp, scale_mlp)       │
│     → Linear(5120→13824) → GELU(tanh) → Linear→5120 │
│     x = x + gate_mlp * ffn_out                       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 关键设计细节

1. **QK-Norm**: Self-attention 和 Cross-attention 的 Q, K 都过 RMSNorm，稳定训练
2. **AdaLN Modulation**: 只作用于 self-attention 和 FFN（有 shift/scale/gate），cross-attention 没有
3. **独立 Image Cross-Attention**: `k_img, v_img` 是独立的 Linear 层，不与 text 的 `k, v` 共享权重
4. **Gate Mechanism**: self-attention 和 FFN 有 gate（初始化接近 0），cross-attention 无 gate
5. **可学习 modulation 基线**: 每个 block 有 `self.modulation` 参数 `(1, 6, 5120)`，加上 time-dependent `t_mod`

---

## 6. Self-Attention 完整计算

```python
def forward(self, x, freqs):
    # x: (B, S, 5120), S = 32760

    q = RMSNorm(Linear_q(x))   # (B, S, 5120) → reshape (B, S, 40, 128)
    k = RMSNorm(Linear_k(x))   # (B, S, 5120) → reshape (B, S, 40, 128)
    v = Linear_v(x)             # (B, S, 5120) → reshape (B, S, 40, 128)

    # 3D RoPE: complex multiplication
    q = view_as_complex(q)      # (B, S, 40, 64) complex
    q = q * freqs               # element-wise multiply with precomputed frequencies
    q = view_as_real(q)         # (B, S, 40, 128)
    k = same_process(k)

    # Flash Attention (全局, causal=False)
    out = flash_attn_func(q, k, v)  # (B, S, 40, 128)
    out = out.flatten(2)             # (B, S, 5120)

    return Linear_o(out)
```

原始 Wan 代码使用 `flash_attn_varlen_func`（支持 batch 内不同长度序列），diffsynth 版使用标准 `flash_attn_func`。

---

## 7. Cross-Attention 完整计算 (I2V)

```python
class WanI2VCrossAttention:
    def forward(self, x, context):
        # context: (B, 257+L_text, 5120)
        # 前 257 = CLIP image tokens (1 CLS + 256 patches from ViT-H/14)
        # 后 L_text = UMT5 text tokens

        img_ctx = context[:, :257]     # (B, 257, 5120)
        txt_ctx = context[:, 257:]     # (B, L_text, 5120)

        q = RMSNorm(Linear_q(x))       # (B, S, 40, 128)

        # --- Text cross-attention ---
        k = RMSNorm(Linear_k(txt_ctx)) # (B, L_text, 40, 128)
        v = Linear_v(txt_ctx)           # (B, L_text, 40, 128)
        text_out = flash_attn(q, k, v)  # (B, S, 40, 128)

        # --- Image cross-attention (独立权重) ---
        k_img = RMSNorm(Linear_k_img(img_ctx))  # (B, 257, 40, 128)
        v_img = Linear_v_img(img_ctx)             # (B, 257, 40, 128)
        img_out = flash_attn(q, k_img, v_img)     # (B, S, 40, 128)

        # 两路结果直接相加
        out = text_out.flatten(2) + img_out.flatten(2)  # (B, S, 5120)
        return Linear_o(out)  # 共享 output projection
```

**注意**: `k_img, v_img` 在 LoRAEdit 训练的 exclude_linear_modules 中被排除。
这意味着 LoRA 不修改图像 cross-attention 的 K/V 投影，只修改文本 cross-attention 和 self-attention。

---

## 8. CLIP Image Encoder

```
首帧图像 (H, W, 3)
  → resize (224, 224)
  → CLIP normalize (ImageNet mean/std)
  → ViT-H/14 前 31 层 (skip last block)
  → (B, 257, 1280)   # 1 CLS + 256 patch tokens
  → MLPProj:
      LayerNorm(1280)
      → Linear(1280, 1280)
      → GELU
      → Linear(1280, 5120)
      → LayerNorm(5120)
  → (B, 257, 5120)
```

CLIP ViT-H/14 配置:
- image_size=224, patch_size=14 → 16×16=256 patches + 1 CLS = 257 tokens
- vision_dim=1280, vision_heads=16, vision_layers=32
- 使用前 31 层，不用 head projection

---

## 9. VAE (Wan2.1 VideoVAE)

| 参数 | 值 |
|------|-----|
| latent channels | 16 |
| 空间下采样 | 8x |
| 时间下采样 | 4x (首帧单独处理) |
| 归一化 | `(mu - mean) / std`，16 维 mean/std |

81 帧 480×832 → VAE → `(16, 21, 60, 104)`

---

## 10. Condition 通道组成 (I2V 36-channel)

```
Conv3d 输入 36 channels:
├── [0:16]  noisy latent x_t (要 denoise 的)
├── [16:20] binary mask (4 channels, 时间维度 packing)
│           首帧 mask=1 (已知), 其余 mask=0 (未知)
└── [20:36] VAE-encoded condition (pseudo video)
            首帧有真实图像 latent, 其余帧为零
```

Mask 的 4 通道来自时间维度 packing：VAE 将每 4 帧压缩为 1 个 latent frame，
对应的 mask 也需要 4 帧合 1，所以 1 channel mask 变成 4 channel。

---

## 11. I2V vs T2V 差异总结

| 方面 | T2V | I2V |
|------|-----|-----|
| `in_dim` | 16 | **36** |
| Cross-Attention | `WanT2VCrossAttention` | **`WanI2VCrossAttention`** |
| `k_img, v_img` | 无 | **有** (每个 block) |
| `img_emb` | 无 | **MLPProj(1280→5120)** |
| `clip_feature` | 无 | (B, 257, 1280) |
| `y` (condition) | 无 | (B, 20, F, H, W) |
| context 长度 | L_text | **257 + L_text** |

T2V 的 cross-attention 只有 text path；I2V 额外增加了独立的 image cross-attention path。

---

## 12. LoRAEdit 训练相关

LoRA 配置:
- rank: 16
- `exclude_linear_modules = ["k_img", "v_img"]` — 不修改图像 cross-attention 的 K/V
- 训练时 `blocks_to_swap = 24` — 24/40 blocks offload 到 CPU 节省 VRAM

LoRA 作用于: self-attention 的 Q/K/V/O、text cross-attention 的 Q/K/V/O、FFN。
**不作用于**: image cross-attention 的 K_img/V_img（保持首帧语义不变）。
