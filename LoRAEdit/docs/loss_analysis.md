# LoRAEdit Loss 精确分析 — Stage 1 vs Stage 2

## 前置知识：Flow Matching

LoRAEdit 基于 Wan2.1，使用 **Flow Matching**（不是传统 DDPM 的 ε-prediction）。

核心思想：给定一个干净视频 latent（x₁）和一个纯噪声（x₀），两者之间存在一条"路径"。在路径上随机取一个中间点 x_t = (1-t)·x₁ + t·x₀。t=0 时完全干净，t=1 时完全是噪声。

模型的任务：看到中间点 x_t，预测从干净到噪声的方向向量（即 x₀ - x₁），称为 velocity field。

训练 loss = 模型预测的 velocity 与真实 velocity 之间的 MSE。

---

## 首先要区分的两个 mask

系统里存在两个名字都叫"mask"但**作用完全不同**的东西。非常容易混淆。

### Conditioning Mask（我们一直在用的那个）

来源是 SAM2 跟踪生成的 source_masks/，被塞进训练视频的最后 81 帧（frames 163-243），最终变成模型输入 36 个 channel 中的 4 个（channels 16-19）。

它的作用是**告诉模型哪个空间区域是编辑区**。模型看到这个 mask 后知道："白色区域（耳环区域）的内容你需要自己生成，黑色区域（背景）的内容我在 condition 里已经给你了。"

打个比方：这是一份**考试大纲**，告诉学生"这些区域会考，其他的不考"。

### Loss Mask（我们没有用的那个）

来源是 dataset.toml 里的 `mask_path` 配置项。如果配了，它会逐元素乘以 MSE loss，让某些区域的预测误差对 loss 贡献更大。

它的作用是**控制训练时哪些区域的误差更"值钱"**。比如可以让前景区域的 loss 权重是背景的 16 倍，这样模型的梯度更新会更关注前景。

打个比方：这是**评分标准**，决定每道题占多少分。

**关键：我们的配置里没有设置 loss mask，也就是说所有像素的 loss 权重完全相同。** 这是 LoRAEdit 的默认行为。

换句话说：模型有"考试大纲"（知道哪里是编辑区），但"评分标准"对所有区域一视同仁（背景答对和前景答对，得分一样）。

---

## Loss 具体算的是什么

### 输入是什么

原始视频（81 帧, 480×832）经过 VAE 编码后，被压缩成一个 latent tensor：
- 空间上：480×832 → 60×104（缩小 8 倍）
- 时间上：81 帧 → 21 帧（缩小 4 倍）
- 通道数：RGB 3 通道 → VAE latent 16 通道

最终 shape：**(1, 16, 21, 60, 104)**，总共约 210 万个数值。

### Loss 怎么算

1. 对这个 latent 加随机噪声，得到 x_t
2. 把 x_t 连同 condition 信号（灰化伪视频、空间 mask、CLIP 特征、文本）喂给 DiT
3. DiT 输出一个预测的 velocity，shape 也是 (1, 16, 21, 60, 104)
4. 与真实 velocity 逐元素算平方差，得到 210 万个误差值
5. **对这 210 万个误差取平均**，得到一个标量 loss

就是这么简单。**每一个 latent 元素对 loss 的贡献完全相等。** 不区分前景/背景，不区分哪一帧，不区分哪个通道。

### 这意味着什么

以 earring case 为例，mask 覆盖约 7% 的面积。在 latent 空间里：
- 背景区域占 ~93% 的 latent 元素 → 贡献 ~93% 的 loss
- 耳环区域占 ~7% 的 latent 元素 → 贡献 ~7% 的 loss

**模型每次参数更新，93% 的力量在学"怎么重建背景"，只有 7% 在学"怎么处理耳环区域"。**

---

## Stage 1 和 Stage 2 的 Loss 到底有什么不同

### Stage 1：标准 LoRAEdit

训练视频的 target 部分（frames 82-162）= **完整的原始视频**，81 帧全都没有耳环。

模型需要做的事情：看到加噪的原始视频 latent，预测去噪方向。

这里有一个关键点：**flow target = noise - 原始视频 latent**。这个 target 里完全没有耳环的视觉信息（因为原始视频里根本就没有耳环）。

所以 Stage 1 学到的是：
- "给定 mask 和 condition，把整个视频恢复成原始内容"
- "mask 区域（耳环区域）恢复成原始的耳朵（没有耳环）"

推理时你塞一个带耳环的首帧进去，模型会想："我训练时 mask 区域的正确答案一直是没有耳环的耳朵，所以我还是生成没有耳环的耳朵吧。"

### Stage 2：Reference Frame

训练视频的 target 部分被修改了两帧：
- **Target frame 0** = 编辑后的首帧（带耳环）
- **Target frame 40** = Gemini 生成的中间帧（带耳环）
- **其余 79 帧** = 原始视频（无耳环）

这意味着 flow target 变了：
- Frame 0 和 Frame 40 的 flow target = noise - **带耳环的帧的 latent**
- 其余帧的 flow target = noise - **原始视频帧的 latent**

VAE 做 4 倍时间压缩后，81 帧变成 21 个 latent frame。带耳环信息的大约占 2 个 latent frame。

所以在 210 万个 loss 元素中：
- **约 9.5%**（2/21 latent frames）的 flow target 包含耳环的视觉信号
- **约 90.5%**（19/21 latent frames）的 flow target 是原始内容

### 核心差异总结

| | Stage 1 | Stage 2 |
|---|---|---|
| Flow target 中有耳环信号的比例 | **0%** | **~9.5%** |
| 模型能学到"mask 区域 = 耳环"吗？ | 不能，从未见过 | 能，target 中直接包含正确答案 |

---

## 实验验证：梯度强度不是瓶颈

直觉上你会觉得 earring 失败是因为"mask 太小 → 前景梯度信号弱 → 模型学不到"。但我们做了客观像素级验证，**否定了这个假设**。

### 验证方法

对比每组实验中 original inference 和 edited inference 的输出视频，逐帧逐像素计算 mask 区域和背景区域的平均像素差异（Mean Absolute Difference）。如果模型确实在 mask 区域"做了事"，那 edited 输出在 mask 区域应该和 original 输出有明显差异。

### 验证结果

| 条件 | Mask% | FG diff | BG diff | FG/BG ratio | Edit Persistence |
|------|-------|---------|---------|-------------|------------------|
| Hair (Run4 ep100) — **成功 case** | 11.6% | 14.41 | 2.34 | 6.2x | 1.59x |
| Earring original mask (Run4 ep100) | 5.3% | 12.36 | 2.21 | 5.6x | 1.69x |
| Earring expanded mask (+33%, ep100) | 7.0% | 14.44 | 2.32 | 6.2x | 1.70x |

**各指标含义：**
- **FG diff**：mask 区域内，original 和 edited 输出之间的平均像素差（frames 1+，即模型生成的帧）。越大 = 模型在 mask 区域的行为因 edit 而改变越多。
- **BG diff**：背景区域的同一指标。应该很小（背景不应因 edit 而改变）。
- **FG/BG ratio**：前景差异 / 背景差异。越大 = 模型越精准地只在 mask 区域做修改。
- **Edit Persistence**：FG diff (frames 1+) / FG diff (frame 0)。>1 表示模型在后续帧放大了首帧的编辑信号。

### 这个表格说明了什么

1. **Earring expanded 的所有指标与 hair 完全一致**：FG diff 14.44 vs 14.41，FG/BG ratio 同为 6.2x，persistence 甚至更高（1.70x vs 1.59x）。

2. **即使原始 earring（mask 仅 5.3%），指标也相差不大**：FG/BG ratio 5.6x，persistence 1.69x。梯度信号在 5.3% mask 下已经足够强。

3. **结论**：模型在 mask 区域做了与 hair 同等幅度的修改。但这些修改没有产生一个视觉上可辨识的耳环——模型改了那个区域的像素，但改出来的不是耳环。

**这意味着问题不在"模型没注意到 mask 区域"（它注意到了，而且反应强度和 hair 一样），而在"模型不知道该生成什么内容"。** 训练 target 中从未出现过耳环，所以模型只学到了"mask 区域需要做某种变化"，但变化的方向是随机的，不是朝着"耳环"的方向。

---

## Stage 2 的意义

Stage 2 不是为了加强梯度信号（那已经足够了），而是为了解决一个更根本的问题：**让模型在训练中见到"正确答案"。**

Stage 1 的 flow target 里没有任何耳环信息。模型学到的"mask 区域要改"只是统计上的副作用——因为 condition 中 mask 区域被灰化了，target 中是原始内容，模型学会了"灰色区域要恢复成原始内容"。推理时塞入一个带耳环的首帧，模型依然"恢复"成它认为正确的原始内容。

Stage 2 直接在 flow target 的两个帧中放入了带耳环的图像。模型现在能在 MSE loss 中看到：这两帧 mask 区域的"正确答案"是耳环，而不是空耳朵。这给了模型一个明确的学习信号：**"mask 区域 = 耳环"。**
