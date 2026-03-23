# Iterative Video Editing via Region-Aware LoRA Composition: Research Plan (v3)

## Changelog from v2
- 问题定义重新框架化：从"LoRA-Edit sequential editing是否fail"升级为"iterative video editing的quality preservation"
- Phase 1大幅强化：增加定量metrics体系、扩大测试规模(50+ video-edit pairs)、增加cross-method对比
- Phase 2重写：基于Wan2.1 full 3D attention架构事实，放弃spatial/temporal LoRA分离假设，改为Region-Aware LoRA Routing + Orthogonal LoRA Training双组件方案
- 实验设计按顶会标准重构：Main Table + Scaling Curve + Ablation + User Study

---

## 0. 核心目标

**研究问题**：在iterative video editing场景中（用户分步对同一视频进行多轮局部编辑），基于model adaptation的编辑方法是否会产生systematic的质量退化？如果是，如何在不改变base model架构的前提下解决？

**为什么这个问题重要**：
- 专业视频编辑工作流本质上是iterative的——后续编辑决策依赖前序编辑的视觉结果（改了肤色发现衣服不搭，才决定改衣服），不可能提前规划所有编辑
- 现有video editing方法（training-free和LoRA-based）都只考虑single-pass editing
- 这不是某个方法的bug，而是一个under-explored problem setting

**关键架构事实**：Wan2.1使用Full 3D Attention（所有spatial+temporal tokens在同一个attention层中交互），不存在独立的spatial/temporal attention层。方案设计必须基于此事实。

---

## Phase 1: Systematic Problem Analysis（预计3-4周）

### 目标
为论文的Section 3（Problem Analysis）提供充分的empirical evidence，证明iterative editing degradation是real的、systematic的、cross-method的。

### 1.1 环境搭建

**代码库**: diffusion-pipe (LoRA-Edit基于此)
**模型**: Wan2.1-I2V 480P（主要）
**硬件**: 单卡A40 (45GB)
**First-frame editing工具**: ACE++ 或任意image editing工具

```
Step 1: Clone diffusion-pipe, 配置环境
Step 2: 下载Wan2.1-I2V 480P权重
Step 3: 复现原文单次编辑结果（确认baseline正常）
Step 4: 配置evaluation metrics pipeline
```

### 1.2 Degradation Metrics体系

定义四个核心指标构成degradation profile：

| Metric | 定义 | 计算方法 |
|--------|------|---------|
| **Edit Fidelity (EF)** | 当轮编辑的实现程度 | edited region crop的CLIP similarity vs target text description |
| **Edit Preservation (EP)** | 前序编辑在后续轮次中的保持程度 | 对Round k编辑的region做crop，比较Round k输出 vs Round N最终输出的CLIP similarity |
| **Background Integrity (BI)** | 未编辑区域的保持程度 | 未编辑区域的SSIM + LPIPS (original vs final) |
| **Temporal Consistency (TC)** | 帧间一致性 | 相邻帧warping error (RAFT optical flow) + 相邻帧CLIP similarity |

**关键输出**：随编辑轮数增加，四个指标的degradation curves。

### 1.3 测试Protocol

**规模**：50+ video-edit pairs（不是50个视频，而是50个"视频+编辑序列"组合）

**视频来源**（不需要自建数据集）：
- DAVIS dataset（~30个视频，自带mask标注）
- Pexels/YouTube free-license视频（~20个）
- I2VEdit test set（如果可获取）

**系统性覆盖四个维度**：

| 维度 | 级别 | 数量分配 |
|------|------|---------|
| Motion magnitude | 静态 / 中等 / 大幅运动 | 各~17个 |
| Edit type | color change / texture change / object replacement | 各~17个 |
| Spatial relationship | 远距离 / 近距离 / overlapping | 各~17个 |
| 编辑轮数 | 2轮 / 3轮 / 4轮 | 各~17个 |

注意：这些维度是交叉的，每个video-edit pair同时属于多个维度的某个级别。

**Mask来源**：DAVIS自带标注 + SAM自动生成 + 少量手动标注

### 1.4 实验设计

#### 实验执行顺序

```
Exp 0: Baseline复现（单次编辑，验证环境）
  ↓
Exp 1: LoRA-Edit sequential degradation measurement（核心）
  ↓
Exp 2: Diagnostic实验（pixel-space vs weight-space）
  ↓
Exp 3: Cross-method验证（证明是general problem）
  ↓
  Decision Gate
```

---

#### Experiment 0: Baseline复现（必做）

对5个视频各做一次标准LoRA-Edit单次编辑，确认和原文demo质量相当。

**如果Exp 0效果明显差于原文 → 停下来排查环境，不继续。**

---

#### Experiment 1: Systematic Degradation Measurement（核心）

**目的**：定量测量LoRA-Edit在iterative editing下的degradation程度和pattern。

Protocol：
```
对50+ video-edit pairs执行：
  Round 1: 编辑Region A → 记录EF_1, BI_1, TC_1
  Round 2: 基于Round 1输出编辑Region B → 记录EF_2, EP_1, BI_2, TC_2
  Round 3 (部分): 继续编辑Region C → 记录EF_3, EP_1, EP_2, BI_3, TC_3
  Round 4 (部分): 继续编辑Region D → 完整degradation profile
```

两种sequential strategy都测：
- **方案a (from-scratch)**：每轮训练全新LoRA
- **方案b (continue-training)**：在前轮LoRA基础上继续训练

**关键输出**：
1. 四个metric的degradation curves（x轴=轮数，y轴=metric值）
2. 按motion magnitude / edit type / spatial relationship分组的degradation对比
3. 统计显著性检验（paired t-test or Wilcoxon）

---

#### Experiment 2: Diagnostic（pixel-space vs weight-space）

**目的**：区分degradation来源，直接影响Phase 2方案设计。

三种方案对比：

| 方案 | 描述 | 诊断作用 |
|------|------|---------|
| 方案a/b | 串行编辑（同Exp 1） | 包含pixel-space + weight-space两种degradation |
| 方案c: LoRA merge | 独立训练LoRA_1和LoRA_2，推理时merge权重 | 只包含weight-space interference |
| 方案d: Oracle single-pass | 用combined mask一次性编辑所有region | Upper bound参考 |

方案c的merge alpha sweep: (1.0,1.0), (0.7,0.7), (0.5,0.5), (1.0,0.5)

**诊断矩阵**：

| 方案a/b | 方案c | 方案d | 诊断结论 |
|---------|-------|-------|---------|
| fail | fail | pass | weight-space interference为主 → Phase 2有价值 ✅ |
| fail | pass | pass | pixel-space cascading为主 → 需重新评估 ⚠️ |
| pass | pass | pass | 问题不存在 → 换方向 ❌ |

**在15-20个representative video-edit pairs上执行**（从Exp 1中选degradation明显的cases）。

---

#### Experiment 3: Cross-method Verification（证明general problem）

**目的**：证明iterative editing degradation不只是LoRA-Edit的问题。

在10-15个video-edit pairs上测试：
- **LoRA-Edit**（主要对象）
- **Training-free方法**：VACE或TokenFlow的sequential application
- **Commercial API**：Kling（如果budget允许）

**预期**：所有方法在multi-pass下都有退化，但LoRA-based方法退化更严重（因为weight-space interference叠加pixel-space cascading）。

如果只有LoRA-Edit退化 → problem scope需要缩小，但仍然有研究价值（LoRA-based video editing是主流方向）。

---

#### ─── Decision Gate ───

**Phase 1完成后判断：**

| 结果 | 行动 |
|------|------|
| Degradation显著 + weight-space interference confirmed + cross-method验证通过 | 🔥 最佳情况 → Phase 2 |
| Degradation显著 + weight-space interference confirmed + 只有LoRA-Edit受影响 | ✅ 可以做 → Phase 2，但scope限定在LoRA-based methods |
| Degradation显著 + 主要是pixel-space cascading | ⚠️ 重新评估 → 可能转向video quality restoration方向 |
| Degradation不显著 | ❌ 换方向 → 备选方向 |

---

## Phase 2: Method Design（仅在Decision Gate通过后执行）

### 核心Method：Region-Conditioned LoRA Composition with Orthogonal Constraint

**设计原则**：不改base model架构，不引入新模块，只改LoRA的训练和推理方式。两个组件分别解决两类问题。

---

### Component 1: Orthogonal LoRA Training（训练时）

**解决的问题**：多个LoRA在weight space中互相干扰（Exp 2方案c fail的原因）。

**方法**：
```
训练LoRA_1：标准LoRA-Edit流程
训练LoRA_2：在标准loss基础上，增加正交性regularization

L_total = L_editing + λ · L_orth
L_orth = Σ_l ||B_2^(l)^T · B_1^(l)||_F^2 + ||A_2^(l)^T · A_1^(l)||_F^2

其中 A_k^(l), B_k^(l) 是第k个LoRA在第l层的低秩矩阵
```

**训练LoRA_k（k≥3）时**：约束对所有前序LoRA_1...LoRA_{k-1}的正交性。

**为什么这在full 3D attention上可行**：正交约束作用在LoRA的参数空间，不依赖attention层是否分离。无论是full 3D还是separated spatial/temporal，只要LoRA修改的是同一组weight matrices，正交约束都成立。

**超参数**：
- λ: 正交loss权重（需要ablation sweep）
- 是否对所有层施加约束 vs 只对self-attention层

---

### Component 2: Region-Aware LoRA Routing（推理时）

**解决的问题**：即使weight space不冲突，全局apply merged LoRA也会导致非目标区域被不必要地影响。

**方法**：
```
推理时，对每个latent token (t, h, w)：
1. 根据mask确定其属于哪个编辑region（mask需下采样到latent space）
2. 动态选择LoRA权重：
   - Region A的token: W_base + LoRA_1
   - Region B的token: W_base + LoRA_2
   - 未编辑区域: W_base（不加任何LoRA）
   - 边界区域: W_base + α(h,w)·LoRA_1 + β(h,w)·LoRA_2（soft blending）
```

**实现细节**：
- Mask下采样：spatial 8x, temporal 4x（匹配Wan-VAE的compression ratio）
- Routing粒度：在attention的QKV projection和FFN层做routing
- 边界blending：用Gaussian blur对binary mask做soft化，blur kernel size是超参数
- **重要**：在full 3D attention中，token之间有全局交互。Routing只控制LoRA的additive contribution，不阻断token间的attention flow。这意味着编辑效果仍然可以通过attention传播，但weight-level interference被消除。

**训练-free nature**：Component 2不需要任何额外训练，只改推理时的forward pass。

---

### 两个Component的关系

| Component | 作用于 | 解决的问题 | 是否需要额外训练 |
|-----------|-------|-----------|--------------|
| Orthogonal LoRA | 训练时 | Weight-space interference | 是（加regularization） |
| Region Routing | 推理时 | 非目标区域的不必要影响 | 否（training-free） |

两者互补：Orthogonal保证LoRA参数不冲突，Routing保证每个spatial区域只受相关LoRA影响。

---

### 关于temporal consistency

在full 3D attention架构中，temporal consistency主要由attention中的temporal token交互维护。我们的方法不破坏这个机制（routing只改LoRA contribution，不改attention结构），因此temporal consistency应当被自然保持。

如果实验发现temporal consistency仍有退化，可以增加一个lightweight的后处理：
- 在denoising过程中，对相邻帧同一spatial位置的latent feature做temporal smoothing
- 这是一个可选的refinement，不是核心contribution

---

## Phase 3: Experiments（论文Sections 4-5）

### 3.1 实验设置

**Test Protocol**：
- 50+ video-edit pairs（从Phase 1继承）
- 每个pair定义2-4轮sequential editing操作
- 所有视频、mask、editing instructions公开发布

**Metrics**：Phase 1定义的四个指标（EF, EP, BI, TC）

**Baselines**：
| Baseline | 描述 |
|----------|------|
| Naive Sequential | LoRA-Edit方案a，每轮from-scratch训练 |
| Continue Training | LoRA-Edit方案b，继续训练 |
| LoRA Merge | 方案c，独立训练后merge权重 |
| Single-pass Combined | 用combined mask一次性编辑（oracle upper bound） |
| VACE (if available) | Training-free baseline |

### 3.2 必做实验

**a) Main Results Table（Table 1）**

| Method | 2-round |  |  |  | 3-round |  |  |  | 4-round |  |  |  |
|--------|---------|--|--|--|---------|--|--|--|---------|--|--|--|
|        | EF↑ | EP↑ | BI↑ | TC↑ | EF↑ | EP↑ | BI↑ | TC↑ | EF↑ | EP↑ | BI↑ | TC↑ |
| Naive Sequential | | | | | | | | | | | | |
| Continue Training | | | | | | | | | | | | |
| LoRA Merge | | | | | | | | | | | | |
| Ours (Routing only) | | | | | | | | | | | | |
| Ours (Ortho only) | | | | | | | | | | | | |
| **Ours (Full)** | | | | | | | | | | | | |
| Single-pass (oracle) | | | | | | | | | | | | |

**b) Scaling Analysis（Figure）**

- X轴：编辑轮数（1-4）
- Y轴：EP（Edit Preservation，最核心指标）
- 多条曲线：各method
- 预期：我们的方法degradation曲线最平缓

**c) Ablation Study（Table 2）**

| Configuration | EP↑ | BI↑ | TC↑ |
|---------------|-----|-----|-----|
| Full method | | | |
| w/o Orthogonal constraint | | | |
| w/o Region routing | | | |
| w/o Boundary blending | | | |
| w/o Both (= LoRA Merge) | | | |

**d) User Study**

- 35+ participants
- Pairwise comparison: Ours vs each baseline
- 两个问题：(1) 哪个编辑质量更好？(2) 哪个temporal consistency更好？
- 报告preference rate + statistical significance

**e) Qualitative Comparison（Figure）**

- 每种failure mode一个visual example
- 展示：original → Round 1 → Round 2 → Round 3 的逐轮对比
- 对比各method在同一case上的表现

### 3.3 加分实验

**f) Hyperparameter Analysis**
- λ（正交loss权重）对EP的影响
- Boundary blending kernel size的影响
- LoRA rank对正交空间容量的影响

**g) Motion-Conditioned Analysis**
- 按motion magnitude分组，分析各方法的degradation差异
- 大运动场景下我们的方法优势应该更明显

**h) Real-world Editing Session**
- 模拟一个完整的iterative editing workflow（4-5轮）
- 展示用户如何基于前序结果做后续决策

---

## Phase 4: 备选方向（如果Phase 1验证失败）

### 备选A: LoRA-Edit Failure Case Analysis
系统测试LoRA-Edit在不同conditions下的表现，分析failure modes。Analysis paper。

### 备选B: Motion Transfer via Mask-LoRA
用LoRA-Edit框架做motion-only editing（保持appearance，改变运动）。

### 备选C: 从Phase 1的大规模测试中发现新问题
Phase 1的50+ video-edit pairs测试本身就会产生大量观察，可能发现其他有研究价值的问题。

---

## Timeline

```
Week 1:     环境搭建 + Baseline复现（Exp 0）+ Metrics pipeline搭建
Week 2-3:   Exp 1: 50+ video-edit pairs的systematic degradation measurement
Week 3:     Exp 2: Diagnostic实验（方案c/d）
Week 4:     Exp 3: Cross-method验证 + Decision Gate
            ┌─ 通过 → Week 5-7: Phase 2实现 + 实验
            └─ 未通过 → Week 5-6: 备选方向
Week 5-6:   实现Orthogonal LoRA Training + Region Routing
Week 7-8:   完整实验（Main Table + Ablation + Scaling）
Week 9:     User Study + 写作
Week 10:    投稿准备
```

---

## 每个实验的记录模板

```
实验编号: Exp X - Video VY - Edit Sequence [描述]
日期:
视频来源: [URL/dataset/ID]
分辨率: [832x480 / 480x832]
帧数: 49

=== Round k ===
  编辑内容: [如：头发颜色 黑→金]
  Mask来源: [DAVIS标注 / SAM / 手动]
  Mask overlap with previous rounds: [无/部分/完全]
  训练方案: [a: from scratch / b: continue training]
  训练步数: 100
  训练时间: X min
  显存占用: X GB

=== Quantitative Metrics ===
  EF (Edit Fidelity): [CLIP score]
  EP (Edit Preservation): [per-round preservation scores]
  BI (Background Integrity): [SSIM / LPIPS]
  TC (Temporal Consistency): [warping error / frame CLIP sim]

=== Qualitative Observation ===
  编辑保持情况: [完好/轻微退化/严重退化/完全丢失]
  边界artifact: [无/轻微/严重]
  Temporal flickering: [无/轻微/严重]

截图/视频: [保存路径]
关键发现: [一句话总结]
```

---

## 关键原则

1. **Phase 1是论文的核心**。Problem analysis的quality直接决定投稿成败。把最多精力放在这里。
2. **定量优先**。所有observation都要有metric支撑，肉眼观察只作为补充。
3. **Cross-method验证**。证明这是general problem，不是单一方法的bug。
4. **Method要well-motivated**。每个component直接对应一个observed failure mode，有ablation支撑。
5. **不over-claim**。如果只在LoRA-based方法上有效，就限定scope，不要声称解决了all video editing的问题。
6. **记录所有negative results**。Phase 1如果发现问题不存在，这本身也是有价值的finding。
7. **保持诚实**。问题不存在就及时止损换方向。