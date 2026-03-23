# Project: LoRAEdit Experiments

## Research Goal

**目标**：发表顶会论文。研究 LoRAEdit 在什么 case 下失效，并提出改进方案使其在这些 case 下也能 work。
- 不是"换一个方法"，而是**改进 LoRAEdit 本身**
- 已发现失效 case：小物体添加（earring），LoRAEdit 擅长属性变换但不擅长物体添加
- 论文 3.3.2 节的 reference frame 机制是作者给出的最后 robustness 手段

## Workflow Rules

- **execute.txt**: 所有可执行命令写到项目根目录的 `execute.txt`，新命令 **append** 到末尾，不要删除旧命令
- **LoRAEdit 源代码**: 不修改任何原始文件。只新增独立的工具脚本
- **output.txt**: 用户贴的运行输出/报错日志

## Key Paths

- LoRAEdit 代码: `/home/users/ntu/song0304/code/LoRAEdit/`
- Wan2.1-I2V 模型: `/scratch/users/ntu/song0304/checkpoints/Wan2.1-I2V-14B-480P`
- SAM2 模型: `/home/users/ntu/song0304/code/LoRAEdit/models_sam/sam2_hiera_large.pt`
- 实验数据: `/scratch/users/ntu/song0304/loraedit_exp/`
- 日志: `/scratch/users/ntu/song0304/loraedit_exp/logs/`
- PBS 脚本: `/home/users/ntu/song0304/code/LoRAEdit/scripts/`

## Custom Tools (新增，非 LoRAEdit 原始代码)

- `preprocess_cli.py` — CLI 预处理（替代 Gradio UI），支持 extract + SAM2 process
- `tools/edit_first_frame.py` — 首帧颜色编辑工具
- `inference_lowmem.py` — 低内存 inference（绕过 ModelManager 的 OOM 问题，DiT 逐 shard 加载）

## Lessons — 不允许再犯的错误模式

1. **严格执行指令，不要自作主张重新解释**
   - 用户说"裁剪视频到81帧"，就生成一个新的mp4文件，不是"提取帧然后跳过这一步"
   - 用户的指令是字面意思，不要用自己的理解替换用户的要求
   - 如果指令不明确，先问，不要擅自决定

2. **公式/逻辑不能套用，必须分场景验证**
   - 写完任何计算逻辑后，用具体数值代入验证结果是否合理
   - 例：连续取帧 fps 应等于原始 fps，不能套用均匀采样的公式
   - 不同 code path（consecutive vs uniform）需要独立推理，不能共用一个公式然后"差不多就行"

3. **输出必须和输入保持一致性**
   - fps、分辨率、帧数等参数，默认行为是和输入保持一致，除非用户明确要求改变
   - 不要引入不必要的变换或"智能计算"，简单直接最好

4. **不要硬编码魔法数字**
   - 任何写死的数值（fps=30, fps=5 等）都是潜在 bug
   - 应该从源数据读取，或者让用户指定

5. **修复必须一次性覆盖所有受影响的数据**
   - 修了 V7_hair 就必须同时修 V7_beard，不能漏
   - 修复代码后，立即检查所有使用该逻辑的已有数据并全部更新

6. **所有可执行命令必须写到 execute.txt，不要只贴在聊天里**
   - 这是最基本的 workflow rule，没有例外
   - 每次给用户命令时，先写 execute.txt，再告诉用户去执行

7. **PBS 脚本写好后直接 qsub 提交，不要等用户确认**
   - 脚本生成完毕 → 立即 `qsub` 提交 → 写入 execute.txt 记录
   - 不要停下来问"要不要提交"，直接执行

## Inference 规范

- 每次 inference 必须生成**两个对比视频**：
  1. **GT vs Original** — GT (左) | Original (右)，验证重建质量
  2. **Original vs Edited** — Original (左) | Edited (右)，观察编辑效果
- 使用 ffmpeg hstack + drawtext 标注
- 输出到独立的 results/ 目录

## 代码规范

- **不要用 cv2 (OpenCV) 写视频**，cv2.VideoWriter 压缩质量差、兼容性差。用其他库替代。

## Cluster Info

- PBS/Torque 作业调度
- GPU: A100-SXM4-40GB
- Conda 环境: `loraedit` (Python 3.12, PyTorch 2.6+cu124)
- 计算节点可联网
