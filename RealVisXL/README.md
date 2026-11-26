# RealVisXL 缓存评测

同一句话、同一个种子，把“减少采样步数”和“保留步数、少算一些分支”放在同一张表上看。模型是 RealVisXL V4.0，分辨率 512×512，Euler 采样，CFG 6。每种设置跑 12 条提示词 × 3 个种子，共 36 张图。

[结果与样图](results/REPORT.md) · [完整提示词](cases.json) · [Wan 视频复验](../Wan2.1/BENCHMARK.md)

## 各设置做了什么

| 代码中的名字 | 做法 |
|---|---|
| native20 / native16 / native14 | 直接运行 diffusers 原生管线，分别采样 20、16、14 步。两条 CFG 分支合成一个 batch。 |
| serial20 | 同样 20 步，每步把两条分支拆开算。可以看出串行与合批本身的速度差。 |
| fixed | 保留 20 步，在编号 2、5、8、11、14、17 的步骤沿用最近一次完整预测。其余步骤按原生方式合批。 |
| global | 将步骤 1–18 每三个分成一组，每组选一步复用。观察当前潜变量与最近一次真算输入的整体 RMS 差值。变化低于校准门限就复用，否则留到组末。 |
| A | 在 global 的分数中加入变化最大的八分之一区块。每块 8×8 个潜空间位置，最终分数取局部与整体 RMS 的平均。同样每组选一步复用。 |
| B | 先用独立提示词试出哪些步骤一旦跳错，对最终成图影响更大，再用这个权重调整 global 的分数。A、B、global 与 fixed 每张图都真算 28 个分支。 |
| C | 沿用 fixed 的跳步位置。从最近两次真算的输出按步号做一阶外推，以估计当前输出。 |
| C_guard | 历史两次真算相隔超过两步，或外推增量的范数超过上次输出范数的 15%，就用上次输出。 |
| D | 保留 fixed 的跳步表。在重算步上，如果历史 CFG 差值漂移 ≤ 0.35，且最后一次双算距今 ≤ 3 步，就只算条件分支。其余重算步保留合批。 |

编号从 0 开始。这里缓存的是 U-Net 的输出。小组的视频实现缓存 DiT 主干残差，并已有 Wan 自身的敏感度权重。A–D 是对分工图中几个想法的具体实现，四种策略的来龙去脉见[原始分工](../thinking/1124_4.png)。

**B 怎么校准。** 自行车和小提琴两条提示词各用种子 31。每条先跑一次全算，再轮流只跳第 1–18 步中的一步，其余步骤照常执行。相同种子让两条轨迹在干预前一致。以最后两张图的像素均方差衡量这一步的影响。两句取平均，除以 18 步的中位数后开方，得到逐步权重。每个三步窗口的门限取该评分在两条全算轨迹上 6 个观测值的下三分位数。各方法的门限在测试前锁定。

**D 怎么估计。** 保存上一次真实双算的差值 Δ = 条件输出 − 无条件输出。下一次单算时，用当前条件输出减去 Δ 来估计无条件分支。风险分数取最近两次真实 Δ 的差范数，乘以 CFG 强度减一，再除以最近一次真实引导输出的范数。最初两次重算用于建立历史。

## 复现

准备 CUDA 版 PyTorch 和对应的 torchvision。在本次实测环境中，Python 为 3.12.7，PyTorch 为 2.8.0+cu129，torchvision 为 0.23.0，diffusers 为 0.35.1，transformers 为 4.54.1，accelerate 为 1.9.0。评测还用到 Pillow、NumPy、SciPy 和 scikit-image。下面从仓库根目录执行。MODEL 等大写名称代表你本地的模型目录。

```bash
python RealVisXL/bench.py MODEL outputs/realvis
python RealVisXL/evaluate.py outputs/realvis GROUNDING_DINO outputs/report --clip CLIP
python RealVisXL/audit_steps.py MODEL outputs/realvis/calibration.json outputs/report/step_audit.json
python RealVisXL/detail_review.py outputs/realvis outputs/blind_sheets
```

- MODEL 是完整的 [RealVisXL V4.0 diffusers 目录](https://huggingface.co/SG161222/RealVisXL_V4.0)。管线使用 FP16、按组件卸载和 VAE 平铺。
- GROUNDING_DINO 是 [Grounding DINO base](https://huggingface.co/IDEA-Research/grounding-dino-base)。它按每条提示词中事先指定的主体短语打分。
- CLIP 使用 [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32)，计算完整英文提示词与图像的余弦相似度。支持 Hugging Face 权重，也支持同一目录中存放官方 ViT-B-32.pt、分词器和预处理配置。
- 生成脚本按图存盘，用同一目录重跑可接着做。改模型、提示词、种子或策略时使用新的输出目录。原始 PNG 在 `outputs/realvis/images/`，每张图的耗时、峰值和逐步动作在 `runs.json`。
- `detail_review.py` 会把指定提示词的样图打乱、隐藏方法名。按脚本中的三条规则为每个编号填 0 或 1，再通过 `--labels labels.json --report detail.json` 导出结果。

## 数字怎么读

每个提示词和种子共用初始噪声，各设置以可复现的随机顺序运行。预热后计时，包含文本编码、去噪、卸载和 VAE 解码，另外记录 PyTorch 分配器报告的 GPU 峰值。加载权重、写 PNG 和后续打分单列。batch 中同时算两条分支记为 2 个分支，耗时按真实时钟记录。

SSIM 和 PSNR 比较同种子原生 20 步的参考图。PSNR 完全相同的图记为 null，均值只统计有限值；`perfect_pairs` 单独给出相同图数量。主体分数是 Grounding DINO 在候选框中，对目标短语各词元取平均后的最高分。CLIP 衡量整句话与图像的匹配，数量、位置和拼字另看遮住方法名的样图。时间加速比取两组平均耗时之比；95% 区间对 12 条提示词作 5,000 次有放回抽样，同一句的三个种子一起抽取。

额外的逐步检查让模型在每个候选步都真算，比较“如果此时用上一步输出”与真实引导输出之间的相对 L2 误差。每条测试提示词取种子 42，报告 18 个候选步里误差最大的 6 步，三种分数各找对几个。这个检查看单步预测误差，B 的校准看误差传到最后对成图的影响。
