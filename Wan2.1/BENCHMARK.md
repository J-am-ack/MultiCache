# 在 Wan 2.1 上比较缓存方法

使用小组已经跑通的 **Wan2.1-T2V-1.3B** 环境。先跑快速对照，可以直接回答“在原版 SenCache 的重算步上加 D，到底再省多少时间”。三个设置使用同一句提示词、同一个初始噪声。

在本目录执行，下面的两个绝对路径换成自己的：

```bash
export PYTHONPATH=/path/to/Wan2.1${PYTHONPATH:+:$PYTHONPATH}
python bench_video.py --checkpoint /path/to/Wan2.1-T2V-1.3B \
  --output ../outputs/wan-quick --quick
```

输出目录有三段 MP4 和 `video_runs.json`：每步全算、原版 SenCache、原版选步加 D。看每条记录的 `seconds`、`branches` 和 `trace`，分别是完整生成耗时、实际执行的主干分支数和逐步决策。视频默认 832×480、17 帧，UniPC 20 步，引导强度 6，shift 8。默认提示词是组里冲浪板上的白猫。

## 一次跑全

```bash
python bench_video.py --checkpoint /path/to/Wan2.1-T2V-1.3B \
  --prompts my_prompts.txt --seeds 42 137 2026 --output ../outputs/wan-full
```

提示词文件每行一句，也支持一行一个带 `prompt` 字段的 JSON 对象。模型在进程内加载一次。先用两句校准提示词分别跑全算，再逐个试跳第 1 至 18 步（从 0 编号），共 38 段校准视频，保存分数、各步影响和累计采样时间到 `video_profile.json`。后续在同一输出目录复用这份校准。更改权重或采样配置时另建输出目录。

| 设置 | 视频里实际做了什么 |
|---|---|
| native20、native14 | 调用原生 Wan，比较 20 步与直接减到 14 步。 |
| sencache | 小组原版前向，自适应选步并复用主干残差。 |
| fixed、global、A、B | 同为 20 步，每连续三个候选步选一个复用，总共复用 6 步。global 使用仓库预计算的敏感度与整体潜变量变化，A 加入局部变化最大的八分之一时空块，B 给分数乘上从校准视频测得的步权重。 |
| C、C_guard、D | 使用和 fixed 相同的刷新表；C 用前两次的真实主干残差预测，C_guard 遇到大改动就退回缓存，D 在双算候选步尝试省掉无条件分支。 |
| sen_C、sen_D | 按提示词和种子分别记录原版 SenCache 的刷新表，再回放给带回退的 C 或 D。可以直接看它们给队友现有实现带来的增益。 |

各视频保存相对同种子 20 步全算的 PSNR。另一个指标 `temporal_rmse_255` 比较相邻帧变化与参考视频有多大差别，按 0–255 像素尺度报告。它可帮助定位运动变化大的片段。若已有 Hugging Face 格式的 CLIP ViT-B/32，可单独对每段视频均匀取 5 帧，计算与整句提示词的平均余弦相似度：

```bash
python score_video_text.py ../outputs/wan-full/video_runs.json /path/to/clip-vit-base-patch32
```

## 适配队友已有的流程

单独启用 D 可在原先的 `sencache.py` 命令末尾加 `--cfg_cache --cfg_tolerance 0.35 --cfg_max_age 3`。各档位的决策在 `video_policy.py`，三步窗口的预算分配与校准在 `budget_policy.py`。修改某一策略后继续沿用 `bench_video.py` 的提示词、计时和结果格式即可。

本地测试入口为 `python test_cfg_cache.py` 和 `python test_video_bench.py`。前者检查缓存跨样本清空，后者在缩小的真实 Wan 网络上逐次统计主干执行并核对每种设置。预训练权重的耗时和视频质量由上述视频命令测量。
