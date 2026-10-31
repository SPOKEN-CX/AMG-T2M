# 开发入口索引

本页用于快速定位源码，不提供完整运行顺序或一键复现流程。训练参数、实验目录和恢复检查以现有入口脚本为准；数据集、模型权重与缓存的来源和目录见[数据集说明](datasets.md)、[数据与模型](data-and-models.md)及[`resources/manifest.json`](../resources/manifest.json)。当前实现范围和未完成部分见[实现范围](implementation-status.md)。

| 模块 | 主要入口 | 用途 |
|---|---|---|
| 基础 VQ | `tools/train_vq.py` | 训练基础动作 VQ 模型 |
| AMG VQ | `tools/train_amg_vq.py`、`tools/run_amg_vq_*.sh` | 训练带 TMR 语义配对的 VQ；shell 脚本记录实验参数 |
| GPT | `tools/train_gpt.py`、`tools/run_amg_gpt*.sh` | 在选定 VQ token 表示上训练 GPT |
| 长度模块 | `tools/train_length_est.py`、`tools/run_amg_lengths.sh` | 训练动作长度估计器 |
| 数据准备 | `preprocess/prepare_amg_tmr.py`、`prepare_amg_gpt.py`、`prepare_amg_lengths.py`、`prepare_amg_length_features.py` | 生成配对特征、冻结 VQ 相关资源和长度模块输入 |
| 评估 | `tools/eval_vq.py`、`tools/eval_gpt.py` | 评估 VQ 重建与 GPT 生成结果 |
| 生成 | `tools/generate_amg_length_controlled.py` | 单段长度控制生成入口 |
| 可视化 | `visualize/` | 潜空间分析与结果绘图 |

`tools/run_*.sh` 中保留的参数、目录检查和恢复逻辑属于各自实验脚本的一部分。修改或运行前应直接阅读对应源码，并核对资源清单和实现范围；本索引不声明这些入口已经组成完整或重新验证过的复现流程。
