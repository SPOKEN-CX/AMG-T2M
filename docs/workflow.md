**HumanML3D 使用顺序与脚本说明**

本文按仓库现有源码说明资源、入口和输出之间的关系。本轮只整理说明，不执行安装、数据准备、训练、评估或demo。已有实验结果继续以原实验记录为准。

按顺序阅读 [环境](environment.md)、[数据与模型](data-and-models.md)、[实现范围](implementation-status.md)。涉及GPU的命令需要另行安排资源。下文参数来自现有脚本，不将这些工程设置自动等同于论文全部设置。

**1．准备基础资源**

HumanML3D动作特征、文本和官方train/val/test划分放在 `dataset/HumanML3D/`；GloVe放在 `glove/`；评估器和对应配置放在 `checkpoints/t2m/`；归一化统计使用 `checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy` 和 `std.npy`。GPT与长度模块还需要根目录的 `ViT-B-32.pt`。

具体文件见 [资源目录](../resources/manifest.json)。生成的 `(T,22,3)` 关节坐标与输入 `(T,263)` 特征用途不同。缓存、token、评估器和模型均应保持配套；不要仅按文件名相同就替换。

**2．区分基线与AMG VQ**

| 路线 | 入口 | 说明 |
|---|---|---|
| 基础VQ | `tools/train_vq.py` | 不需要TMR配对特征；单独保留基线输出 |
| AMG流程短跑 | `tools/run_amg_vq_check100.sh` | batch32、100步、正/负近邻各10；用于检查流程 |
| AMG batch128路线 | `tools/run_amg_vq_full.sh` | batch128、300000步；输出 `VQ_amg_full` |
| AMG batch512第一阶段 | `tools/run_amg_vq_b512_check10000.sh` | batch512、10000步、正/负近邻各50 |
| AMG batch512续训 | `tools/run_amg_vq_b512_100k.sh` | 检查第一阶段状态并续训至总计100000步；输出 `VQ_amg_b512_k50_100k` |

AMG路线先准备TMR动作/文本表示。`preprocess/prepare_amg_tmr.py` 使用TMR权重、统计量及预计算DistilBERT token特征；`preprocess/prepare_amg_missing_text.py` 补算缺失caption。最终配对入口是 `output/amg_prepare/humanml3d_train_ready/train_pairs.jsonl` 及同目录 `motion/`、`text/`。配对准备的筛选、缺失记录和审计应与选定实验一起保存。

两个batch512脚本的公共设置为：窗口64帧、学习率2e-4、码本512、下采样4倍、对比温度0.07、对比权重0.01、正负近邻各50；每1000步验证。`100k` 表示最终总步数，不能理解为在10000步后再新增100000步。

**3．冻结VQ，再训练配套GPT**

`preprocess/prepare_amg_gpt.py` 默认输入是旧路线 `VQ_amg_full`，默认输出为 `VQ_amg_frozen`。它支持显式指定来源与输出。batch512路线应使用：

```bash
python preprocess/prepare_amg_gpt.py \
  --source-dir output/hml3d/VQ_amg_b512_k50_100k \
  --output output/hml3d/VQ_amg_b512_frozen
```

该入口会加载模型、检查训练状态及历史最佳checkpoint，并使用GPU做编码/解码检查；它会创建冻结权重和manifest。本轮不执行。迁移服务器时应在目标路径生成适用的manifest，避免直接沿用内部绝对路径。

`tools/run_amg_gpt.sh` 使用旧路线冻结目录；`tools/run_amg_gpt_b512.sh` 使用 `VQ_amg_b512_frozen`。后者各模式如下：

| 模式 | 实际microbatch | 梯度累积 | 有效batch（单卡） | 总优化步数 |
|---|---:|---:|---:|---:|
| `check` | 32 | 1 | 32 | 100 |
| `full` | 128 | 1 | 128 | 100000 |
| `extend300k` | 128 | 1 | 128 | 300000 |
| `gpt512300k` | 512 | 1 | 512 | 300000 |
| `gpt4096check` | 512 | 8 | 4096 | 2 |
| `gpt4096full` | 512 | 8 | 4096 | 150000 |

目录名中的VQ batch与GPT实际batch需要分别确认。`extend300k` 延续自己的完整训练状态；`gpt512300k`、`gpt4096full` 使用各自实验目录。所有非 `check` 模式依赖100步检查通过；`gpt4096full` 还要求对应2步累积检查通过。不要手工伪造这些检查文件。

查看命令的dry-run写法如下，支持的脚本会在创建实验目录之前退出：

```bash
bash tools/run_amg_gpt_b512.sh gpt512300k --dry-run
```

其余 `run_*.sh` 不一定支持 `--dry-run`，应先阅读脚本。正式运行时保留 `vq_identity.json`、完整训练checkpoint、命令和验证记录。token缓存必须由该冻结VQ导出。

**4．长度模块单独准备与训练**

先按 [环境文档](environment.md) 准备长度附加依赖和 `en_core_web_sm==3.5.0`。长度记录按官方split从caption和动作帧数生成：

```bash
python preprocess/prepare_amg_lengths.py \
  --output output/amg_prepare/lengths_full_v2
```

默认输出目录是 `lengths`，而特征入口默认读取 `lengths_full_v2`，因此这里显式指定。阅读生成的 `audit.json`，处理异常记录并保存筛选依据，再导出CLIP词特征和依存图：

```bash
PYTHONPATH="$PWD/.runtime/length_deps${PYTHONPATH:+:$PYTHONPATH}" \
  python preprocess/prepare_amg_length_features.py \
  --records output/amg_prepare/lengths_full_v2 \
  --output output/amg_prepare/length_features_v1 \
  --spacy-model .runtime/length_models/en_core_web_sm/en_core_web_sm-3.5.0
```

`tools/run_amg_lengths.sh` 读取上述记录/特征；包含20步检查和batch512、50000步训练，学习率1e-3，验证帧数MAE用于选模。结果目录为 `output/hml3d/Length_amg_gat_b512_50k/`。长度预测与GPT训练分别运行；单段长度控制入口见 `tools/generate_amg_length_controlled.py`。原默认上限为49个动作token，即196帧，完整长动作过渡流程见实现范围中的待补部分。

单段生成脚本的 `--spacy-model` 默认仍指向历史 `output/amg_prepare/length_models/` 目录。若按环境文档安装到 `.runtime/`，运行该入口时显式传入 `--spacy-model .runtime/length_models/en_core_web_sm/en_core_web_sm-3.5.0`，并按长度模块方式设置 `PYTHONPATH`。

**5．评估与结果整理**

| 入口 | 当前用途 |
|---|---|
| `tools/eval_vq.py` | VQ重建测试；支持 `--repeat-time`，默认20 |
| `tools/run_amg_vq_test_once.sh` | batch512 VQ单次测试，显式 `--repeat-time 1` |
| `tools/eval_gpt.py` | 配套VQ/GPT生成测试；支持 `--repeat-time`，默认20 |
| `tools/run_gpt_a100_test.sh` | 基线VQ/GPT路径；先单次，再20次测试，不是AMG通用评估启动器 |
| `visualize/latent_tsne_comparison/run.sh` | 特征/图表分析，具体checkpoint与样本配置见该目录 |

VQ重建FID、GPT生成FID和长度MAE分别记录。主训练/评估参数的默认seed为123；如原实验使用其他种子，应以其配置和checkpoint记录为准。每项结果保留实际样本数、split、checkpoint哈希、重复次数及统计方法。

`net_best_*.pth` 通常是推理模型；完整续训还需要 `training_last.pth` 中的优化器、scheduler及相关状态。已有输出目录可能触发自动恢复或拒绝覆盖，运行前应确认目标目录，而非删除原实验记录。
