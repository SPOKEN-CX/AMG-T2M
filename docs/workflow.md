# 执行顺序与命令

本页给出仓库当前 HumanML3D 组件的执行顺序。命令均从仓库根目录运行。数据、归一化统计、评估器、GloVe、CLIP/TMR 资源和模型权重不随源码提供，目录要求见[数据集说明](datasets.md)、[数据与模型](data-and-models.md)及[`resources/manifest.json`](../resources/manifest.json)。先完成[环境准备](environment.md)。

当前代码包含 TMR 引导 VQ、GPT、长度估计和单段长度控制生成。服务器中的 `train_transition.py` 实际仍使用普通 `Text2Motion_Transformer` 和 GPT 训练目标，没有调用 `TransitionGenerator`；`train_GPT_split.py` 为空文件。因此下面的顺序复现的是现有模块，不代表完整的任意长度转场系统或论文全部结果。实现范围见[状态说明](implementation-status.md)。

## 一、准备 AMG VQ 输入

确保 HumanML3D 特征和划分已按[数据说明](datasets.md)放好，且 TMR 的模型、统计文件和文本 token 缓存已就绪。先生成配对清单和审计文件：

```bash
python preprocess/prepare_amg_tmr.py \
  --output output/amg_prepare/humanml3d_train_ready \
  --resolve-whitespace
```

检查 `output/amg_prepare/humanml3d_train_ready/audit.json`：`errors` 应为空。若 `missing_precomputed_text` 大于 0，先按下方命令恢复特征，并在编码时提供 `--missing-token-dir`；若为 0，则无需恢复步骤。TMR 编码使用统一环境中的依赖：

```bash
python -m pip install -r requirements.txt

# 仅当 audit.json 显示 missing_precomputed_text > 0 时运行：
python preprocess/prepare_amg_missing_text.py \
    --audit output/amg_prepare/humanml3d_train_ready/audit.json \
    --model-dir output/amg_prepare/models/distilbert-base-uncased \
    --output output/amg_prepare/missing_tokens
```

配对清单审计通过后，编码动作和文本的 TMR 表示。若生成了缺失文本特征，带上 `--missing-token-dir`；若没有缺失项，省略该参数：

```bash
python preprocess/prepare_amg_tmr.py \
    --output output/amg_prepare/humanml3d_train_ready \
    --resolve-whitespace \
    --missing-token-dir output/amg_prepare/missing_tokens \
    --encode --device cuda --batch-size 16
```

输出目录中应有 `train_pairs.jsonl`、`motion/`、`text/`；最终 `audit.json` 中 `errors` 和 `missing_unresolved_text` 应为空或为 0。若初次审计没有缺失项，删除上面命令中的 `--missing-token-dir output/amg_prepare/missing_tokens` 即可。预处理脚本不会替使用者下载数据或模型。

## 二、训练并冻结 AMG VQ

推荐的 batch 512 配置分两段：第一段训练到 10,000 步，第二段从其 checkpoint 续训到总计 100,000 步。第二条命令会检查并恢复第一段的最终状态，不要跳过第一段：

```bash
bash tools/run_amg_vq_b512_check10000.sh
bash tools/run_amg_vq_b512_100k.sh
```

每段结束后检查对应目录的 `status.txt`、`result_summary.json` 和日志。完成后冻结并核验最佳验证 checkpoint：

```bash
python preprocess/prepare_amg_gpt.py \
  --source-dir output/hml3d/VQ_amg_b512_k50_100k \
  --output output/hml3d/VQ_amg_b512_frozen
```

GPT 必须使用这份冻结 VQ 及其 `manifest.json` 身份；不要把另一组 VQ 权重与 GPT checkpoint 混用。

## 三、训练 AMG GPT

先执行 100 步检查，再运行 100,000 步配置：

```bash
bash tools/run_amg_gpt_b512.sh check
bash tools/run_amg_gpt_b512.sh full
```

这两个模式的实际训练 batch 都不是脚本名中的 `b512`：`check` 为 batch 32、100 步；`full` 为 batch 128、100,000 步。其它模式的参数如下，属于不同训练协议，不能把它们的 checkpoint 当成同一次训练的续训结果：

| 模式 | 命令 | microbatch × 梯度累积 | 有效 batch | 总步数 | 前置条件 |
|---|---|---:|---:|---:|---|
| `extend300k` | `bash tools/run_amg_gpt_b512.sh extend300k` | 128 × 1 | 128 | 300,000 | `full` 的训练状态及 100k 归档已完成 |
| `gpt512300k` | `bash tools/run_amg_gpt_b512.sh gpt512300k` | 512 × 1 | 512 | 300,000 | 先完成 `check` |
| `gpt4096check` | `bash tools/run_amg_gpt_b512.sh gpt4096check` | 512 × 8 | 4,096 | 2 | 先完成 `check` |
| `gpt4096full` | `bash tools/run_amg_gpt_b512.sh gpt4096full` | 512 × 8 | 4,096 | 150,000 | 先完成 `gpt4096check` |

启动脚本会执行相应的 checkpoint、VQ 身份和运行状态检查。训练结束以运行目录中的 `status.txt` 显示 `completed and verified` 为准；失败后先查看 `console.log` 和 `verification.log`，不要删除运行目录后盲目重跑。

## 四、评估

VQ 的一次测试评估入口使用上面的 100k checkpoint：

```bash
bash tools/run_amg_vq_test_once.sh
```

GPT 可用同一组冻结 VQ 和 GPT checkpoint 执行标准 HumanML3D 测试评估。以下示例评估 `full` 模式并重复 20 次；如使用其它 GPT 模式，只替换 GPT checkpoint 路径和输出实验名：

```bash
python tools/eval_gpt.py \
  --dataname t2m --num-layers 9 --embed-dim-gpt 1024 --n-head-gpt 16 \
  --block-size 51 --ff-rate 4 --drop-out-rate 0.1 \
  --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 \
  --quantizer ema_reset --vq-act relu \
  --resume-pth output/hml3d/VQ_amg_b512_frozen/net_best_fid.pth \
  --resume-trans output/hml3d/GPT_amg_b512_full_bf16/net_best_fid.pth \
  --out-dir output/hml3d_eval --exp-name AMG_GPT_full --repeat-time 20
```

评估依赖与训练数据相同的 HumanML3D split、评估器、GloVe、CLIP 和冻结 VQ。请保留评估命令、checkpoint 哈希和输出日志。

## 五、长度预测（独立分支）

长度估计不依赖 AMG VQ/GPT 训练。先生成 full/timed caption 长度记录，确认已按[环境说明](environment.md)安装统一依赖：

```bash
python preprocess/prepare_amg_lengths.py \
  --output output/amg_prepare/lengths_full_v2

# TMR, length dependencies and the spaCy model are already included in requirements.txt.
bash tools/run_amg_lengths.sh
```

该启动脚本会准备 CLIP 词特征和依存图、检查缓存、执行 batch 512 的 20 步启动检查，再训练 50,000 步并评估。最终最佳 checkpoint 位于 `output/hml3d/Length_amg_gat_b512_50k/net_best_mae.pth`。

## 六、单段长度控制生成

已具备匹配的 VQ、GPT、长度 checkpoint 后，可运行：

```bash
python tools/generate_amg_length_controlled.py \
    --gpt-checkpoint output/hml3d/GPT_amg_b512_full_bf16/net_best_fid.pth \
    --vq-checkpoint output/hml3d/VQ_amg_b512_frozen/net_best_fid.pth \
    --length-checkpoint output/hml3d/Length_amg_gat_b512_50k/net_best_mae.pth \
    --spacy-model en_core_web_sm \
    --text "A person walks forward, then sits down and stands up." \
    --out-dir output/demo/amg_single_segment
```

当前入口一次为每条 caption 生成一段动作，GPT 上限为 49 个动作 token（196 帧，20 FPS）。它不执行长文本分段、段间转场生成或多段动作拼接；超出上限的预测默认报错，不会自动裁剪。完整任意长度生成和转场模型的训练/评估仍需补齐。
