> Current cleanup: obsolete downloaders, standalone tests, the duplicate GPT trainer, empty placeholders, fixed-result/sample scratch scripts, and stale check outputs have been removed. See [the current repository audit](../../docs/repository-audit.md). Names in historical notes below identify removed files where applicable.

> 当前安装入口：在仓库根目录执行 `python -m pip install -r requirements.txt`，环境说明见 `../../docs/environment.md`。下文服务器配置为历史运行记录。

# AMG：Linux GPU 服务器上先跑通 HumanML3D 训练和评估

核对日期：2026-10-26。依据用户提供的 `本地论文 PDF`、本项目当前源码和本机数据目录编写。

本文给出执行顺序和必要的代码适配位置。本次没有修改训练源码，没有登录服务器，也没有在服务器执行训练或评估。下文标为“修复后”的命令，不能跳过前面的修复直接运行。

阅读顺序：第 3–9 节是运行步骤，第 5 节是运行前的必要适配，第 10 节是 AMG 扩展模块；第 12–16 节逐文件说明数据格式、参数、损失、输出和指标计算。

## 1. 先确定验收目标

论文是 **AMG: Long-Term Text-to-Motion Generation from Arbitrary-Length Prompts**，当前 README 仍是 T2M-GPT 原版说明。

建议分两轮：

1. **先跑通 HumanML3D 的基础链路**：真实数据 → 基础 VQ-VAE 训练 → 固定 VQ 编码动作 token → GPT 训练 → 官方测试划分评估。先各跑 100 步，确认训练、验证、保存和重新加载都成功，再做长训练。
2. **再接入 AMG 的改进**：TMR 语义特征 → Weighted InfoNCE VQ-VAE → 用新 VQ 重新训练 GPT；另外训练长度预测器、过渡模块，最后做完整长动作评估。

第一轮得到的是基础链路结果。它不能直接作为完整 AMG 的论文复现结果。以下 30 万步等配置来自项目的 T2M-GPT 基线运行设置；不是从这份 AMG 论文中核实出的完整最佳超参数。

## 2. 当前资源状态：动作数据已找到，旧文档需要更新

本机实际数据位置：`/path/to/AMG-T2M/reproduction/Dataset/HumanML3D`。这是项目的同级目录，不是项目内部的 `dataset/HumanML3D`。

| 检查项 | 本次核对结果 |
|---|---|
| `new_joint_vecs/` | 29,228 个 `.npy` 文件 |
| `train.txt` | 23,384 个 ID，对应动作文件缺失数为 0 |
| `val.txt` | 1,460 个 ID，对应动作文件缺失数为 0 |
| `test.txt` | 4,384 个 ID，对应动作文件缺失数为 0 |
| 三个划分之间的 ID 交集 | 均为 0 |
| `texts.zip` | 存在，压缩包覆盖上述三个划分的所有文本 ID |
| `texts/` | 尚未解压 |
| 抽查动作 | `000000`、`M000000`、`000090` 均是有限值的 `(T,263)` float32 特征 |
| `Mean.npy`、`Std.npy` | 均为 263 维；Std 为正数 |
| GloVe、HumanML3D 评估器、CLIP、本地预训练 VQ/GPT | 本机已找到 |
| `motion_retrival/`、`texts_retrival/` | 未找到；AMG 对比学习需要额外生成 |
| `texts_GPT_processed/` | 未找到；基础 GPT 当前没有实际使用步骤文本，可解除这一强制依赖 |

这说明本机已具备基础 HumanML3D 训练所需的动作和压缩文本资源。服务器上的文件是否齐全，仍需在服务器检查。旧文档中的“没有完整动作数据”不再适用于当前本机状态。

## 3. 服务器准备：沿用已有环境，先检查

`SERVER_RUN_GUIDE.zh-CN.md` 记录过以下服务器配置：项目目录 `/path/to/AMG-T2M/reproduction/local`，环境 `/path/to/envs/t2mgpt`，Python 3.10，PyTorch 2.0.1+cu117，A100 80GB。这些是历史记录，本次没有远程复核。如果继续使用同一套服务器，按下面进入；否则替换项目和环境路径。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /path/to/envs/t2mgpt
cd /path/to/AMG-T2M/reproduction/local
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
```

```bash
python - <<'PY'
import sys
import torch
import numpy
import scipy
import clip
from torch.utils.tensorboard import SummaryWriter
print('python:', sys.executable)
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), '当前终端没有可用 CUDA'
print('GPU:', torch.cuda.get_device_name(0))
a = torch.randn(128, 128, device='cuda')
print('CUDA matmul:', (a @ a).shape)
PY
```

如果上述检查通过，不用重新安装 PyTorch，也不要把 Mac 演示用的 历史 Mac 最小依赖清单（已归档） 安装到服务器来覆盖环境。项目的旧 `environment.yml` 锁定了较老的 Linux/CUDA 包，不建议直接覆盖已经可用的服务器环境。

## 4. 把真实数据放到服务器正确位置

目标目录：

```text
/path/to/AMG-T2M/reproduction/
├── T2M-GPT-main/
│   ├── train_vq.py
│   ├── train_t2m_trans.py
│   ├── glove/
│   ├── checkpoints/
│   └── ViT-B-32.pt
└── Dataset/
    └── HumanML3D/
        ├── new_joint_vecs/
        ├── texts/
        ├── Mean.npy
        ├── Std.npy
        ├── train.txt
        ├── val.txt
        ├── test.txt
        ├── train_val.txt
        └── all.txt
```

当前 HumanML3D 加载器主要使用 `../Dataset/HumanML3D`。Linux 区分大小写，统一使用这里的大写 `Dataset`。`options/get_eval_option.py` 中还有小写 `../dataset/HumanML3D`，也应统一，避免后续模块从不同位置读取数据。

可以用 VS Code 的 SSH 文件窗口传输，或在 Mac 终端用 rsync。下面的 `n1` 必须是你日常能连接的 SSH 地址或别名；若实际地址不同，替换它。

```bash
ssh user@server.example 'mkdir -p /path/to/AMG-T2M/reproduction/Dataset/HumanML3D'
rsync -avh --progress /path/to/AMG-T2M/reproduction/Dataset/HumanML3D/ \
  user@server.example:/path/to/AMG-T2M/reproduction/Dataset/HumanML3D/
```

项目代码及后续修复也要同步到服务器。GloVe、`checkpoints/`、`ViT-B-32.pt` 若已在服务器，不必重复上传。基础训练评估至少检查：

```text
glove/our_vab_data.npy
glove/our_vab_words.pkl
glove/our_vab_idx.pkl
checkpoints/t2m/Comp_v6_KLD005/opt.txt
checkpoints/t2m/text_mot_match/model/finest_1.tar
checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy
checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy
ViT-B-32.pt
```

在服务器项目根目录解压文本：

```bash
python -m zipfile -e ../Dataset/HumanML3D/texts.zip ../Dataset/HumanML3D
```

这个压缩包内部已经有 `texts/` 前缀，所以解压目标是 `HumanML3D`，不要解压成 `texts/texts/`。

然后在服务器执行数据检查：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
root = Path('../Dataset/HumanML3D')
splits = {}
for split in ['train', 'val', 'test']:
    ids = (root / f'{split}.txt').read_text().split()
    missing = [i for i in ids if not (root/'new_joint_vecs'/f'{i}.npy').is_file()
               or not (root/'texts'/f'{i}.txt').is_file()]
    print(split, 'IDs:', len(ids), 'missing:', len(missing), missing[:5])
    assert not missing
    splits[split] = set(ids)
assert not (splits['train'] & splits['val'])
assert not (splits['train'] & splits['test'])
assert not (splits['val'] & splits['test'])
for i in ['000000', 'M000000', '000090']:
    x = np.load(root/'new_joint_vecs'/f'{i}.npy')
    print(i, x.shape, x.dtype)
    assert x.ndim == 2 and x.shape[1] == 263 and np.isfinite(x).all()
print('数据检查通过')
PY
```

训练加载器会按动作长度等条件筛选，因此有效训练样本数不一定等于 ID 总数，但必须大于 0 且足够组成完整 batch。生成的 `(T,22,3)` 关节坐标和 GIF 不能直接替代 `(T,263)` 动作特征。

数据格式和处理来源：[HumanML3D 官方仓库](https://github.com/EricGuo5513/HumanML3D)。使用项目已有的标准化统计，并在训练、token 编码、重建和评估中保持一致；不要把已经标准化的动作再标准化一次。

## 5. 正式训练前必须完成的源码适配

这些修改尚未由本次工作执行。建议先备份源文件，再按下面的顺序改。

### 5.1 基础 VQ：复用现有不要求 TMR 的加载类

`train_vq.py`：

1. 删除 `import parser`。Python 3.10 已没有这个旧标准库模块。
2. 删除未使用的 `from models.Retrival.text_motion_sim import text_motion_sim`。基础 VQ 不需要因此导入整套 TMR。
3. 对 HumanML3D 基础训练，使用 `dataset/dataset_VQ.py` 已有的 **`VQMotionDataset_V1`**。它返回 `(motion, caption)`，不强制读取检索嵌入，与该训练脚本的两项解包一致。保留原来的 `VQMotionDataset` 给 AMG 对比学习使用。

把基础训练的 `train_loader` 创建改为下面的逻辑；不要全局更改对比学习共用的 `DATALoader`：

```python
train_set = dataset_VQ.VQMotionDataset_V1(
    args.dataname, window_size=args.window_size,
    unit_length=2**args.down_t,
)
assert len(train_set) >= args.batch_size, '有效训练样本不足一个 batch'
train_loader = DataLoader(
    train_set, batch_size=args.batch_size,
    shuffle=True, num_workers=8, drop_last=True,
)
```

4. HumanML3D 分支目前没有创建后续使用的 `val_loader`。在 BABEL 分支之外补上：

```python
val_loader = dataset_TM_eval.DATALoader(
    args.dataname, False, 32, w_vectorizer,
    unit_length=2**args.down_t,
)
```

这里 `False` 表示 `val.txt`。训练期间用验证集选择 checkpoint，不能用测试集选模型。

### 5.2 GPT：去掉没有参与基础训练的额外输入依赖

`train_t2m_trans.py`：

1. 删除未使用的 `from models.MotionLenEstimator.LengthEst import *`。基础 GPT 不需要提前安装图网络和 spaCy。
2. 非 BABEL 分支的验证加载改为 `DATALoader(args.dataname, False, 32, w_vectorizer, unit_length=2**args.down_t)`。原代码使用 `True, 16`，既读错划分，检索候选 batch 大小也与标准测试不一致。
3. CLIP 改为显式加载根目录已有文件：`clip.load('./ViT-B-32.pt', device=device, jit=False)`。`GPT_eval_multi.py` 同样改成本地路径。CLIP 会先在 CPU 打开这个文件并按 device 放到 GPU。
4. `dataset/dataset_TM_train.py` 现在发现缺少 `texts_GPT_processed/<id>.txt` 就跳过整条样本，但当前基础 GPT 的实际损失没有使用步骤文本。

   对基础模式，将步骤文件变为可选：先设置 `steps_info_list = []`，仅在文件存在时执行原来的 JSON 读取；删除“文件不存在就 continue”的逻辑。已有 `__getitem__` 会在空列表时返回 `total_steps=0`、`steps_dict={}`，训练脚本仍可接收五项返回值。删除“不是三步就打印”的调试输出。不要批量伪造步骤标注文件。

5. 检查 token 加载后的有效样本数至少有一个完整 batch。不要用 `except: pass` 把格式错误和缺文件都隐藏起来。

真实入口是 `train_t2m_trans.py`。README 的 `train_t2m_trans_1.py` 在此项目不存在。

### 5.3 评估：恢复官方测试划分，接上真实多模态指标

`dataset/dataset_TM_eval.py` 的 `is_test=True` 当前读取 `long_test.txt`。改成 `test.txt`，`False` 分支继续读取 `val.txt`。自定义长文本子集以后单独设置入口和实验名。

`GPT_eval_multi.py` 当前创建了空的 `multi=[]`，却没有往其中添加计算结果，因此最终打印的 `multi: 0.0` 和 NaN 置信区间没有评价意义。

如果需要完整五类指标，在 20 次循环中调用已经存在的 `utils.eval_trans.evaluation_transformer_test`，而不是普通的 `evaluation_transformer`。调用和返回值顺序如下：

```python
result = eval_trans.evaluation_transformer_test(
    args.out_dir, val_loader, net, trans_encoder,
    logger, writer, 0,
    best_fid=1000, best_iter=0, best_div=100,
    best_top1=0, best_top2=0, best_top3=0,
    best_matching=100, best_multi=0,
    clip_model=clip_model, eval_wrapper=eval_wrapper,
    draw=False, save=False, savegif=False, savenpy=False,
)
fid_i, _, div_i, top1_i, top2_i, top3_i, match_i, multi_i, writer, logger = result
fid.append(fid_i)
div.append(div_i)
top1.append(top1_i)
top2.append(top2_i)
top3.append(top3_i)
matching.append(match_i)
multi.append(multi_i)
```

注意替换原循环的调用和 append，不要在原 append 后再额外添加一遍。该测试函数为每条文本采样 30 次、用其中的距离估计多模态性；耗时明显高于训练期间的普通验证。先用 `repeat_time=1` 检查流程，正式报告再恢复 20 次。论文 Table I 要求报告 20 次评估的均值和 95% 置信区间。

额外检查：

- `GPT_eval_multi.py` 若没有提供 `--resume-trans`，当前代码仍能继续评估随机 GPT。必须在加载前检查 VQ 和 GPT 两个 checkpoint 路径有效。
- `models/t2m_trans.py` 的 `sample()` 第一步遇到 EOS 时，`xs` 尚未初始化；评估函数现在用宽泛 `except` 和固定 token 代替失败生成。应修复首步 EOS 的边界处理，并让其它采样异常直接报错；否则程序结束也不能证明指标有效。
- 基础评估保持 batch=32。两个评估入口内部把它写死为 32，命令行 `--batch-size` 不能改变这里。
- 全量评估必须使用相同的文本采样、运动长度处理、随机种子策略和评估器。不要把贪心生成的普通验证结果混作随机采样的多模态结果。
- 若使用官方预训练 VQ，本分支额外的 `projection_layer.weight/bias` 不在官方权重里。仅允许这两个已知缺项，其余缺项和多余项仍报错。完全随机初始化后从本分支训练出的 VQ，不存在这类跨版本缺项。

对于首步 EOS，可以在 `sample()` 取出当前步 `logits`、计算 softmax 之前加入下面的边界策略，保证至少生成一个有效动作 token；同时删除评估中将其它异常吞掉的宽泛 `except`，保留明确的错误信息：

```python
if k == 0:
    logits[:, self.num_vq] = float('-inf')
```

这是为了处理首步空序列的工程策略，应在实验配置中保留说明。另需在 `dataset_TM_eval.py` 每个样本读取 caption 前初始化 `flag=False`，避免沿用上一个样本的状态。

初次跑训练和指标时，建议在训练入口调用评估函数的地方显式传 `draw=False`，先关闭视频绘制，避免初始第 0 步验证就引入 MoviePy/绘图库问题。当前函数还把验证 `writer.add_scalar` 放在 `if draw` 内；如需继续记录验证 TensorBoard 标量，将标量记录移到该条件之外，视频绘制仍由 `draw` 控制。两者都不改变指标公式。

## 6. 先跑很小的检查

### 6.1 可以先独立检查 CUDA 反向传播

已同步 `scripts/run_local.py`、`motions/` 样例及预训练 VQ 后，在服务器根目录运行：

```bash
python scripts/run_local.py train-smoke --device cuda --steps 3 --no-gif
```

这个入口会检查 loss、梯度是否有限，以及编码器参数是否实际改变，输出 `result.json` 和 `smoke_vq.pth`。它使用现有样例和预训练初始化，只是计算链路检查；不要把这个权重拿去当正式训练结果。这里不用将完整动作目录传给 `--source`，避免将所有动作加载进这个演示入口。

### 6.2 完成第 5 节修复后，跑正式入口的 100 步

```bash
python -u train_vq.py \
  --dataname t2m --batch-size 32 --window-size 64 \
  --warm-up-iter 10 --total-iter 100 \
  --print-iter 10 --eval-iter 50 \
  --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 \
  --quantizer ema_reset --vq-act relu \
  --loss-vel 0.5 --recons-loss l1_smooth \
  --out-dir output/hml3d --exp-name VQ_check100
```

通过标准：取到真实 batch，warm-up 和正式反向传播成功，完整验证结束，`output/hml3d/VQ_check100/` 出现非空 checkpoint 与 `run.log`，checkpoint 可重新加载。

100 步中 loss 不必单调下降。重点是没有空数据、NaN、维度错误、保存或加载错误。原入口开始就会验证，不能只拿几条样例替代真实验证集；当前 diversity 计算对样本数也有要求。

## 7. 正式训练基础 VQ-VAE

只在第 6.2 节通过后执行：

```bash
python -u train_vq.py \
  --dataname t2m --batch-size 256 --window-size 64 \
  --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 \
  --nb-code 512 --code-dim 512 --down-t 2 --depth 3 \
  --dilation-growth-rate 3 --quantizer ema_reset --vq-act relu \
  --loss-vel 0.5 --recons-loss l1_smooth \
  --eval-iter 1000 --print-iter 200 \
  --out-dir output/hml3d --exp-name VQ_base
```

在 `output/hml3d/VQ_base/` 中看 `run.log`、TensorBoard、`net_best_fid.pth`、`net_last.pth`。用验证 FID 选定 VQ；训练完成后固定该文件，不要在后续 GPT 训练过程中替换它。

```bash
tensorboard --logdir output/hml3d --port 6006
```

观察重建损失、commitment loss、码本 perplexity、验证 FID；perplexity 长期极低时检查码本塌缩。不能只凭训练 loss 变小判断模型好坏。若显存不足，可先减 batch；记录最终配置以便比较。

## 8. 固定 VQ，生成 token，训练 GPT

第 5.2 节修复后，先用独立实验名检查 GPT 100 步：

```bash
python -u train_t2m_trans.py \
  --dataname t2m --exp-name GPT_check100 --out-dir output/hml3d \
  --batch-size 32 --total-iter 100 --print-iter 10 --eval-iter 50 \
  --num-layers 9 --embed-dim-gpt 1024 --n-head-gpt 16 \
  --block-size 51 --ff-rate 4 --drop-out-rate 0.1 \
  --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 \
  --quantizer ema_reset --vq-act relu --pkeep 0.5 --lr 1e-4 \
  --resume-pth output/hml3d/VQ_base/net_best_fid.pth \
  --vq-name VQ_base_tokens
```

通过后，完整训练用新实验名：

```bash
python -u train_t2m_trans.py \
  --dataname t2m --exp-name GPT_base --out-dir output/hml3d \
  --batch-size 128 --total-iter 300000 --lr 1e-4 --lr-scheduler 150000 \
  --num-layers 9 --embed-dim-gpt 1024 --n-head-gpt 16 \
  --block-size 51 --ff-rate 4 --drop-out-rate 0.1 \
  --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 \
  --quantizer ema_reset --vq-act relu --pkeep 0.5 \
  --eval-iter 10000 --print-iter 200 \
  --resume-pth output/hml3d/VQ_base/net_best_fid.pth \
  --vq-name VQ_base_tokens
```

这个脚本先遍历训练动作，自动把 token 保存到 `../Dataset/HumanML3D/VQ_base_tokens/`，然后开始 GPT 训练；不需要猜一个不存在的独立 token 导出命令。token 导出阶段应让 VQ 保持 eval 并使用 `torch.no_grad()`，固定 VQ 参数。

关键关系：**VQ checkpoint → 它生成的 token 目录 → 用该目录训练的 GPT checkpoint**。换 VQ 后必须换 token 目录并重新编码、重新训练对应 GPT。即使码本大小相同，另一套 VQ 的 token 编号语义也可能不同。

输出在 `output/hml3d/GPT_base/`。不要把命令行中的 `--resume-pth` 当成 GPT 权重；它是 VQ。GPT 初始化权重使用 `--resume-trans`，从零训练 GPT 时不传该项。

## 9. 标准测试集评估

### 9.1 VQ 重建评估

第 5.3 节中的测试划分问题修好后：

```bash
python -u VQ_eval.py \
  --dataname t2m --nb-code 512 --down-t 2 --depth 3 \
  --dilation-growth-rate 3 --quantizer ema_reset --vq-act relu \
  --resume-pth output/hml3d/VQ_base/net_best_fid.pth \
  --out-dir output/hml3d_eval --exp-name VQ_base_test
```

它评估的是输入真实动作经过 VQ 编解码的质量。

### 9.2 文本生成动作评估

第 5.3 节中的采样、指标和 checkpoint 检查接好后：

```bash
python -u GPT_eval_multi.py \
  --dataname t2m --num-layers 9 --embed-dim-gpt 1024 --n-head-gpt 16 \
  --block-size 51 --ff-rate 4 --drop-out-rate 0.1 \
  --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 \
  --quantizer ema_reset --vq-act relu \
  --resume-pth output/hml3d/VQ_base/net_best_fid.pth \
  --resume-trans output/hml3d/GPT_base/net_best_fid.pth \
  --out-dir output/hml3d_eval --exp-name GPT_base_test
```

VQ 必须和第 8 节生成 token 时的文件完全相同。可用 `sha256sum` 记录 checkpoint 身份，连同实验参数保存。

| 指标 | 含义与判断 |
|---|---|
| FID | 生成动作特征分布与真实分布的差异，越低越好 |
| R-precision Top1/2/3 | 文本动作检索匹配率，越高越好 |
| Matching / MM-Dist | 文本与动作的嵌入距离，越低越好 |
| Diversity | 不同生成动作之间的多样性，要结合真实动作的 diversity 比较 |
| Multimodality | 同一文本多次采样的动作差异，应与质量和文本匹配共同分析 |

基础 GPT 测试是生成质量评估。VQ 重建 FID 和 GPT 生成 FID 回答的问题不同，不能当成同一个实验结果。

正式汇报：固定官方测试划分、batch=32、评估器与采样设置，重复 20 次，记录均值与 95% 置信区间。先用一次检查运行，再恢复正式次数。测试集只用于最终报告，模型选择使用验证集。

## 10. 跑通基础链路后，接入 AMG

### 10.1 对比学习增强 VQ-VAE

对应 `train_vq_constra_100.py`、`train_vq_constra.py`，不等于第 7 节的基础 VQ。

顺序：

1. 使用 `models/Retrival/` 中 HumanML3D 的预训练 TMR，先核对一组动作和文本能编码。TMR 是语义特征来源，不是本项目最终生成结果所用的标准评估器。
2. 批量提取训练样本的 motion/text embedding，分别保存到 `../Dataset/HumanML3D/motion_retrival/<id>.npy`、`texts_retrival/<id>.npy`，拼写保持 `retrival`。如果一个 ID 随机采样多个 caption，特征必须与实际使用的 caption 对应；局部动作裁剪与嵌入的对应关系也要明确。
3. 本地 `encode_motion.py` / `encode_text.py` 能演示单条编码，但保存目录和命名不是上述训练目录，需要适配成批量导出程序。不要假设运行一次单条示例就生成完整训练特征。
4. 导入 TMR 时从项目根目录设置 `export PYTHONPATH="$PWD/models/Retrival${PYTHONPATH:+:$PYTHONPATH}"`，并把 `/path/to/...` 等固定路径替换成实际路径。已有环境按缺失的 TMR 依赖补充，不要直接覆盖整套 PyTorch。
5. 对比 VQ 的验证加载也改为 `False, 32`，不要用测试集选权重。
6. 修复正负样本选择后再训练；当前 `_100.py` 固定 K=100，另一版固定 K=150，不能直接缩小 batch 而保留 K。

当前代码把相似度矩阵对角线设为 `-inf` 后，同时取最大和最小 K 项，导致自己进入负样本。需要分别屏蔽正负对角线，例如：

```python
B = score_matrix.shape[0]
K = min(50, (B - 1) // 2)
if K < 1:
    raise ValueError('对比学习 batch 太小')
pos_matrix = score_matrix.clone()
neg_matrix = score_matrix.clone()
pos_matrix.fill_diagonal_(float('-inf'))
neg_matrix.fill_diagonal_(float('inf'))
top_values, top_indices = pos_matrix.topk(K, dim=1)
_, bottom_indices = neg_matrix.topk(K, dim=1, largest=False)
```

K=50 可作为论文 Table V 消融范围内的起点，但这不表示所有论文主结果都已核实采用 K=50。记录 K、batch、温度和对比损失权重；当前脚本内部还写死了对比损失权重、温度和重建项缩放，这些并非都能由命令行修改。应与所要复现的实验配置核对。

完成特征准备和上述修复后，可以复用第 7 节的 VQ 架构/优化参数，把入口换成 `train_vq_constra_100.py`、实验名换成 `VQ_amg`，单独保存输出。建议先跑短任务验证有效候选与梯度，再长训练。随后用 `VQ_amg` 重新生成 `VQ_amg_tokens/`、训练 `GPT_amg`，保持这一组 VQ/GPT 配对，重复第 9 节评估。

单条 TMR 编码核对命令（先修复 TMR `config.json` 中 normalizer、文本模型缓存等旧机器路径；下面的服务器根目录按实际位置调整）：

```bash
cd /path/to/AMG-T2M/reproduction/local/models/Retrival
python encode_motion.py \
  run_dir=/path/to/AMG-T2M/reproduction/local/models/Retrival/models/tmr_humanml3d_guoh3dfeats \
  npy=/path/to/AMG-T2M/reproduction/Dataset/HumanML3D/new_joint_vecs/000000.npy \
  ckpt_name=last device=cuda
python encode_text.py \
  run_dir=/path/to/AMG-T2M/reproduction/local/models/Retrival/models/tmr_humanml3d_guoh3dfeats \
  text="a man kicks something or someone with his left leg." \
  ckpt_name=last device=cuda
cd /path/to/AMG-T2M/reproduction/local
```

`run_dir` 是 TMR 模型和配置目录；`npy` 是未做 VQ 标准化的真实 263 维动作特征；TMR 使用自己的 normalizer。`ckpt_name=last` 读取 TMR 的 `last_weights`。动作结果保存在该 `run_dir/encoded/000000.npy`，文本结果在同一目录，以文本的小写下划线形式命名。当前配置的隐向量为 256 维。这两个命令只是核对单条接口，不会自动形成 VQ 训练的两种完整嵌入目录。

完整导出并修复 K/对角线/验证集后，对比 VQ 的正式命令为：

```bash
export PYTHONPATH="$PWD/models/Retrival${PYTHONPATH:+:$PYTHONPATH}"
python -u train_vq_constra_100.py \
  --dataname t2m --batch-size 256 --window-size 64 \
  --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 \
  --nb-code 512 --code-dim 512 --down-t 2 --depth 3 \
  --dilation-growth-rate 3 --quantizer ema_reset --vq-act relu \
  --commit 0.02 --loss-vel 0.5 --recons-loss l1_smooth \
  --eval-iter 1000 --save-iter 1000 --print-iter 200 \
  --out-dir output/hml3d --exp-name VQ_amg
```

这份脚本的定期训练状态还会保存到 `checkpoints_model/VQ_amg/model_<iter>.pth`；验证最优模型仍在 `output/hml3d/VQ_amg/net_best_fid.pth`。K、对比权重和温度由前述源码设置决定，当前没有对应有效命令行开关。后续 GPT 用第 8 节完整命令，只替换 VQ 路径为 `output/hml3d/VQ_amg/net_best_fid.pth`、`--vq-name VQ_amg_tokens`、`--exp-name GPT_amg`；测试也对应替换为同一组 VQ/GPT 路径。

参考：[TMR 官方说明](https://github.com/Mathux/TMR)，[T2M-GPT 基线训练说明](https://github.com/Mael-zys/T2M-GPT)。

### 10.2 长度预测器

已有 `models/MotionLenEstimator/LengthEst.py` 和 `LengthEstTrainer.py`，但 `run_order` 中的 `train_length_est.py` 不存在。

需要补训练入口和数据接口：

1. 输入真实 caption 及真实动作帧数；使用 train/val/test 的对应划分。
2. 修复 spaCy 英语模型的固定机器路径，加载服务器实际安装的模型；安装对应的 `torch_geometric` / spaCy 依赖，CLIP 指定本地权重。
3. 核对 CLIP 子词与 spaCy 词级依存图的节点对应。按论文使用 L1 训练，保证预测和标签的归一化/帧数单位一致。
4. 当前 trainer 有分布式 `all_reduce`、`rank/world_size` 假设，单 GPU 入口需要去掉或保护这些调用，不能直接当成独立单卡脚本运行。
5. 在验证集选择 checkpoint；测试报告还原到帧数后的 `mean(abs(pred_frames - true_frames))`。如果同时报告秒数，HumanML3D 用 20 fps 换算，并明确单位。

还需修复一个实际接口冲突：`LengthEst.py` 当前 `forward(caption_list, cap_len)` 只接收两个输入，而 trainer 调用 `estimator(total_caption[j], cap_lens[j], total_step)` 传了三个。必须按最终模型设计统一签名和输入，不能只新增一个启动文件就认定能训练。

这部分没有可直接照抄的完整训练命令。缺少入口不是 CUDA 安装问题。论文 Table V 的长度误差也不能用 GPT 自己产生 EOS 的长度差替代。

### 10.3 过渡生成器

模型在 `models/TransitionGenerator.py::StepWeightedTransformer`。当前 `train_transition.py` 内容与 GPT 训练脚本相同，没有实例化和训练这个类。

需要补真正的过渡训练：冻结选定 VQ，从真实动作获得 token 和码本向量，按论文随机遮盖/预测目标建立训练样本，用对应的真实码本索引计算交叉熵，确保输入没有目标泄漏。独立保存过渡 checkpoint。现有距离权重在历史长度为 1 时分母为 0，也要处理。

推理时接上“文本分段 → 长度预测 → GPT 生成各段 → 过渡 token → VQ 解码”。目前简单拼接多段生成只验证演示流程，不等于这条完整 AMG 管线。`run_order` 提及的 `gen_long_seq.py` 也不在当前项目中。

### 10.4 论文完整评估

HumanML3D 基础测试跑通后，分开执行：

- Table I：HumanML3D 完整测试集的生成指标。
- Table I 右侧：按官方测试集文本筛选超过 40/50 个词的子集；必须固定按 caption 还是按 motion ID 筛选，并记录样本数。不能随意拿现有 `long_test.txt` 代替。
- Table V：长度预测的 L1 误差，以及有/无 Weighted InfoNCE、有/无过渡模块的消融。
- 长动作：固定片段采样、顺序和随机种子，记录 Transition Fluency；Avg-MPJPE 使用论文的子采样协议，不能拿普通重建 MPJPE 代替。
- Table II：补齐 BABEL-TEACH 的真实预处理、对应统计和评估模型，再跑其独立训练/评估。当前 GPT 评估入口没有完整 BABEL 专用分支，不能只换一个 `--dataname` 就复现该表。

论文 Table I 的 AMG FID 为 `0.037±0.003`，这是完整方法在指定协议下的结果，不是基础 VQ 重建或 100 步测试的验收阈值。

## 11. 推荐的实际执行顺序

1. 在服务器确认环境和 GPU，通过第 3 节检查。
2. 上传同级 `Dataset/HumanML3D`，解压文本，通过第 4 节文件/划分检查。
3. 完成第 5 节基础 VQ、GPT 和评估适配，同步修改后的源码。
4. 独立 3 步 CUDA 检查 → 正式 VQ 100 步 → 完整 VQ 训练。
5. 固定 VQ → GPT 100 步 → 完整 GPT 训练。
6. 测试入口单次检查 → 20 次正式评估；保留配置、checkpoint、日志、指标和置信区间。
7. TMR 特征与对比 VQ → 重新编码/训练 GPT → 对照基础链路。
8. 补齐长度和过渡训练入口，执行长动作及 BABEL-TEACH 实验。

原始入口加载 checkpoint 多数只恢复模型参数，不等于精确恢复 optimizer、scheduler、迭代数、随机状态与 EMA 统计。长任务开始前应把恢复语义和 checkpoint 内容核对清楚，避免任务中断后误认为已完整续训。

## 12. 每个文件负责什么，哪些需要直接运行

下面链接均指向本机项目文件；服务器运行的是同步后的对应文件。除命令中明确指定的入口外，其余文件由入口自动 import，不需要挨个 `python` 执行。

| 文件 | 是否直接运行 | 输入与职责 | 输出/下游 |
|---|---|---|---|
| [train_vq.py](train_vq.py) | 是，第 6–7 节 | 读取真实动作窗口，训练基础 VQ，定期调用验证 | VQ 权重、日志 |
| [train_vq_constra_100.py](train_vq_constra_100.py) | AMG 扩展时运行 | 动作、caption、TMR 两种嵌入；加入加权对比损失 | AMG VQ 权重 |
| [train_vq_constra.py](train_vq_constra.py) | 对比训练的另一版本 | 候选数量和部分实现与 `_100` 不同 | 独立实验，不能混用配置 |
| [train_t2m_trans.py](train_t2m_trans.py) | 是，第 8 节 | 加载固定 VQ，先导出训练 token，再用文本条件训练 GPT | token 缓存、GPT 权重 |
| [VQ_eval.py](VQ_eval.py) | 是，第 9.1 节 | 测试真实动作 → VQ 重建 → 冻结评估器 | 重建指标与首轮坐标文件 |
| [GPT_eval_multi.py](GPT_eval_multi.py) | 是，第 9.2 节 | 测试文本 → GPT token → 同一 VQ 解码 → 冻结评估器 | 生成指标、均值和置信区间 |
| [scripts/run_local.py](scripts/run_local.py) | 仅作独立检查/演示 | `train-smoke` 检查反向传播；不是标准训练器 | 检查报告、演示动作 |
| [options/option_vq.py](options/option_vq.py) | 否 | 解析 VQ 训练、VQ 测试参数 | `args` |
| [options/option_transformer.py](options/option_transformer.py) | 否 | 解析 GPT 训练、GPT 测试参数 | `args` |
| [dataset/dataset_VQ.py](dataset/dataset_VQ.py) | 否 | 基础模式用 `VQMotionDataset_V1` 返回动作窗口和 caption；对比模式用 `VQMotionDataset` 返回四项 | VQ 训练 batch |
| [dataset/dataset_tokenize.py](dataset/dataset_tokenize.py) | 否 | 读取训练集完整动作，按 4 帧对齐、标准化，返回动作和 ID | 由 GPT 入口调用 `net.encode` |
| [dataset/dataset_TM_train.py](dataset/dataset_TM_train.py) | 否 | 读取 caption 和动作 token，添加 EOS/PAD；当前还读步骤 JSON | GPT 五项 batch |
| [dataset/dataset_TM_eval.py](dataset/dataset_TM_eval.py) | 否 | 验证/测试配对样本、GloVe、词性、真实长度 | 评估八项 batch |
| [models/vqvae.py](models/vqvae.py) | 否 | `HumanVQVAE`；编解码、量化、提取序列表示 | 重建、commitment、PPL、表示 |
| [models/encdec.py](models/encdec.py) | 否 | 一维卷积编码器/解码器，时间降采样/上采样 | 连续隐变量和动作特征 |
| [models/quantize_cnn.py](models/quantize_cnn.py) | 否 | `ema_reset` 码本最近邻、EMA 更新、低使用码本重置 | 离散索引、量化向量、PPL |
| [models/t2m_trans.py](models/t2m_trans.py) | 否 | 文本条件的因果自回归模型，`forward` 和 `sample` | 下一个 token logits、生成序列 |
| [utils/losses.py](utils/losses.py) | 否 | VQ 重建及局部关节位置子集的损失 | 标量损失 |
| [models/evaluator_wrapper.py](models/evaluator_wrapper.py) | 否 | 加载冻结文本/动作评估编码器 | 文本和动作的 512 维评估嵌入 |
| [utils/eval_trans.py](utils/eval_trans.py) | 否 | 验证、测试、FID/R-precision/多样性等公式 | 指标和验证 checkpoint |
| [utils/motion_process.py](utils/motion_process.py) | 否 | `recover_from_ric` 将反标准化 263 维表示恢复为关节坐标 | `(T,22,3)` 可视化/几何分析 |
| [models/Retrival/encode_motion.py](models/Retrival/encode_motion.py) / [encode_text.py](models/Retrival/encode_text.py) | AMG 时做单条编码核对 | TMR 模型与动作/文本 | TMR 隐向量，需另接批量导出 |
| [models/MotionLenEstimator/LengthEst.py](models/MotionLenEstimator/LengthEst.py) | 否 | caption、文本长度 → CLIP/依存图/GAT | 归一化动作长度预测 |
| [models/MotionLenEstimator/LengthEstTrainer.py](models/MotionLenEstimator/LengthEstTrainer.py) | 否 | 训练器类，当前有数据接口和 DDP 假设 | 长度模型 checkpoint |
| [models/TransitionGenerator.py](models/TransitionGenerator.py) | 否 | 历史码本向量 `[B,T,512]` → 注意力与距离加权 | 下一过渡 token 的 `[B,512]` logits |
| `train_transition.py` (removed duplicate) | 当前不要作为过渡训练入口 | 实际复制了 GPT 训练代码 | 尚未接上过渡模型 |
| `train_GPT_split.py` (removed empty file) | 否 | 当前为空文件 | 无 |

`train_vq_multi.py`、`train_vq_constra_multi.py` 是另一些多卡实验入口。先跑单卡链路时不用同时启动它们。`run_order` 是历史命令笔记，含不存在的文件和拼写不一致的参数，不应逐行当可执行脚本运行。

## 13. 数据和模型之间的形状：为什么需要先 VQ 后 GPT

### 13.1 动作文件不是普通的 xyz 坐标

HumanML3D 的 `new_joint_vecs/000000.npy` 为 `[T,263]`，20 fps。当前特征布局为：

| Python 切片 | 维数 | 内容 |
|---|---:|---|
| `0:1` | 1 | 根节点朝向角速度 |
| `1:3` | 2 | 根节点地面平面线速度 |
| `3:4` | 1 | 根节点高度 |
| `4:67` | 63 | 其余 21 个关节的局部位置 |
| `67:193` | 126 | 21 个关节的 6D 旋转 |
| `193:259` | 66 | 22 个关节的局部速度 |
| `259:263` | 4 | 足部接触标记 |

读入后计算 `(motion - mean) / std`。解码结果先乘 std 加 mean，之后才可恢复到 xyz 坐标。Mean、Std 是统计文件，不能把它们当动作文件喂给模型。

原始文本一行形如：

```text
a man walks forward.#a/DET man/NOUN walk/VERB forward/ADV#0.0#0.0
```

四段依次为 caption、词/POS、片段开始秒数、片段结束秒数。`0.0#0.0` 表示整段动作描述；其它时间戳对应局部片段。`M000000` 是镜像版本，左右词和动作应保持对应。

### 13.2 基础 VQ 的一次训练

```text
原始动作 [T,263]
→ 随机取 64 帧（3.2 秒）并标准化
→ batch [256,64,263]
→ 卷积编码器 [256,512,16]
→ 最近邻量化到 512 个码本向量
→ 解码重建 [256,64,263]
→ 与输入计算损失并更新参数/EMA 码本
```

当前 `down-t=2`、`stride-t=2` 表示两次 stride=2 的时间降采样，共缩短为四分之一；一个 token 对应约 4 帧，即 0.2 秒。解码器每层固定放大 2 倍，因此先保持 `stride-t=2`，不要随意改成 3。

`HumanVQVAE.forward()` 返回四项：`(x_out, loss_commit, perplexity, quant_repr)`。最后一项在本分支已经是量化序列在时间维平均后的 `[B,512]`，不是完整 `[B,512,T']` 序列；对比训练使用这一表示。定义了 `projection_layer`，但当前 forward 中没有使用它。

### 13.3 token 导出和 GPT 的一次训练

token 导出时读取完整训练动作，对齐到 4 的整数倍后，通过固定 VQ 的 `encode()` 生成 `[1,L]` 整数数组，保存为 `<id>.npy`。例如 196 帧约对应 49 个动作 token。

动作 token 范围 `0..511`；EOS 为 `512`，PAD 为 `513`。PAD 只用于补齐 batch，不能作为动作码送入 VQ 解码。GPT 的输出类别数为 513，预测动作码和 EOS，不预测 PAD。

```text
caption → 冻结 CLIP ViT-B/32 → [B,512] 文本条件
动作 token + EOS + PAD → [B,51]
→ 输入 token 的最后一项去掉，部分动作码随机替换
→ 文本条件映射到 GPT 1024 维并作为首位置
→ 因果 Transformer → [B,51,513] logits
→ 只在有效动作长度 + EOS 的位置计算交叉熵
```

`block-size=51` 是 token/条件位置容量，不是 51 帧，更不是任意长动作的总容量。当前单段生成仍受窗口限制；完整长动作需要分段和过渡管线。

### 13.4 评估加载器的八项返回值

`word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token_string, name`：

- GloVe 词向量为 300 维；词性 one-hot 由 `utils/word_vectorizer.py` 定义。
- 评估文本最多保留 20 个词，加 SOS/EOS 后序列长 22；这里与生成器 CLIP 的 tokenizer 是两套输入。
- `motion` 补到 196 帧，`m_length` 表示有效长度，补零部分不应被当成真实动作。
- 当前加载器对 40–199 帧动作做筛选、随机 caption、随机长度对齐/裁剪，且 `shuffle=True, drop_last=True`。因此实际评估条数和 ID 总数可能不同，要记录 `nb_sample`。
- 对长于 40/50 词的子集比较，评估编码器仍有 20 词截断；必须和对照方法使用相同协议，不能把这一限制隐藏。

## 14. 运行命令中的每个关键参数

### 14.1 VQ 参数：对应 `options/option_vq.py`

以下为第 7 节命令的有效配置，未显式写在命令中的值来自当前 parser 默认值。

| 参数 | 使用值 | 作用及注意事项 |
|---|---|---|
| `--dataname` | `t2m` | HumanML3D 分支，22 关节、263 维 |
| `--batch-size` | 正式 256，检查 32 | 每次优化的动作窗口数 |
| `--window-size` | 64 | 每个训练窗口帧数；不是 token 数 |
| `--total-iter` | 正式 300000，检查 100 | 正式优化步数，不是 epoch 数；warm-up 另计 |
| `--warm-up-iter` | 默认 1000，检查 10 | 学习率预热；基础脚本 `range(1,warm_up_iter)` 实际少一步 |
| `--lr` | `2e-4` | AdamW 初始正式学习率 |
| `--lr-scheduler` | `200000` | 第 20 万正式优化步衰减 |
| `--gamma` | 默认 `0.05` | 衰减乘数，衰减后 LR 为 `1e-5` |
| `--weight-decay` | 默认 0 | 基础 VQ AdamW 的权重衰减 |
| `--commit` | 默认 0.02 | commitment loss 的外部系数 |
| `--loss-vel` | 0.5 | 当前实现对 `4:67` 局部位置子集加权，不能仅凭名称解释成速度差分 |
| `--recons-loss` | `l1_smooth` | PyTorch SmoothL1Loss；不是普通 L1，也不是 MPJPE |
| `--nb-code` | 512 | 码本向量个数，即动作离散词表大小 |
| `--code-dim` | 512 | 每个码本向量维数 |
| `--output-emb-width` | 默认 512 | 编码器输出通道数；与当前量化维数匹配 |
| `--width` | 默认 512 | 编解码器内部通道宽度 |
| `--down-t` | 2 | 时间降采样层数，不是直接除以 2 |
| `--stride-t` | 默认 2 | 每层时间步长；与当前解码器的 2 倍上采样配套 |
| `--depth` | 3 | 各残差堆叠深度 |
| `--dilation-growth-rate` | 3 | 残差卷积膨胀率增长倍率 |
| `--vq-act` | `relu` | VQ 残差模块激活 |
| `--vq-norm` | 默认 None | 当前配置不额外指定该归一化 |
| `--quantizer` | `ema_reset` | EMA 更新码本并重置低使用向量 |
| `--mu` | 默认 0.99 | EMA 动量 |
| `--print-iter` | 正式 200，检查 10 | 每多少步打印平均训练损失 |
| `--eval-iter` | 正式 1000，检查 50 | 每多少步验证，同时更新/保存 checkpoint |
| `--save-iter` | 基础脚本不使用 | parser 有此参数，但 `train_vq.py` 实际保存跟随验证；对比版本另有保存逻辑 |
| `--seed` | 默认 123 | 当前入口主要设置 torch 随机种子，详见第 16 节 |
| `--out-dir` | `output/hml3d` | 输出父目录，按服务器项目根目录解析 |
| `--exp-name` | `VQ_base` | 输出子目录，最终为 `output/hml3d/VQ_base/` |
| `--resume-pth` | 从零训练不传 | 若传则加载 VQ 的 `ckpt['net']`；基础入口不恢复完整训练状态 |

基础 VQ 的 AdamW 在源码设置 `betas=(0.9,0.99)`。`--beta` 是另一类量化器的选项，本方案使用 `ema_reset`，不要把它误当 `--commit`。`--results-dir`、`--visual-name`、`--resume-gpt` 等并未构成当前基础训练的主要控制链路。

### 14.2 GPT 参数：对应 `options/option_transformer.py`

| 参数 | 使用值 | 作用及注意事项 |
|---|---|---|
| `--dataname` | `t2m` | HumanML3D |
| `--resume-pth` | `output/hml3d/VQ_base/net_best_fid.pth` | 用于编码 token 和解码的 VQ；必须固定 |
| `--resume-trans` | 从零训练不传 | 加载 GPT 的 `ckpt['trans']`；不是 VQ |
| `--vq-name` | `VQ_base_tokens` | 数据根目录内的 token 缓存目录名，不是 checkpoint 路径 |
| `--batch-size` | 正式 128，检查 32 | 每个训练 batch 的文本动作对数 |
| `--embed-dim-gpt` | 1024 | GPT 隐藏维数 |
| `--clip-dim` | 默认 512 | 冻结 CLIP 文本输出维数 |
| `--num-layers` | 9 | 当前实现 base 和 head **各建 9 个 Block，共 18 个**，不是整个网络只有 9 个 Block |
| `--n-head-gpt` | 16 | 多头注意力头数；1024/16=64 维每头 |
| `--block-size` | 51 | 上下文位置容量，需与 token 长度、下采样率配套 |
| `--ff-rate` | 4 | FFN 隐层是 GPT 维数的 4 倍，即 4096 |
| `--drop-out-rate` | 0.1 | Transformer dropout，评估时 eval 模式关闭 |
| `--pkeep` | 0.5 | 训练输入 token 保留概率；其余替换为随机动作码，目标标签保持原序列 |
| `--lr` | `1e-4` | GPT 学习率 |
| `--lr-scheduler` | 150000 | 第 15 万步学习率衰减 |
| `--gamma` | 默认 0.05 | 衰减后 LR 为 `5e-6` |
| `--optimizer` | 默认 `adamw` | 当前 `utils_model.initial_optim` 使用 `betas=(0.5,0.9)` |
| `--weight-decay` | 默认 `1e-6` | GPT 权重衰减 |
| `--decay-option` | 默认 `all` | 使用当前全参数分支；`noVQ` 分支引用不适用于此 GPT 的成员，先不要切换 |
| `--total-iter` | 300000 | GPT 优化步数，不含前面的 token 编码耗时 |
| `--eval-iter` | 10000 | 验证和 checkpoint 更新间隔 |
| `--print-iter` | 200 | 训练损失及 token ACC 日志间隔 |
| `--out-dir` / `--exp-name` | `output/hml3d` / `GPT_base` | 输出为 `output/hml3d/GPT_base/` |
| VQ 架构参数 | 与固定 VQ 相同 | `nb-code=512, code-dim=512, down-t=2, stride-t=2, width=512, depth=3, dilation-growth-rate=3` |

当前 GPT 构建 VQ 时没有传入 `args.vq_act/vq_norm`，实际使用构造默认值 ReLU/None，所以本方案保持这个设置。如果训练 VQ 时改过激活或归一化，需要同时修改 GPT 和测试的 VQ 构造，单改命令行不够。

parser 中 `--warm-up-iter`、`--seq-len`、`--fps` 不代表该 GPT 入口已实现相应 warm-up/动态长度/动态帧率控制；核心加载器仍有固定长度和 20 fps 逻辑。`--if-maxtest` 影响训练日志中 token ACC 的取法，不是最终动作测试开关。

参数名使用连字符 `--resume-trans`。历史笔记里的 `--resume_trans` 不属于当前有效 GPT 参数。`--resume_pth` 又与 `--resume-pth` 共用 argparse 目标名，保留历史重复定义；本方案始终显式使用 `--resume-pth` 并提供文件路径。

### 14.3 评估时哪些参数有效

评估只做前向传播；真正需要的是 `--dataname`、模型结构、VQ/GPT checkpoint、输出目录和种子。`lr/total-iter/lr-scheduler/pkeep` 即使能被共享 parser 接受，也不会让测试执行对应训练行为，因此第 9 节命令省掉它们。

当前 `repeat_time=20`、评估 batch=32、每文本采样 30 次、多模态距离采样数 10 是源码设置，没有现成的对应命令行参数。若要改变，应改明确位置并记录，不要在命令中杜撰 `--repeat-time` 或 `--multimodality-times`。

## 15. 训练日志的每个数值是什么

### 15.1 VQ 的优化目标

本方案基础 VQ 实际计算：

`L = SmoothL1(x_hat, x) + 0.02*MSE(z_e, stop_grad(z_q)) + 0.5*SmoothL1(x_hat[...,4:67], x[...,4:67])`

第一项覆盖 263 维标准化动作；第二项约束编码器输出接近所选码本向量；第三项在当前代码中强调局部关节位置。`forward_vel` 这个函数名不等于其内部做了时间速度差分。

| 日志名 | 计算含义 | 怎样读 |
|---|---|---|
| `Recons.` / TensorBoard `Train/L1` | 按当前 `recons-loss` 的重建项，本方案是 SmoothL1 | 同一标准化和同一 loss 配置下比较，低通常更好；名称 L1 不代表一定是普通 L1 |
| `Commit.` | commitment MSE 的原始数值，打印时尚未乘 0.02 | 不是最终总损失 |
| `PPL.` | `exp(-sum(p_k*log(p_k)))`，p 为 batch 中的码本使用频率 | 约在 1 到 512；接近 1 表示集中使用极少码本，不能单独以越高越好选模型 |
| `lr` | 当前优化器学习率 | 检查预热和衰减是否符合设置 |
| `Contrastive` | 仅对比版本有，Weighted InfoNCE | 与重建、commitment 共同优化，不是 FID |

对比 `_100.py` 正式训练当前写为 `4*L_recon + commit*L_commit + loss_vel*L_subset + 0.01*L_contrastive`，温度写死为 0.07。它和基础 VQ 的重建项系数不同，因此比较时不仅要记录“是否加对比损失”，还要控制其它权重。TMR 相似度使用归一化余弦的 `cos/2+0.5`；正样本权重为相似度除以温度后的 softmax。

### 15.2 GPT 的优化目标

对每条样本，只统计有效动作 token 和 EOS 的交叉熵，忽略其后的 PAD；先在单条有效序列上平均，再对 batch 平均：

`L_CE = (1/B) * sum_i mean_t[-log P(target_i,t | corrupted_prefix_i,<t, text_i)]`

训练日志 `Loss.` 是交叉熵；`ACC.` 是逐 token 预测准确率，以百分数输出，不是 R-precision。当前默认按分类分布采样后算训练 ACC，传 `--if-maxtest` 才改成 argmax，所以不要拿两个设置的 ACC 直接比较。能预测 token 不等于解码动作一定自然，最终仍需生成后评估。

### 15.3 训练完成的最后一行不能代替最终测试

训练保存和最后一行会跟踪不同时间的历史最佳 FID、Diversity、Top-k 等；这些最佳值不一定出自同一个 checkpoint。要报告一份模型的完整结果，必须重新加载选定的 VQ/GPT，在测试集上一次性计算全部指标。

## 16. 指标究竟怎样计算，输出文件怎样看

### 16.1 先区分三种文本/动作表示

- **CLIP**：给 GPT 提供文本条件，输出 512 维。
- **TMR**：在 AMG 对比训练中估计文本与动作的语义相似度，当前随附 TMR 配置的隐空间为 256 维。
- **冻结评估器**：`models/evaluator_wrapper.py` 加载 `text_mot_match/model/finest_1.tar`，以 GloVe+词性编码文本、卷积+BiGRU 编码动作，得到 512 维共同嵌入，用于下述 FID 和检索指标。

维数相同也不表示是同一个模型。正式测试不能把 TMR/CLIP 分数代替项目标准评估器输出。评估动作编码器当前去掉最后 4 维足部接触标记，实际先编码 259 维输入，并根据有效动作长度处理序列。

### 16.2 FID：比较生成动作和真实动作的特征分布

记真实动作评估嵌入的均值、协方差为 `mu_r, Sigma_r`，生成嵌入为 `mu_g, Sigma_g`：

`FID = ||mu_r-mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrtm(Sigma_r*Sigma_g))`

实现在 `utils/eval_trans.py::calculate_frechet_distance`。越低说明特征分布越接近；它不是单个动作的关节坐标误差，也不是训练 loss。小样本 FID 波动很大，不能用几条动作推断论文级结果。

### 16.3 R-precision：在 32 个候选动作中检索

对一个 batch，计算每个文本嵌入与 32 个动作嵌入的欧氏距离矩阵，按距离升序排候选。正确配对动作进入前 k 名记 1，否则记 0，对全部测试样本平均，得到 Top1、Top2、Top3。

输出范围为 0 到 1，例如 Top3=0.78 表示约 78% 的配对动作进入前三。改变 batch 大小就改变候选难度，因此正式比较保持 32；训练中的 token ACC 与它没有换算关系。

### 16.4 Matching Score / MM-Dist：正确配对的嵌入距离

`MM-Dist = (1/N) * sum_i ||e_text_i - e_motion_i||_2`

源码对距离矩阵取 trace 再除样本数。数值越低越匹配。这里的 MM-Dist 不是 Multimodality，也不是毫米单位的几何误差。

### 16.5 Diversity：不同输入生成动作之间的差异

从动作评估嵌入中分别抽取两组索引，计算配对的欧氏距离并求平均。当前样本数大于 300 时抽 300 对，否则抽 100 对，且代码要求总样本数严格大于抽样数。过小的验证子集会断言失败。

同时输出 `Diversity Real` 和 `Diversity`。应结合真实分布比较；随机噪声也可能让差异变大，所以不能单独以越大越好判断。当前 VQ 的 `net_best_div.pth` 按接近真实 Diversity 的程度保存。

### 16.6 Multimodality：同一文本的多种生成

接通 `evaluation_transformer_test()` 后，每条 caption 随机采样 30 份动作，得到 `[N,30,512]` 评估嵌入；再从 30 个生成索引分别取两组 10 项，计算两组对应动作嵌入距离，对文本和抽样项求平均。

它回答同一句话是否能产生不同合理动作。使用普通 `evaluation_transformer()` 的贪心单次生成无法提供这个结果；当前原始 `GPT_eval_multi.py` 的空 `multi` 列表必须修复。这个函数在每轮评估的第一次生成上计算 FID/检索/Matching，30 次生成共同计算 Multimodality，避免把所有 30 倍样本混成另一个 FID 协议。

### 16.7 重复 20 次与 95% 置信区间

每轮都对选定测试集执行评估得到一个指标值 `m_r`，最后按当前项目公式：

`mean = mean(m_1,...,m_20)`

`CI95 = 1.96 * std(m_1,...,m_20) / sqrt(20)`

当前源码 `np.std` 使用默认 `ddof=0`。日志如 `FID. 0.050, conf. 0.003` 表示按该公式报告 `0.050±0.003`，不表示第 20 轮单独达到 0.050。

完整复现需同时控制 Python `random`、NumPy、PyTorch 的种子。现有若干入口只设置 `torch.manual_seed`，不足以固定 caption 和随机裁剪。建议每轮明确使用 `base_seed + repeat_index` 设置三者，并记录生成方式、数据加载 worker、实际样本数；不要每轮都重设成完全相同的种子而把相同抽样复制 20 次。

### 16.8 MPJPE、长度误差和 Transition Fluency

- 普通 MPJPE 是关节坐标的平均欧氏误差：`mean_{t,j}(||p_pred[t,j]-p_gt[t,j]||_2)`。是否平移/旋转对齐、是否包含根位移、坐标单位是否转毫米，必须另外说明。
- 论文 Table III 对超过 190 帧的动作使用 25–35 帧子片段，在 3/5/7 个片段的设置下计算 Avg-MPJPE。现有基础 VQ/GPT 测试脚本没有接好这个实验，不应把普通重建 MPJPE 标成 Table III 的结果。
- 长度预测评估用帧数单位的平均绝对误差。若 MAE=20 帧，在 20 fps 下相当于平均 1 秒误差；这不是生成质量的 FID。
- 论文采用 ST2M 的 Transition Fluency，但当前项目没有找到与论文完整设置接好的计算入口，所给 PDF 也没有展开其精确公式和边界处理。需要获取对应评估定义/代码并核对；不能临时定义相邻帧速度差就声称复现该指标。

### 16.9 每一步留下什么文件

```text
T2M-GPT-main/
└── output/
    ├── hml3d/
    │   ├── VQ_check100/
    │   ├── VQ_base/
    │   │   ├── run.log
    │   │   ├── events.out.tfevents.*
    │   │   ├── net_best_fid.pth
    │   │   ├── net_best_div.pth          # 对应指标改善时保存
    │   │   ├── net_best_top1.pth         # 对应指标改善时保存
    │   │   ├── net_best_matching.pth     # 对应指标改善时保存
    │   │   └── net_last.pth
    │   ├── GPT_check100/
    │   └── GPT_base/
    │       ├── run.log
    │       ├── events.out.tfevents.*
    │       ├── net_best_fid.pth
    │       └── net_last.pth
    └── hml3d_eval/
        ├── VQ_base_test/
        │   ├── run.log
        │   ├── <name>_gt.npy            # 当前 VQ_eval 首轮开启 savenpy
        │   └── <name>_pred.npy
        └── GPT_base_test/
            └── run.log

Dataset/HumanML3D/
└── VQ_base_tokens/
    ├── 000000.npy
    └── ...
```

VQ/GPT checkpoint 同名不表示能互换：VQ 文件主要内容是 `{'net': state_dict}`，GPT 是 `{'trans': state_dict}`。`net_best_fid` 是验证 FID 最优的模型；`net_last` 是最近一次触发保存时的模型，若训练步数不整除验证间隔，不保证是训练最后一步。

第 9 节测试以 `save=False` 调评估函数，不产生新的训练权重，也不会修正输入权重。按本指南修正的 GPT 测试默认 `savenpy=False`，所以主要结果是 `run.log`；要导出动作，需要在该函数调用中显式开启 `savenpy`。保存的关节坐标通常为 `[1,T,22,3]`，用途是可视化和几何分析，不是 token 或 263 维训练特征。

第一次报告结果时，至少保留：完整运行命令、修复后的源码版本/备份、实际有效样本数、模型结构、固定 VQ 与 GPT 的 SHA256、随机种子、每轮指标、最终均值/置信区间，以及是否启用长度和过渡模块。检查输出目录中有文件只是完成条件之一，还需确认评估没有吞掉生成错误，指标均为有限数值。
