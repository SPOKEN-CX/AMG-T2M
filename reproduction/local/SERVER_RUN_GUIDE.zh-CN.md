> 当前安装入口：在仓库根目录执行 `python -m pip install -r requirements.txt`，环境说明见 `../../docs/environment.md`。下文服务器配置为历史运行记录。

# T2M-GPT 服务器运行说明

适用日期：2026-10-21。本文依据当前项目代码，以及会话“检查训练与实验流程”中你最后提供的服务器终端输出编写。

## 1. 你的服务器环境已经搭好

| 项目 | 已确认信息 |
|---|---|
| 用户 / 主机 | `user@server.example` |
| 系统 | Ubuntu 18.04.6 LTS |
| 项目目录 | `/path/to/AMG-T2M/reproduction/local` |
| Conda安装位置 | `/opt/conda` |
| 已建好的环境 | `/path/to/envs/t2mgpt` |
| Python | 3.10 |
| PyTorch | **2.0.1+cu117** |
| PyTorch CUDA运行时 | **11.7** |
| GPU | NVIDIA A100 80GB PCIe |
| 已通过的检查 | torch、numpy、scipy、clip、pytorch_lightning、hydra、tensorboard导入；CUDA矩阵运算 |

以这个最终成功状态为准，之前会话中讨论过的其他Python/CUDA安装方案不再作为执行步骤。**不用重新安装环境，不用重新下载PyTorch。** 已解决的 `pkg_resources` 问题不需要重复处理；当时的弃用警告不妨碍已经通过的运行检查。

本文中的GPU命令让你在服务器执行。我没有登录服务器执行它们；上一次生成/重建/反向传播的实测发生在Mac的独立环境中，因此不能把那些结果当作服务器CUDA实测。

## 2. 目前先运行什么

你目前没有完整HumanML3D数据，建议按以下顺序操作：

1. 官方预训练权重：一句文字生成动作。
2. 本地 `self_train` 权重：生成动作，核对权重能否加载。
3. 现有 `motions/` 样例：VQ重建、3步训练检查。
4. 三段文字：生成长动作。
5. 完整数据到位、训练入口修复后：正式VQ训练 → GPT训练 → 标准评估。

前四项不需要完整HumanML3D，也不需要 `texts_retrival`、`motion_retrival`、`texts_GPT_processed`。第3项的小样本训练仅验证计算链路，不是论文训练实验。

## 3. 先把本次新增入口同步到服务器

你之前上传项目的时间早于本会话新增文件的时间，服务器上可能还没有 `scripts/run_local.py`。

请在VS Code的SSH远程窗口中，将下面文件上传至对应位置；不必重传整个项目：

| 本机文件（相对项目根目录） | 服务器目标 |
|---|---|
| `scripts/run_local.py` | `/path/to/AMG-T2M/reproduction/local/scripts/run_local.py` |
| `models/quantize_cnn.py` | `/path/to/AMG-T2M/reproduction/local/models/quantize_cnn.py` |
| `SERVER_RUN_GUIDE.zh-CN.md` | `/path/to/AMG-T2M/reproduction/local/SERVER_RUN_GUIDE.zh-CN.md` |

其中 `quantize_cnn.py` 的本次变更只有两处：码本buffer初始化移除强制 `.cuda()`，设备交由整个模型的 `.to(device)` 管理。若服务器这份文件已有你另外修改的内容，应只合入这两处变更，保留服务器的其他改动。新增入口在CUDA上也能使用原来的强制CUDA码本；此文件同步是为了保持两端版本一致。

新入口名字虽为 `run_local.py`，它是直接运行项目模型的入口，**支持服务器CUDA**，不是仅供Mac使用。

不要上传本机 `.venv-smoke/`，不要在服务器安装 历史 Mac 最小依赖清单（已归档） 来替换已经搭好的环境。这份依赖清单对应上一轮的Mac独立验证。

如果你习惯使用终端传输，可以在Mac上运行下面命令。把 `n1` 换成你平时能实际连接的SSH地址或别名；若VS Code使用跳板机，按原有SSH配置连接即可：

```bash
ssh user@server.example 'mkdir -p /path/to/AMG-T2M/reproduction/local/scripts'

scp /path/to/AMG-T2M/reproduction/local/scripts/run_local.py \
  user@server.example:/path/to/AMG-T2M/reproduction/local/scripts/run_local.py

scp /path/to/AMG-T2M/reproduction/local/SERVER_RUN_GUIDE.zh-CN.md \
  user@server.example:/path/to/AMG-T2M/reproduction/local/SERVER_RUN_GUIDE.zh-CN.md
```

上面的传输命令没有覆盖 `quantize_cnn.py`；该文件按前述说明核对差异后合入即可。后续始终使用服务器已有的 `t2mgpt` 环境。

## 4. 每次连接服务器后的准备

以下命令均在 **VS Code的SSH远程终端 / 服务器终端**执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /path/to/envs/t2mgpt
cd /path/to/AMG-T2M/reproduction/local

export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
```

GPU编号按你实际使用的GPU调整；设置后代码中的 `cuda:0` 指当前可见的第一张GPU。`MPLBACKEND=Agg` 用于服务器无桌面环境下绘图。

首次运行新增入口前，检查资源和额外绘图依赖是否在服务器上：

```bash
python - <<'PY'
from pathlib import Path
import sys
import torch
import clip
import matplotlib
from PIL import Image

print('Python:', sys.executable)
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), '当前终端没有可用CUDA，请检查环境和可见GPU'
print('GPU:', torch.cuda.get_device_name(0))

required = [
    'scripts/run_local.py',
    'pretrained/VQVAE/net_best_fid.pth',
    'pretrained/VQTransformer_corruption05/net_best_fid.pth',
    'ViT-B-32.pt',
    'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy',
    'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy',
    'motions/000090.npy',
]
missing = []
for name in required:
    p = Path(name)
    ok = p.is_file() and p.stat().st_size > 0
    print('[OK]' if ok else '[MISSING]', name)
    if not ok:
        missing.append(name)
assert not missing, '请只补传上面缺失的文件，再继续'
print('预检查通过')
PY
```

这一步是在确认文件同步情况，不是重新搭环境。若失败，按缺失文件或实际导入报错处理，不要直接升级整套依赖。

## 5. 现在就能执行的命令

### 5.1 官方权重：一句文字生成动作

建议先不导出GIF，先确认GPU推理和权重加载：

```bash
python scripts/run_local.py generate \
  --device cuda \
  --text 'a person walks forward' \
  --no-gif
```

成功标志：

- 输出 `device=cuda`；
- 输出实际生成的token数量；
- 最终输出含 `xyz_shape`、`segment_tokens` 的JSON；
- 打印的输出目录下有 `motion.npy`、`features.npy`、`tokens.npy`、`result.json`。

程序默认使用一对官方模型：

```text
pretrained/VQVAE/net_best_fid.pth
pretrained/VQTransformer_corruption05/net_best_fid.pth
```

CLIP直接读取根目录的 `ViT-B-32.pt`，不会自动再下载同一份权重。

要直接生成动画，去掉 `--no-gif`：

```bash
python scripts/run_local.py generate \
  --device cuda \
  --text 'a person walks forward'
```

新增的 `motion.gif` 是22关节骨架动画，不需要SMPL人体模型或ffmpeg。服务器生成后可以通过VS Code文件浏览器下载到Mac查看。

### 5.2 使用你的self_train权重

先确认这两个文件存在，再执行：

```bash
python scripts/run_local.py generate \
  --device cuda \
  --text 'a person is jumping' \
  --vq checkpoints/self_train/VQVAE/t2m/net_best_fid.pth \
  --gpt checkpoints/self_train/GPT/net_best_fid.pth
```

这里使用原 `motion_gen.py` 指定的同一对本地训练权重。张量加载成功只说明结构兼容；不证明训练已收敛，也不证明它优于官方模型。

不要随意把某个VQ和另一轮GPT混搭。GPT生成的是码本索引，VQ更新后索引对应的含义可能改变。

### 5.3 VQ重建现有动作

```bash
python scripts/run_local.py reconstruct \
  --device cuda \
  --source motions/000090.npy
```

输出包括：

```text
original.npy / original.gif
reconstruction.npy / reconstruction.gif
reconstruction_features.npy
tokens.npy
result.json
```

`motions/000090.npy` 是125帧263维原始特征。VQ下采样倍数为4，程序将尾部不足4帧的部分截去，所以此例为124帧 → 31 tokens → 124帧。`raw_feature_mae` 是这个样例的特征重建误差，不是FID。

### 5.4 检查GPU上能否反向传播

```bash
python scripts/run_local.py train-smoke \
  --device cuda \
  --source motions \
  --steps 3
```

这条命令实际执行：加载官方VQ → 读取现有263维样例 → 取64帧窗口，batch=2 → 计算重建/commitment损失 → backward → optimizer.step → 保存单独权重。

成功标志：打印3个step；`result.json` 中 `encoder_max_weight_change > 0`；保存 `smoke_vq.pth`。程序也会检查loss和梯度是否有限。

本机的11个 `motions/` 文件中有10个满足64帧窗口；如果服务器上传了相同文件，可用数量应一致。不要把 `--steps` 简单改为300000当成正式实验：这里没有独立验证集、正式指标、完整训练调度或GPT训练，也不保证损失下降。

### 5.5 三段文字生成长动作

```bash
python scripts/run_local.py generate \
  --device cuda \
  --text 'a person is jumping' \
  --text 'a person walks forward' \
  --text 'a person sits down'
```

当前实现是逐段生成token、拼接、统一VQ解码。**没有使用训练好的过渡网络，段间自然连续性需要另行评价。** 每段最多50 tokens，每token对应4帧，播放速度20fps；EOS可以提前结束，所以这不是精确时长控制。

### 5.6 只渲染已有坐标文件

```bash
python scripts/run_local.py preview --source motion.npy
```

也可以指定新生成的坐标，例如把上一条运行输出目录中的完整 `motion.npy` 路径传给 `--source`。此模式不加载GPT/VQ，直接输出骨架GIF。

## 6. 输出与日志怎么找

未指定 `--out` 时，每次创建一个新目录：

```text
/path/to/AMG-T2M/reproduction/local/output/local_runs/
└── generate_日期_时间/
    ├── motion.npy
    ├── motion.gif          # 仅未指定 --no-gif 时
    ├── features.npy
    ├── tokens.npy
    └── result.json
```

`local_runs` 是脚本默认输出目录名，在服务器运行时结果仍保存在服务器，不会保存到Mac。`motion.npy` 形状是 `(1,T,22,3)`；`features.npy` 是恢复归一化后的 `(T,263)`；`tokens.npy` 是 `(1,L)`。

可用 `--out output/server_runs/你自己的新实验名` 指定位置。目录必须尚不存在，以免覆盖旧结果；重跑时换一个新实验名。

记录前台日志的例子：

```bash
mkdir -p logs
set -o pipefail

python -u scripts/run_local.py generate \
  --device cuda \
  --text 'a person walks forward' \
  --no-gif 2>&1 | tee logs/server_generate.log
```

查看GPU：

```bash
nvidia-smi
```

首次建议前台运行，直接观察报错。若后续需要断开SSH后继续执行，可以使用服务器已有的tmux，或以下后台方式；只结束自己确认过PID的任务：

```bash
mkdir -p logs
nohup python -u scripts/run_local.py generate \
  --device cuda \
  --text 'a person walks forward' \
  > logs/server_generate_background.log 2>&1 &

echo $!
tail -f logs/server_generate_background.log
```

`tail -f` 中按Ctrl+C只是退出日志查看，不会终止后台进程。

## 7. 正式HumanML3D训练前需要补齐什么

### 7.1 数据目录以现有代码为准

```text
/path/to/AMG-T2M/reproduction/
├── T2M-GPT-main/
│   ├── train_vq.py
│   ├── train_t2m_trans.py
│   ├── checkpoints/
│   └── glove/
└── Dataset/
    └── HumanML3D/
        ├── new_joint_vecs/         # (T,263) 原始特征
        ├── texts/                  # caption#词/POS#起始秒#结束秒
        ├── Mean.npy
        ├── Std.npy
        ├── train.txt
        ├── val.txt
        ├── test.txt
        ├── train_val.txt
        └── all.txt
```

`Dataset` 的D大写，位于项目外，与 `T2M-GPT-main` 同级。当前 `dataset_VQ.py`、`dataset_tokenize.py`、`dataset_TM_train.py`、`dataset_TM_eval.py` 的HumanML3D分支读取 `../Dataset/HumanML3D`。

目前没有数据，不需要创建一个空目录后立即开训。根目录 `kit.zip` 是KIT模型包，不是KIT动作数据；已有生成坐标 `(1,T,22,3)` 也不能直接当成263维训练特征。

### 7.2 你的改进版还有额外输入

| 任务 | 额外输入 |
|---|---|
| 当前 `dataset_VQ.VQMotionDataset` 的t2m分支 / 对比学习VQ | `motion_retrival/<id>.npy`、`texts_retrival/<id>.npy`，来自对应TMR模型的嵌入 |
| 当前GPT训练 | `texts_GPT_processed/<id>.txt`，每行步骤字典JSON；另需先用选定VQ产生动作token |
| 标准评估 | 完整配对文本和真实动作、GloVe、匹配的评估模型、官方测试划分 |

注意代码中 `retrival` 的拼写就是如此。基础VQ-VAE只需要动作特征，但本分支的现有数据加载器已加入检索特征依赖。不能拿标准数据、漏掉这两个目录就直接跑自定义训练。

### 7.3 还没修好的代码问题

这些是当前代码事实，不是服务器环境缺包；本次只写运行文档，没有修改这些训练入口：

1. **`train_vq.py`**：Python3.10已不能导入旧标准库 `parser`；HumanML3D训练批次按2项解包，但加载器返回4项；非BABEL分支缺少 `val_loader`。基础训练应移除不必要的检索依赖，或完整提供检索输入并统一返回接口。
2. **划分**：`dataset_TM_eval.py` 在 `is_test=True` 时读 `long_test.txt`。要做标准评估，应使用官方 `test.txt`；训练期间的模型选择应使用 `val.txt`，相应训练入口调用应改为验证模式。自定义长动作子集应另设实验名。
3. **官方VQ权重兼容**：本分支多了 `projection_layer.weight/bias`，官方checkpoint不含这两项。新增入口仅容许缺这两项，其他不匹配仍报错；原训练/评估脚本的严格加载还没有统一处理，不能直接无条件 `strict=False` 掩盖其他错误。
4. **GPT入口**：真实文件是 `train_t2m_trans.py`，不是README里的 `train_t2m_trans_1.py`。它额外导入长度估计模块，涉及torch_geometric、spaCy；你之前的成功导入检查没有覆盖这两个包。应核对是否确实使用该模块，再处理对应依赖或去掉未使用导入。
5. **CLIP路径**：新增入口使用根目录 `ViT-B-32.pt`；原训练/评估脚本部分用 `clip.load('ViT-B/32')`，会查CLIP自己的缓存，不会自动复用根目录文件。正式入口应统一指定已有本地文件，避免误以为需要重新下载。
6. **对比学习小批量**：`train_vq_constra.py` 固定选150个正/负候选，`train_vq_constra_100.py` 固定100个；不能拿batch=2照搬 `topk`。还需纠正负样本筛选中将对角线置 `-inf` 后又取最小值导致自身被选为负样本的问题。
7. **过渡模块**：当前 `train_transition.py` 与GPT训练文件相同，没有接上 `StepWeightedTransformer`，不能当成已完成的过渡网络训练。
8. **断点续训**：官方 `net_best_fid.pth` 通常只有模型参数。加载它属于初始化/微调，不等于恢复迭代、优化器、调度器和EMA内部统计。对比训练保存了部分训练状态，但当前EMA统计也需核对，不能直接称为精确恢复。

因此，你现在可以运行第5节的推理与训练检查；正式训练仍然需要数据和上述代码适配，不能因CUDA检查成功就直接启动30万步任务。

## 8. 数据与入口修复后，正式实验怎样排顺序

以下是**后续阶段命令模板**，不是当前缺数据状态下立即执行的命令。只有第7节对应问题解决后再使用；本文没有声称这些原入口已经在你的服务器跑通。

### 阶段A：先修复基础VQ入口，再检查100步

```bash
python -u train_vq.py \
  --dataname t2m \
  --batch-size 32 \
  --window-size 64 \
  --warm-up-iter 10 \
  --total-iter 100 \
  --print-iter 10 \
  --eval-iter 50 \
  --nb-code 512 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --quantizer ema_reset \
  --vq-act relu \
  --loss-vel 0.5 \
  --recons-loss l1_smooth \
  --out-dir output/server_train \
  --exp-name VQ_pipeline_100
```

这里要求真实完整验证数据，因为原入口从训练开始就会做评估，不能用现有11条样例替代它的评估集。100步检查需同时通过取batch、warm-up、评估和保存，再进行长任务。

### 阶段B：基础VQ正式训练

```bash
python -u train_vq.py \
  --dataname t2m \
  --batch-size 256 \
  --window-size 64 \
  --lr 2e-4 \
  --total-iter 300000 \
  --lr-scheduler 200000 \
  --nb-code 512 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --quantizer ema_reset \
  --vq-act relu \
  --loss-vel 0.5 \
  --recons-loss l1_smooth \
  --out-dir output/server_train \
  --exp-name VQVAE_baseline
```

上述batch来自项目基线配置；以服务器实际显存峰值为准。A100有80GB并不意味着必须把batch设置成4096。对比学习版本还受batch内候选数量约束，应作为独立实验接入并核对损失实现。

### 阶段C：固定VQ、生成token、训练GPT

确认步骤文本已经准备好，且训练/验证入口修复后：

```bash
export PYTHONPATH="$PWD/models/Retrival${PYTHONPATH:+:$PYTHONPATH}"

python -u train_t2m_trans.py \
  --dataname t2m \
  --exp-name GPT_baseline \
  --batch-size 128 \
  --num-layers 9 \
  --embed-dim-gpt 1024 \
  --nb-code 512 \
  --n-head-gpt 16 \
  --block-size 51 \
  --ff-rate 4 \
  --drop-out-rate 0.1 \
  --resume-pth output/server_train/VQVAE_baseline/net_last.pth \
  --vq-name VQVAE_baseline_tokens \
  --out-dir output/server_train \
  --total-iter 300000 \
  --lr-scheduler 150000 \
  --lr 0.0001 \
  --down-t 2 \
  --depth 3 \
  --quantizer ema_reset \
  --eval-iter 10000 \
  --pkeep 0.5 \
  --dilation-growth-rate 3 \
  --vq-act relu
```

现有GPT入口会先遍历训练动作，自动编码并保存到 `../Dataset/HumanML3D/VQVAE_baseline_tokens/`，随后才开始GPT训练。这一阶段先出现大量token文件属于正常流程。每次更换VQ或码本都要重新生成token，并保持后续GPT与同一VQ配对。正式跑30万步前，应另用独立实验名缩短迭代并验证一次完整链路。

### 阶段D：标准测试集评估

评估前先确认第7节的划分、权重兼容、CLIP路径问题已解决：

```bash
python -u GPT_eval_multi.py \
  --dataname t2m \
  --exp-name GPT_test_baseline \
  --batch-size 128 \
  --num-layers 9 \
  --embed-dim-gpt 1024 \
  --nb-code 512 \
  --n-head-gpt 16 \
  --block-size 51 \
  --ff-rate 4 \
  --drop-out-rate 0.1 \
  --down-t 2 \
  --depth 3 \
  --quantizer ema_reset \
  --dilation-growth-rate 3 \
  --vq-act relu \
  --resume-pth output/server_train/VQVAE_baseline/net_last.pth \
  --resume-trans output/server_train/GPT_baseline/net_best_fid.pth \
  --out-dir output/server_eval
```

`--resume-pth` 必须是生成该GPT训练token的同一份VQ；不能临时换成另一份 `net_best_fid.pth`。当前脚本在代码中固定重复20次，数据加载器也固定使用batch=32，命令行 `--batch-size` 并不会改变这处硬编码。多样性、多模态和标准重复协议还需与要对标的论文设定逐项核对，不能只看脚本名 `multi` 就认定协议一致。

如要单独评估VQ，待其加载与数据划分修复后使用：

```bash
python -u VQ_eval.py \
  --dataname t2m \
  --nb-code 512 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --vq-act relu \
  --quantizer ema_reset \
  --exp-name VQ_test_baseline \
  --resume-pth output/server_train/VQVAE_baseline/net_last.pth \
  --out-dir output/server_eval
```

## 9. 常见报错对应处理

| 报错 / 现象 | 处理 |
|---|---|
| `scripts/run_local.py` 不存在 | 先同步第3节的新脚本，不是缺Python依赖 |
| `CUDA available=False` 或CPU运行 | 核对 `sys.executable`、激活环境、GPU可见性；新增入口明确使用 `--device cuda` |
| 找不到 `pretrained/...`、`ViT-B-32.pt` | 文件尚未同步完整；只补传缺失文件 |
| 官方VQ提示缺 `projection_layer` | 新入口已只容许这两项缺失；原入口需要单独处理 |
| `Expected finite raw (T,263)` | 重建/训练传入了坐标文件或非法特征；使用 `motions/*.npy` 原始特征 |
| `No ... at least 64 frames` | 检查样例目录和长度；该训练检查使用64帧窗口 |
| 输出目录已存在 | 换一个新 `--out`，或不传 `--out` 自动建新目录 |
| CUDA显存不足 | 查看自己的进程与显存，先减少训练batch或同时运行的任务；推理默认batch=1 |
| `No module named parser` | 原 `train_vq.py` 的旧导入问题，不能通过重装CUDA解决 |
| `No module named src` | TMR导入路径问题；从项目根目录设置第8节的PYTHONPATH |
| `num_samples=0`、一直等不到batch | 检查配对数据、检索嵌入或步骤JSON；当前加载器会静默跳过缺文件的样本 |
| Matplotlib/图形显示报错 | 先加 `--no-gif` 确认推理；渲染用新增入口及 `MPLBACKEND=Agg`，不要改回旧绘图接口 |

不要为补评估器直接运行现有 `dataset/prepare/download_extractor.sh`：它包含 `rm -rf checkpoints`，会删除项目已有权重。你本地已存在所需评估资源，应优先核对服务器是否同步完整。

## 10. 最短执行清单

若只想现在开始，请依次做：

1. 同步 `scripts/run_local.py`。
2. 在服务器激活 `~/envs/t2mgpt`，进入 `~/work/T2M-GPT-main`。
3. 运行第4节资源检查。
4. 运行第5.1节 `generate --device cuda --no-gif`。
5. 通过后运行第5.3节重建和第5.4节3步训练检查。
6. 需要展示时去掉 `--no-gif`；需要完整论文实验时再进入第7～8节。

这份说明保留你已经搭好的服务器环境；新增脚本负责当前无完整数据时可以执行的任务，原项目正式训练代码的后续适配另行处理。
