# 本项目当前能跑什么，以及缺少 HumanML3D 时怎么做

适用范围：本文记录Mac上的独立验证。你的实际实验使用已搭好的服务器环境，请以 [服务器运行说明](SERVER_RUN_GUIDE.zh-CN.md) 中的路径、环境和CUDA命令为准。

检查日期：2026-10-21。检查范围：README、主要模型、数据加载、训练/评估/生成/可视化入口、权重和样本文件，以及实际本地运行。以下“已验证”均来自本项目这次运行的结果。

## 1. 结论

你说的 human3d 在本项目中应指 **HumanML3D**。现在缺少完整动作数据，但已有足够资源跑通预训练推理、动作重建，以及小样本反向传播。无需等完整数据集才能开始。

当前可以做：

| 任务 | 状态 | 本次实际结果 |
|---|---|---|
| 已有动作的骨架动画 | 已跑通 | 原有 `motion.npy`，308 帧 |
| 官方权重文字生成动作 | 已跑通 | `a person walks forward` → 29 tokens → 116 帧，5.8 秒 |
| 本地 self_train 权重推理 | 已跑通 | `a person is jumping` → 12 tokens → 48 帧，2.4 秒 |
| 多段文字生成长动作 | 已跑通 | 跳跃、行走、坐下 → 12+29+48 tokens → 356 帧，17.8 秒 |
| VQ-VAE 编码/解码重建 | 已跑通 | 124 帧 → 31 tokens → 124 帧 |
| 小样本训练流程检查 | 已跑通 | batch=2，窗口64帧，完成3次 forward/backward/optimizer step |
| 正式 HumanML3D 全量训练与论文指标 | 暂不能直接跑 | 缺数据；原训练/评估代码还有接口和路径问题 |
| KIT-ML 完整实验 | 可作为下一步 | 需获取真实 KIT-ML 数据并修复 KIT 分支 |
| SMPL 人体网格渲染 | 本次未配置和验证 | 现有代码依赖 SMPL 资源、CUDA、OpenGL/OSMesa |

**小样本检查证明链路可以执行，不代表训练收敛、复现论文或取得有效的 FID/R-Precision。** 3次训练损失没有持续下降，不能据此判断模型效果。

## 2. 本地资源到底有哪些

- `pretrained/VQVAE/net_best_fid.pth`：官方预训练 VQ-VAE，码本512×512。
- `pretrained/VQTransformer_corruption05/net_best_fid.pth`：官方 GPT，embedding=1024，block_size=51，base/head各9层，约2.28亿参数。
- `checkpoints/self_train/VQVAE/t2m/net_best_fid.pth` 和 `checkpoints/self_train/GPT/net_best_fid.pth`：本地训练权重；这次验证了两者一起推理可执行，未验证训练来源及指标。
- `ViT-B-32.pt`：本地 CLIP，推理时直接读这个文件，不再下载一份。
- `checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy`、`std.npy`：263维特征归一化参数。
- `motions/000090.npy` 至 `000100.npy`：11条有限数值的原始263维特征，长度47～199帧；其中10条能使用64帧训练窗口。
- `models/Retrival/Babel/motion/`：42个263维 BABEL 样例，未用于本次 HumanML3D 训练检查。
- `generation/`：11634个 `.npy` 文件，其中11627个可读，格式都是 `(1,T,22,3)` 坐标；另7个为空文件。
- `long_motions/`、`short_motions/`、根目录 `motion.npy`：已有坐标结果，可以可视化。
- `glove/`：词向量存在；HumanML3D 的 `text_mot_match/model/finest_1.tar` 评估器也存在。
- `models/Retrival/`：包含 TMR 检索模型代码、HumanML3D/KIT 配置与权重、文本嵌入等。它是额外的检索/对比学习组件。

特别容易误判的地方：

1. `dataset/HumanML3D/` 目前只有空的 `VQVAE/` 子目录，没有 `new_joint_vecs/`、`texts/`、官方数据划分文件。
2. 根目录 **`kit.zip` 不是 KIT-ML 数据集**。检查压缩包后确认是 `kit/text_mot_match` 等模型、元数据和日志。其评估权重叫 `finest.tar`，当前评估加载器期待 `finest_1.tar`，即使解压也需统一文件名/配置。
3. `texts_no_word_des.zip` 只是文本，不包含动作；根目录 `texts/` 的10个文本也不与 `motions/` 的11个编号配对，不能直接组成文本到动作训练集。
4. 已检查同级备份 `bishe1.zip` 的目录索引，其中也没有 `new_joint_vecs`，不含 `train_length_est.py` 或 `gen_long_seq.py`。
5. `(1,T,22,3)` 生成坐标不能直接改名放进 `new_joint_vecs/`。模型输入是 `(T,263)` 特征，包含根运动、局部关节、旋转、速度、接触等，二者不等价。

## 3. 直接运行

这次在项目内新建了独立 `.venv-smoke/`，没有改动现有 Conda 环境。验证配置为 Python 3.11.15、PyTorch 2.5.1、NumPy 1.26.4、Matplotlib 3.9.4，全部使用 CPU。设备的 MPS 可用，但本次没有验证 MPS；原 CUDA 训练入口仍不能直接在 Mac 上运行。

先进入项目：

```bash
cd /path/to/AMG-T2M/reproduction/local
```

### 输入文字，生成动作

```bash
.venv-smoke/bin/python scripts/run_local.py generate \
  --text 'a person walks forward'
```

输出 `motion.npy`、`motion.gif`、`features.npy`、`tokens.npy`、`result.json`。默认使用官方 VQ/GPT 配对权重。

### 使用本地 self_train 权重

```bash
.venv-smoke/bin/python scripts/run_local.py generate \
  --text 'a person is jumping' \
  --vq checkpoints/self_train/VQVAE/t2m/net_best_fid.pth \
  --gpt checkpoints/self_train/GPT/net_best_fid.pth
```

注意保持 VQ/GPT 的配对关系。即使张量尺寸相同，GPT token对应的码本含义也可能随 VQ 训练发生变化；任意混搭不能据此判断效果。

### 多段文本长动作

```bash
.venv-smoke/bin/python scripts/run_local.py generate \
  --text 'a person is jumping' \
  --text 'a person walks forward' \
  --text 'a person sits down'
```

这里是各段生成token后拼接、统一解码，与原 `motion_gen.py` 的思路一致。**没有调用训练好的过渡模块，不能据此声称段间动作自然连续。** 每段最多50个token；EOS会提前结束生成，不能保证精确时长。

### 重建已有263维动作特征

```bash
.venv-smoke/bin/python scripts/run_local.py reconstruct \
  --source motions/000090.npy
```

VQ下采样倍数为4，125帧样例会截成124帧。输出原始动作与重建动作的 `.npy`、GIF，以及tokens。此次原始特征MAE约0.03431，仅为该样例的重建检查值。

### 跑几个训练步骤

```bash
.venv-smoke/bin/python scripts/run_local.py train-smoke --steps 3
```

这条命令在现有10条可用样例上微调 VQ-VAE，检查有限loss、有限梯度和实际权重更新。输出单独的 `smoke_vq.pth`，不覆盖已有权重；不是原项目完整对比学习训练，也没有文本/GPT训练、数据集划分或正式评估。

### 只看现成结果

```bash
.venv-smoke/bin/python scripts/run_local.py preview --source motion.npy
```

也可以把 `--source` 换为 `short_motions/npy/` 下的单个坐标文件。此模式无需加载模型。

默认输出放在 `output/local_runs/<模式>_<时间>/`，每次独立目录。可以用 `--out 新目录` 指定位置；已存在的输出目录会被拒绝，避免覆盖。`--no-gif` 可跳过模型任务的动画导出；`--sample` 可使用概率采样。

若以后重建这个最小环境：

```bash
uv venv --python 3.11 .venv-smoke
uv pip install --python .venv-smoke/bin/python -r requirements-local.txt
```

这份依赖清单仅用于新增本地入口，不包括 TMR、图网络、spaCy、TensorBoard、SMPL 等完整研究栈。

## 4. 原项目结构与当前阻塞

主链路是：原始263维动作 → VQ-VAE → 离散token；文字 → CLIP → GPT → token → VQ解码 → 263维特征 → 22关节坐标。

| 文件/目录 | 用途与检查发现 |
|---|---|
| `models/vqvae.py`、`encdec.py`、`resnet.py` | VQ编码器、解码器和残差卷积；当前分支额外有 `projection_layer`，官方VQ权重没有这两个参数，但当前 `forward()` 没有使用该层 |
| `models/quantize_cnn.py` | 量化码本；原本在构造函数强制 `.cuda()`，本次移除两处，使buffer随整个模型 `.to(device)` 移动 |
| `models/t2m_trans.py` | CLIP条件自回归GPT；原 `sample()` 若第一步就生成EOS会引用未定义的 `xs`，新增入口自行处理这一情况 |
| `dataset/dataset_VQ.py` | 当前t2m版本额外读取 `motion_retrival/`、`texts_retrival/`，返回4项；读取失败会被宽泛 `except` 静默跳过 |
| `train_vq.py` | 仍按2项解包上述数据；非BABEL分支没有建立后续使用的 `val_loader`；导入已在新Python删除的 `parser` 模块；还导入TMR形成额外依赖 |
| `train_vq_constra.py`、`train_vq_constra_100.py` | 改进版对比训练依赖TMR预计算嵌入；每个样本取150/100个正负候选，小样本时不能沿用该值直接 `topk` |
| `train_vq_multi.py`、`train_vq_constra_multi.py` | CUDA/DDP多卡入口，不能直接用于本机CPU验证 |
| `dataset/dataset_tokenize.py` | 把训练动作编码成token；KIT分支缺 `split_file`，BABEL分支还引用缺少的属性，需要分别修复 |
| `dataset/dataset_TM_train.py` | GPT训练额外要求 `texts_GPT_processed/` 等步骤JSON，缺失时直接跳过样本；BABEL命名还存在横线/下划线不一致 |
| `train_t2m_trans.py` | 先生成动作token，再训练GPT；导入长度估计模块会带入torch_geometric、spaCy等依赖，并有硬编码CUDA |
| `train_transition.py` | 当前内容与 `train_t2m_trans.py` 完全相同；没有训练 `StepWeightedTransformer`，不能把此文件视为已经接好的过渡训练流程 |
| `models/TransitionGenerator.py` | 定义过渡预测网络；参数/损失/训练接入需要另行核对和实现 |
| `VQ_eval.py`、`GPT_eval_multi.py`、`utils/eval_trans.py` | 正式评估入口，依赖真实动作、文本、评估器和CUDA；已有输出坐标不足以直接执行完整标准评估 |
| `dataset/dataset_TM_eval.py` | `is_test=True` 使用的是自定义 `long_test.txt`，不是官方 `test.txt`；复现标准指标前必须恢复并核对划分 |
| 多处 `data_root` | 同时出现 `../Dataset/HumanML3D`、`../dataset/HumanML3D` 和 `./dataset/...`，与README目录不一致，需要集中配置 |
| `motion_gen.py` | 已有不依赖数据集的分段推理思路，但写死 `/home/peng/...`、CUDA，并重置命令行参数；这次新增入口替代其本地运行用途 |
| `models/MotionLenEstimator/` | 有图网络长度估计模块和权重，依赖spaCy英语模型和固定机器路径；`run_order`提及的 `train_length_est.py` 不存在 |
| `models/Retrival/` | TMR检索/语义相似度组件；部分代码硬编码Linux路径，`src.*`导入依赖PYTHONPATH；不是当前最小推理所必需 |
| `visualization/plot_3d_global.py` | 旧Matplotlib绘图接口；新入口使用兼容当前版本的Matplotlib/Pillow骨架渲染 |
| `render_final.py` | SMPL网格渲染，并非普通骨架动画；当前写死CUDA和OSMesa |
| `VQVAE_test.py` | 实际是扫描动作长度的辅助程序，不是VQ重建测试 |
| `train_GPT_split.py` | 空文件 |
| `Long_ave_test.py`、`short_gen.py` | 按固定路径统计/复制样例的辅助脚本 |
| `draw_comparison.py` | 用写在代码里的固定数值画图，不会计算实验指标 |
| `text_t5.py` | 单独的T5英德翻译示例，不是动作生成主链路 |
| `LibMTL/` | 多任务学习辅助库，不是当前VQ→GPT最小链路的必要步骤 |

原 `environment.yml` 固定了Linux/CUDA10.1和旧Python/PyTorch的具体build，不能在Apple Silicon上原样创建；新入口采用单独的最小环境。

**现有 `dataset/prepare/download_extractor.sh` 包含 `rm -rf checkpoints`，会删除当前整个权重目录；已经有的约12GB checkpoints 不要用它重下。** 该脚本还写死本机7890代理。需要补资源时应逐文件下载并保留现有权重。

## 5. 缺少HumanML3D，如何安排实验

### A. 今天先验证方法和生成效果

使用上面的官方/本地权重推理、重建和小样本训练检查。可比较固定文字、固定seed下的结果；可分析生成长度、段间跳变等。若算坐标速度或平滑性，应明确这是样例诊断，不是官方论文指标。

### B. 希望尽快做完整数据集上的训练/评估：KIT-ML

[HumanML3D官方仓库](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data) 提供预处理KIT-ML的直接下载入口：

[KIT-ML官方提供的Google Drive目录](https://drive.google.com/drive/folders/1D3bf2G2o4Hv-Ale26YW18r1Wrh7oIAwK?usp=sharing)。本次核对了官方入口，没有下载数据或验证该网盘在你的网络下的传输情况。

下载所需内容是 `new_joint_vecs`、`texts`、`Mean.npy`、`Std.npy`、`train.txt`、`val.txt`、`test.txt` 等；无需先下载动画。

但这份项目当前不能只改 `--dataname kit`：KIT为21关节/251维，现有最小演示使用22关节/263维的HumanML3D权重；KIT加载器路径、未定义变量和检索分支、评估权重名称都要先修。KIT结果应报告为KIT实验，不能写成HumanML3D结果。

建议先修一个只需要动作特征的基础VQ训练入口，再接文本/GPT，最后恢复官方评估。需要正式训练时再配置NVIDIA GPU；本次CPU检查不用于估计全量训练耗时。

### C. 必须复现HumanML3D论文结果

官方说明因AMASS分发规则，不能直接分发HumanML3D动作数据，而是提供从AMASS重建的脚本。按照 [HumanML3D官方说明](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data)：获取对应AMASS数据、SMPL+H和DMPL模型，再依次运行：

1. `raw_pose_processing.ipynb`
2. `motion_representation.ipynb`
3. `cal_mean_variance.ipynb`

这条路径适用于需要完整可复现数据的正式实验。也可先向已有合法访问权限的课题组获取在许可范围内可使用的预处理数据或小子集，明确来源与许可，不把来源未知的镜像当官方数据。

最小子集仍应有相互对应的263维动作、原格式文本及独立数据划分。仅验证VQ可暂不需要文字；训练文字条件GPT则需要真实配对。不要把测试输出或同一动作的切片分到训练和测试两侧来报告指标。

**建议顺序：先用已跑通的本地入口检查你的想法 → 获取KIT-ML做一个完整基线 → 需要HumanML3D对标时再补齐官方数据，并修复本分支评估与额外标注依赖。**

## 6. 本次修改与结果位置

修改已有代码仅两处码本初始化的设备绑定；新增 `scripts/run_local.py`、`requirements-local.txt` 和本说明。`.gitignore` 增加独立环境目录。原训练入口、原权重、原动作文件保持原样。

实际运行产物：

- `output/local_verified/preview/motion.gif`
- `output/local_verified/reconstruct/original.gif`、`reconstruction.gif`、`result.json`
- `output/local_verified/train_smoke/result.json`、`smoke_vq.pth`
- `output/local_verified/generate/motion.gif`、`motion.npy`、`result.json`
- `output/local_verified/self_trained_generate/motion.gif`、`motion.npy`、`result.json`
- `output/local_verified/multi_segment/motion.gif`、`motion.npy`、`result.json`
- `output/local_verified/contact_sheet.png`：原动作、重建、两套模型生成结果的中间帧检查图。

动画均检查了文件帧数和尺寸；已查看上述单段/重建结果的中间帧，确认渲染出了可见骨架。生成内容与文本是否一致、长动作过渡是否合理，还需逐样例评价。
