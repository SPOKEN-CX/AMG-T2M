# 本地项目复现源码基准

本目录保存本地 `T2M-GPT-main` 的源码、实验配置和运行说明，按本地原目录结构组织。模型、加载器、训练器与工具来自同一份本地工程；运行命令时以本目录为工作目录，避免与仓库根目录的 `src/` 整理版交叉导入。

200 个本地文件的去向、内容哈希和排除项见 [source-manifest.json](../source-manifest.json)，审查发现和验证范围见 [审查报告](../../docs/local-source-sync.md)。源码逻辑按原文件保存，三份中文指南仅替换个人机器路径、账号和本地文件链接。原项目 README 保存为 [README.upstream.md](README.upstream.md)。

## 环境与运行位置

从 GitHub 仓库根目录进入本目录：

```bash
cd reproduction/local
export PYTHONPATH="$PWD:$PWD/models/Retrival${PYTHONPATH:+:$PYTHONPATH}"
```

先按照 [AMG 训练评估指南](AMG_TRAIN_EVAL_GUIDE.zh-CN.md) 选择所需实验及依赖。仓库根目录的 [环境说明](../../docs/environment.md) 记录服务器主环境；本目录 `environment.yml` 是本地保留的上游历史环境，`requirements-local.txt` 是独立的最小本地演示环境（PyTorch 2.5.1），`models/Retrival/requirements.txt` 是 TMR 依赖清单。这些文件用于不同环境，不应合并安装。

中文指南中的 `/path/to/...`、`user@server.example` 为示例占位，运行前替换成实际路径和服务器。原脚本中仍保留历史资源路径；下面的工具可以在当前克隆副本中配置这些路径：

```bash
# 先查看将被修改的文件；不下载、不读取或上传数据
python ../configure_paths.py --data-root /path/to/Dataset
# 确认路径后应用；只改清单中 Python/JSON 的已知路径字符串
python ../configure_paths.py --data-root /path/to/Dataset --spacy-model en_core_web_sm --write
```

`--data-root` 指包含 `HumanML3D/`、`KIT-ML/` 或 `BABEL_TEACH/` 的父目录。spaCy 模型由使用者另行安装或指定本地模型路径。配置后源码哈希会与发布清单不同，这是预期的路径变更；原始文件仍可从 Git 恢复。工具不改变数据内容、模型计算和训练参数。

## 入口与目录

- VQ：`train_vq.py`；对比训练变体 `train_vq_constra.py`、`train_vq_constra_100.py`；多卡变体 `train_vq_multi.py`、`train_vq_constra_multi.py`。
- GPT：`train_t2m_trans.py`；`train_transition.py` 与它内容相同，名称不能作为独立转场训练的依据。
- 正式评估：`VQ_eval.py`、`GPT_eval_multi.py`。它们是复现评价指标的入口，因此保留；排除的是临时测试/探测脚本。
- 长度模块：`models/MotionLenEstimator/`；其中历史 trainer 的缺失依赖见审查报告。整理版长度训练入口位于仓库根目录 `tools/train_length_est.py`，使用该入口时回到仓库根目录按整理版工作流运行。
- TMR：`models/Retrival/`，保留本地拼写、工具、特征处理及配置。
- 可视化：`visualize/`、`visualization/`、`render_final.py`、`draw_comparison.py`、`gif2mp4.py`；这里只保存源码。
- 本地快速推理：`scripts/run_local.py`，按 [本地运行指南](LOCAL_RUN_GUIDE.zh-CN.md) 使用。
- 历史命令记录：`run_order`，其中不是每个命令都对应当前存在的入口，具体见审查报告。

训练及评估的完整参数请使用 [AMG 训练评估指南](AMG_TRAIN_EVAL_GUIDE.zh-CN.md)，服务器准备见 [服务器指南](SERVER_RUN_GUIDE.zh-CN.md)。本次同步未重新执行 GPU 训练或计算论文指标。

## 数据与外部资源

数据本体、标注、样本、划分、派生索引、均值方差、权重和运行产物都不随本目录上传。`dataset/` 中仅有加载器和资源获取脚本；`checkpoints/` 中仅保留 7 份文本 `opt.txt` 配置，不含 checkpoint 权重。

- HumanML3D：[官方获取说明](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data)
- AMASS：[官方获取页](https://amass.is.tue.mpg.de/)
- KIT-ML：[官方数据页](https://motion-annotation.humanoids.kit.edu/dataset/)
- BABEL：[官方数据页](https://babel.is.tue.mpg.de/)
- TMR：[官方代码与模型准备说明](https://github.com/Mathux/TMR)
- T2M-GPT 预训练与评估资源：[上游说明](https://github.com/Mael-zys/T2M-GPT)

原训练目录使用的资源相对位置为 `checkpoints/`、`pretrained/`、`glove/`、`ViT-B-32.pt`，TMR 另需 `models/Retrival/stats/`、模型权重及 annotations/cache。准备资源时必须匹配所选 VQ、GPT、统计量和评估器；详细格式见 [数据集说明](../../docs/datasets.md)。历史 `dataset/prepare/*.sh` 含删除资源目录和固定代理、目录拼写等问题，请按官方资源说明准备；审查没有执行这些脚本。

第三方文件保留其原始版权头；[第三方说明](../../THIRD_PARTY_NOTICES.md) 与文件自带条款共同适用。SMPL/SMPLify 的受限源码条款仍在文件头，相关人体模型和先验二进制不随源码提供。
