# AMG-T2M

面向长时序文本到动作生成的项目，围绕 TMR 引导的动作 VQ、文本条件 GPT、动作长度预测和潜空间分析整理源码与复现入口。

## 本地完整源码与复现基准

本地项目源码按原目录结构保存在 [`reproduction/local/`](reproduction/local/README.md)，包含模型、数据加载器、多卡训练、TMR 辅助工具、长度模块、渲染代码、依赖与运行指南。复现本地工程时，以该目录为工作目录，按其中的指南配置环境和外部资源。

本次共归档 200 个本地源码、配置与说明文件，并提供[逐文件来源与哈希清单](reproduction/source-manifest.json)、[审查报告](docs/local-source-sync.md)及[路径配置工具](reproduction/configure_paths.py)。数据、权重、结果、虚拟环境和临时测试脚本均未上传。仓库根目录的 `src/`、`tools/` 为整理版入口，使用其工作流时从仓库根目录运行。

![AMG-T2M 文本提示与生成动作序列示例](assets/amg_teaser.png)

*文本提示与对应生成动作序列示例。*

![AMG-T2M 长时序动作生成对比](assets/long_term_comparison.png)

*长时序动作生成结果对比。*

![AMG-T2M 与其他方法的动作生成对比](assets/method_comparison.png)

*AMG-T2M 与其他方法在长时序和短时序提示上的定性对比。*

论文标题：**AMG: Long-Term Text-to-Motion Generation from Arbitrary-Length Prompts**。论文信息暂未公开，仓库当前保持 Private。

## 整理版快速开始

从项目根目录创建主环境：

```bash
conda env create -f environment.yml
conda activate AMG-T2M
python -m pip install -r requirements.txt
```

主环境版本、分任务附加依赖和安装说明见[环境与依赖](docs/environment.md)。训练、评估、长度控制生成和潜空间分析的入口顺序见[执行流程](docs/workflow.md)。

## 本地快速检查与生成

`tools/run_local.py` 提供动作预览、VQ 重建、训练 smoke check 和文本生成入口。数据、统计文件、权重和 CLIP 模型需事先放在本地资源目录，或通过参数传入：

```bash
# 预览一个本地关节序列 (T,22,3)
python tools/run_local.py preview --source /path/to/joints.npy

# 用 VQ-VAE 重建一个本地 HumanML3D 特征序列 (T,263)
python tools/run_local.py reconstruct --source /path/to/motion_features.npy

# 在本地特征样本目录执行少量训练步，用于检查训练路径
python tools/run_local.py train-smoke --source /path/to/motion_features_dir --steps 3

# 文本生成；如资源不在默认位置，可传入 --vq、--gpt、--meta-dir、--clip-model
python tools/run_local.py generate --text "a person walks forward" --device auto
```

预览和 smoke check 需要使用者本地准备的动作样本；这些样本不会随仓库分发。多条 `--text` 会按生成顺序拼接动作 token，详见入口说明。

## 数据集与模型资源

本仓库只提供数据集的官方获取入口和使用说明，不托管数据集、标注、派生缓存、预训练权重、训练 checkpoint 或生成动作文件。

- **HumanML3D（主要数据集）**：[官方代码与获取说明](https://github.com/EricGuo5513/HumanML3D)
- **AMASS（HumanML3D 的来源数据之一）**：[官方获取页](https://amass.is.tue.mpg.de/) · [许可说明](https://amass.is.tue.mpg.de/license.html)
- **KIT-ML（可选数据入口）**：[官方数据页](https://motion-annotation.humanoids.kit.edu/dataset/)
- **模型、CLIP、统计量和缓存路径**：[数据与模型清单](docs/data-and-models.md) · [资源路径索引](resources/manifest.json)

下载或重建数据前，请先阅读对应上游的获取方式、许可和引用要求。HumanML3D 数据准备与输入格式说明见[数据集文档](docs/datasets.md)。

## 代码结构

```text
reproduction/     本地完整源码基准、同步清单和路径配置工具
src/              模型、数据集、损失、训练选项和通用工具
preprocess/       TMR、GPT、长度特征及训练输入准备
tools/            训练、评估、生成、检查和启动脚本
visualize/        潜空间分析与动作可视化
configs/          实验配置说明
resources/        外部模型与数据资源路径清单
assets/           README 图片
docs/             环境、数据、工作流和复现说明
```

关键文档：[执行流程](docs/workflow.md)、[FAQ](docs/FAQ.md)、[第三方代码与许可](THIRD_PARTY_NOTICES.md)。运行实验时请记录使用的数据 split、checkpoint 身份、统计量和配置，以便对应结果。

## 引用与许可

论文正式公开后会补充作者、发表信息和 BibTeX。上游代码和第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。根目录 [LICENSE](LICENSE) 不覆盖第三方代码、数据集或模型权重各自的授权条款。
