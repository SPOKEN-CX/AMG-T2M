# AMG-T2M

HumanML3D 文本到动作研究代码，包含 TMR 引导的 VQ、配套 GPT、动作长度预测及潜空间分析模块。

## 效果预览

![AMG-T2M 文本提示与生成动作序列示例](assets/amg_teaser.png)

*文本提示与对应生成动作序列示例。*

![AMG-T2M 长时序动作生成对比](assets/long_term_comparison.png)

*长时序动作生成结果对比。*

![AMG-T2M 与其他方法的动作生成对比](assets/method_comparison.png)

*AMG-T2M 与其他方法在长时序和短时序提示上的定性对比。*

论文标题：**AMG: Long-Term Text-to-Motion Generation from Arbitrary-Length Prompts**。论文信息暂未公开，仓库当前保持 Private。

## 快速开始

从项目根目录创建主环境：

```bash
conda env create -f environment.yml
conda activate AMG-T2M
python -m pip install -r requirements.txt
```

主环境固定 Python 3.10.18、PyTorch 2.0.1（CUDA 11.7 构建）等已记录版本。TMR 文本准备和长度预测只需按对应任务安装附加依赖；不要把所有依赖锁文件合并安装。详见[环境与依赖](docs/environment.md)。

数据集、评估资源、预训练模型和训练产物需单独准备。HumanML3D 的来源、引用、许可和目录格式见[数据集说明](docs/datasets.md)；资源路径见[数据与模型](docs/data-and-models.md)和[资源清单](resources/manifest.json)。

## 执行顺序

HumanML3D 主要组件的准备、VQ 两阶段训练（batch 512，10,000 步续训至 100,000 步）、VQ 冻结、GPT 各 batch 模式、评估、长度模块和单段生成命令统一列在[执行顺序与命令](docs/workflow.md)。

仓库现有单段长度控制入口最多生成 196 帧；转场模型尚未形成完整训练与推理流程。因此当前源码不等同于完整任意长度生成系统或论文全部结果，详见[实现范围](docs/implementation-status.md)。

## 代码结构

```text
src/              模型、数据集、损失与工具
configs/          实验配置说明；启动参数保存在 tools/run_*.sh
preprocess/       特征及训练输入准备
tools/            训练、评估、生成和启动脚本
visualize/        可视化与潜空间分析
assets/           README 图片
docs/             环境、数据、执行顺序、FAQ 和实现状态
```

实验启动脚本会检查输入和恢复状态；运行前请确认数据、checkpoint、显存、输出目录及相应实验配置。已有输出目录遇到失败时，先检查日志和 checkpoint，再决定恢复或使用新实验名。

## 引用与许可

论文正式公开后会补充作者、发表信息和 BibTeX。上游代码和第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。根目录 [LICENSE](LICENSE) 不自动覆盖第三方代码、数据集或模型权重的授权条款。

Issues 可用于报告环境问题和复现问题。请附上依赖配置、完整命令、数据 split、模型身份及脱敏日志。
