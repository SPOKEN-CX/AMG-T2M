**AMG-T2M — private research snapshot**

基于 T2M-GPT 的 HumanML3D 文本到动作研究代码，包含 AMG 语义对齐 VQ、配套 GPT、动作长度预测及潜空间分析的工程实现。当前版本用于私有开发和代码整理；不代表完整 AMG 官方实现、完整论文复现或已审定的公开发行版。

本仓库保留源码、配置和必要第三方许可证。数据、动作/文本缓存、模型权重、训练输出、用户研究材料、论文 PDF 和 SMPL 网格渲染资源均不随包分发。先阅读 [实现范围](docs/implementation-status.md)、[数据与模型](docs/data-and-models.md) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。

**环境准备**

基准版本来自工作环境：Python 3.10.18、PyTorch 2.0.1（原环境 CUDA 构建为 cu117）、NumPy 1.24.1。下面是整理后的安装候选方案，尚未在全新环境完成端到端验证；不要把这些命令当成已通过复现验收的承诺。

```bash
# 使用 Python 3.10 创建隔离环境
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

长度预测模块另需：

```bash
python -m pip install -r requirements-length.txt
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl
```

TMR 特征准备另需 `requirements-tmr.txt`；潜空间分析另需 `requirements-analysis.txt`。所有安装均应在隔离环境内进行。源码不再自动向 Python 路径注入原服务器的 output 依赖目录。可通过 `T2M_PYTHON` 指定启动脚本使用的解释器，通过 `T2M_SPACY_MODEL` 指定已安装的 spaCy 模型名或合法本地路径。

**使用入口**

| 任务 | 入口 | 前置条件 |
|---|---|---|
| 基础 VQ | `train_vq.py` | HumanML3D、归一化统计、评估器和 GloVe |
| AMG 对比 VQ | `train_vq_constra_100.py` | 上述资源及核实的 TMR 配对特征 |
| GPT | `train_t2m_trans.py` | 确定的冻结 VQ、配套 token、CLIP 和评估资源 |
| 长度预测 | `train_length_est.py` | 长度记录、CLIP 词特征、依存图和可选依赖 |
| VQ/GPT 评估 | `VQ_eval.py`、`GPT_eval_multi.py` | 正确配套的权重和指定 split |
| 单段长度控制 | `scripts/generate_amg_length_controlled.py` | 已核验的 VQ/GPT/长度权重及身份文件 |
| 潜空间分析 | `latent_tsne_comparison/run.sh` | 合法取得的数据、选定 checkpoint、分析依赖 |

`scripts/run_*.sh` 保存了不同工程实验的参数与恢复规则，运行前检查资源及输出目录。它们可能执行长时间训练；不要为了检查代码上传而直接启动。首次准备环境后，可以先做不需要真实数据或预训练权重的 CPU 回归检查：

```bash
python -m scripts.test_gpt_accumulation
python -m scripts.test_gpt_length_control
```

这两个检查只覆盖相应代码逻辑，不能替代动作生成 demo、正式训练或论文指标复现。本次整理后的验证记录见 [快照改动及验证](docs/snapshot-changes.md)。

**来源和授权**

T2M-GPT 上游：[项目仓库](https://github.com/Mael-zys/T2M-GPT)、[论文](https://arxiv.org/abs/2301.06052)。TMR 上游：[项目仓库](https://github.com/Mathux/TMR)、[论文](https://arxiv.org/abs/2305.00976)。数据来源与相关代码：[HumanML3D](https://github.com/EricGuo5513/HumanML3D)。AMG 的目标论文为《AMG: Long-Term Text-to-Motion Generation from Arbitrary-Length Prompts》；本地工程来源版本与完整书目信息仍待核实，暂不编造作者、会议、DOI 或上游 commit。

根目录 [LICENSE](LICENSE) 是工作区已有的 Apache-2.0 文本。第三方文件存在不同条款，特别是四元数工具所保留的 QuaterNet 非商业条款；本项目尚未完成统一公开授权审查，不能据根目录许可证推定所有文件、数据和权重均允许商用。必要权利声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

当前不发布虚构的论文结果、录用信息或软件引用作者列表。Issues 可用于记录复现问题，请提供命令、软件版本及脱敏错误信息，不上传数据、密钥或完整环境转储。后续更新取决于研究进度和维护者时间。
