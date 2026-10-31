# 环境与依赖

项目按任务使用一个基础环境和两个可选依赖组。一般只需安装基础环境；只有运行对应的数据准备或长度预测入口时，才安装附加依赖。

## 基础环境

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate AMG-T2M
python -m pip install -r requirements.txt
```

`environment.yml` 固定 Python 3.10.18 和 pip；根目录 `requirements.txt` 引用锁定的主环境版本。关键版本为 PyTorch 2.0.1（CUDA 11.7 构建）、torchvision 0.15.2、torchaudio 2.0.2、NumPy 1.24.1、SciPy 1.11.2。CLIP 使用仓库记录的固定 Git commit。完整版本表见 [`environment-versions.json`](environment-versions.json) 和 `requirements/base.lock.txt`。

## 按需安装的附加依赖

| 任务 | 依赖文件 | 安装位置 |
|---|---|---|
| TMR 文本特征准备 | `requirements-tmr.txt` | `.runtime/tmr_deps` |
| 长度特征、长度训练与预测 | `requirements-length.txt`、`requirements-spacy-model.txt` | `.runtime/length_deps`、`.runtime/length_models` |

**需要运行 TMR 文本准备时：**

```bash
python -m pip install --no-deps --target .runtime/tmr_deps -r requirements-tmr.txt
```

运行 TMR 编码或缺失文本恢复命令时，将该目录加到 `PYTHONPATH`；完整调用见[执行顺序与命令](workflow.md)。

**需要训练或使用长度模块时：**

```bash
python -m pip install --no-deps --target .runtime/length_deps -r requirements-length.txt
python -m pip install --no-deps --target .runtime/length_models -r requirements-spacy-model.txt
```

`tools/run_amg_lengths.sh` 会从这些目录加载依赖。默认路径也可通过 `T2M_LENGTH_DEPS` 和 `T2M_SPACY_MODEL` 覆盖。`--no-deps --target` 是有意采用的隔离安装方式：附加锁文件保留原实验环境中的版本覆盖关系，不要把它们和主锁文件合并后安装进同一个环境。

启动脚本默认使用当前环境的 `python`；需要指定解释器时可设置 `T2M_PYTHON=/path/to/python`。

以上版本取自项目记录的运行环境。文档整理没有重新创建干净环境或运行训练，因此新机器上的安装与复现仍需按 [`workflow.md`](workflow.md) 的顺序准备数据和模型资源，并自行核对安装结果。
