**实际运行环境的依赖版本记录**

本记录依据2026-10-30现有运行环境的包元数据、PyTorch系列 `version.py`、原启动脚本及已有长度特征manifest整理。按用户要求，本轮暂缓全新环境安装验收和demo验证；已有复现结果由原实验记录保存。本次工作固定版本和加载顺序，不调整实验超参数或模型实现。

**主环境**

| 项目 | 记录值 |
|---|---|
| 平台 | Linux x86_64 |
| Python | 3.10.18 |
| PyTorch | 包元数据2.0.1；运行构建2.0.1+cu117 |
| torchvision | 包元数据0.15.2；运行构建0.15.2+cu117 |
| torchaudio | 包元数据2.0.2；运行构建2.0.2+cu117 |
| PyTorch编译对应CUDA | 11.7 |
| NumPy / SciPy | 1.24.1 / 1.11.2 |
| Matplotlib / ImageIO | 3.7.3 / 2.38.0 |
| PyTorch Lightning / torchmetrics | 2.0.9 / 1.1.2 |
| Hydra / OmegaConf | 1.3.2 / 2.3.0 |
| scikit-learn / threadpoolctl | 1.3.2 / 3.7.0 |
| pip / setuptools / wheel | 26.2.1 / 80.9.0 / 0.45.1 |
| CLIP | 包版本1.0；Git commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |

主环境共记录105个Python分发包，完整精确版本见 [base.lock.txt](../requirements/base.lock.txt)。其中包含原环境已有的工具和辅助包，不表示这105个包全都是最小推理必需依赖。NVIDIA CUDA/cuDNN等Python分发包也保留实际版本。

PyTorch的包元数据版本和 `torch.__version__` 运行构建字符串需要同时记录。这里的 `cu117` 指该PyTorch构建对应的CUDA运行时，不能把 `nvidia-smi` 显示的最高CUDA支持版本直接写成项目依赖。Conda旧记录中的工具版本可能与后续pip更新不同；本记录采用当前Python包元数据中的有效版本。

**TMR文本特征附加依赖**

| 包 | 版本 |
|---|---|
| transformers | 4.38.2 |
| huggingface-hub | 0.23.5 |
| tokenizers | 0.15.2 |
| safetensors | 0.4.5 |

原工程将这4个包单独安装在附加目录中，通过 `PYTHONPATH` 加载。主环境中的Lightning、Hydra、einops等仍由主环境提供。完整列表见 [tmr-overlay.lock.txt](../requirements/tmr-overlay.lock.txt)。

**长度预测附加依赖**

长度模块的附加目录共记录50个包，包括 `torch-geometric==2.5.3`、`spacy==3.5.4`、`thinc==8.1.12`、`pydantic==1.10.27`。英语解析模型为 `en_core_web_sm==3.5.0`。完整列表见 [length-overlay.lock.txt](../requirements/length-overlay.lock.txt)。

此目录在主环境前加载，因此同名包以附加目录版本为准。例如Jinja2由主环境3.1.2变为3.1.6，MarkupSafe由2.1.2变为3.0.4，setuptools由80.9.0变为84.2.1。合并成单次pip安装会改变原加载关系，或因两个精确版本相互冲突而无法解析。

机器可读记录见 [environment-versions.json](environment-versions.json)。按包声明的依赖约束核对，主环境105个包、TMR有效环境109个包、长度模块有效环境129个包均未发现缺失或版本范围冲突。这是已安装元数据的静态核对，未启动训练或执行新环境复现。

**按记录重建的命令**

以下命令供之后需要重建环境时使用，本轮不执行。先在项目根目录创建基础环境并安装锁定版本：

```bash
conda env create -f environment.yml
conda activate AMG-T2M
python -m pip install -r requirements.txt
```

`environment.yml` 固定Python和pip；Python包由105项主环境锁文件固定。这里按照参考环境记录CUDA版本的PyTorch依赖，没有将CPU构建作为项目基准。

需要TMR文本特征准备时，再准备独立附加目录：

```bash
python -m pip install --no-deps --target .runtime/tmr_deps -r requirements-tmr.txt
PYTHONPATH="$PWD/.runtime/tmr_deps${PYTHONPATH:+:$PYTHONPATH}" \
  python scripts/prepare_amg_missing_text.py --help
```

需要长度模块时：

```bash
python -m pip install --no-deps --target .runtime/length_deps -r requirements-length.txt
python -m pip install --no-deps --target .runtime/length_models -r requirements-spacy-model.txt
```

`scripts/run_amg_lengths.sh` 默认从 `.runtime/length_deps` 加载附加包，并使用 `.runtime/length_models/en_core_web_sm/en_core_web_sm-3.5.0`。可通过 `T2M_LENGTH_DEPS`、`T2M_SPACY_MODEL` 指定现有目录；这样可以继续使用原服务器已有依赖。不要向原训练环境直接执行升级安装。新建的 `.runtime/` 不进入Git。

单独运行长度相关Python脚本时，同样需要附加目录优先：

```bash
PYTHONPATH="$PWD/.runtime/length_deps${PYTHONPATH:+:$PYTHONPATH}" \
  python scripts/predict_amg_lengths.py --help
```

`--no-deps` 用于保持已记录的主环境及完整附加包列表，避免解析器引入新的传递依赖版本。请使用新的空目标目录；已有目录若混有其他版本，应保留原目录并换一个新目录安装。两组附加目录按各自任务使用，本轮没有验证将两组叠加的第三种环境。

spaCy模型wheel地址及哈希保留自原有环境记录，当前没有重新下载核验。数据集、CLIP/TMR等模型权重、归一化统计和训练产物仍按各自许可单独准备，参见 [数据与模型](data-and-models.md)。
