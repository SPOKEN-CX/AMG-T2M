> Dependency maintenance: all active Python dependencies now use the root `requirements.txt`. Original dependency inputs are preserved verbatim in [dependency-history.json](dependency-history.json). Source hashes and archive locations are updated in the source manifest; numerical byte-identity counts below describe the initial source sync.

# 本地源码同步审查（2026-10-31）

本次以本地项目目录为源码依据，审查本地文件、运行入口、导入关系、依赖清单、配置与资源路径，再将可发布的源码内容归档到 `reproduction/local/`。原工作目录未改动；GitHub 原有 `src/`、`tools/` 工作流继续保留。

## 覆盖范围

- 本地扫描文件总数：45,039。
- 源码、配置和文档候选：239 项，其中包含额外识别出的 7 份模型/评估器文本 `opt.txt` 和原 `.gitignore`。
- 收录本地文件：200 项，共 915,409 字节，包含 145 个 Python 文件。
- 候选中排除：39 项，主要是数据标注/索引 JSON、运行指标、Hydra 运行输出及临时测试/探测脚本；逐项原因保存在清单中。
- 模型、数据加载器、训练与正式评估入口、TMR 编码/检索工具、特征处理、LibMTL、SMPL 相关渲染源码、配置、依赖和运行指南均在收录范围。

完整来源、目标路径、原始/上传 SHA256 和排除项见 [source-manifest.json](../reproduction/source-manifest.json)。其中 197 个本地文件内容逐字保留，三份中文运行指南只清理个人路径、服务器账号并修正本地绝对文件链接。原根 README 改名为 `README.upstream.md`，原 `.gitignore` 保存为 `.gitignore.original`。新增的源码入口 README、路径配置工具及本报告不计入 200 个本地文件。

`dataset/*.py` 是源码，应与数据集文件区分。`checkpoints/**/opt.txt` 是可读模型配置，应与权重区分。本次仅收录这两类所需文本内容，没有上传动作、caption、split、annotation、embedding、统计张量、词向量或 checkpoint 二进制。

## 目录和运行方式

完整本地结构保存在 [reproduction/local](../reproduction/local/README.md)，以该目录为运行工作目录。`models/Retrival`、`models/MotionLenEstimator` 等大小写和拼写原样保留。根目录 `src/`、`tools/` 是另一套整理版入口，不与本地基准交叉导入。

## 审查发现与处理

| 发现 | 处理及使用方式 |
|---|---|
| 旧机器绝对路径、`../Dataset` 和本地 `dataset/` 路径混用 | 源码保持原样；提供 [configure_paths.py](../reproduction/configure_paths.py)，使用清单限定文件，只配置 Python/JSON 中已知的资源路径字符串。默认仅预览，`--write` 后生效。 |
| 初始本地演示依赖与主环境版本不同 | 历史依赖已归档到 `dependency-history.json`；当前统一使用根目录 `requirements.txt`，包含共同解析和固定的版本。 |
| `train_transition.py` 与 `train_t2m_trans.py` 字节相同 | 两者均保留，以便核对历史命令；按内容识别为同一 GPT 训练入口。 |
| `train_GPT_split.py` 为零字节 | 保留并明确记录，不将它列作可运行训练器。 |
| `run_order` 引用 `gen_long_seq.py`、`train_length_est.py`，本地根目录无这两个文件 | 历史命令记录原样保留；不虚构缺失入口。根目录整理版的长度训练入口实际为 `tools/train_length_est.py`，应按整理版工作流运行。 |
| `models/MotionLenEstimator/LengthEstTrainer.py` 引用本地缺失的 `utils/utils.py` | 保留源码并记录依赖缺口；不把这个历史 trainer 的成功导入当作已验证事实。 |
| `dataset/prepare/download_extractor.sh` / `download_glove.sh` / `download_smpl.sh` 会删除资源目录；extractor 还固定代理；`download_model.sh` 创建 `pretrained` 却进入 `pretrained1` | 归档原始脚本供核对，但未执行。资源准备使用官方获取说明，不能直接对已有实验资源运行这些历史下载脚本。 |
| TMR app、特征转换、渲染依赖外部 annotation、样例骨架或 SMPL 先验资源 | 源码与配置收录；数据、样例张量、身体模型、先验和权重只由使用者按上游说明获取。 |

这些是对具体文件和历史入口的审查结果。项目训练与指标复现应使用选定指南中相互匹配的代码、环境、数据、缓存、VQ/GPT 和评估资源。

## 检查范围

- 对收录的 145 个 Python 源文件进行语法解析；无语法错误。
- 对 shell 资源脚本执行语法检查，未执行下载或删除命令。
- 对新增路径配置工具在临时副本中检查预览不写入、实际替换、含空格/引号路径、重复运行、配置后语法和原始文件不被修改。
- GitHub 发布前逐文件计算 Git blob SHA，发布后核对提交差异、树中路径及 blob 身份。

本次没有重新执行 GPU 训练、完整推理或论文指标评估，因此不将源码同步与这些运行结果等同。数据与资源仅提供 [官方获取链接](../reproduction/local/README.md#数据与外部资源)。
