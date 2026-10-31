# 数据集说明、获取与引用

本页说明当前仓库的主要数据集 HumanML3D，以及代码中仍保留的 KIT-ML、BABEL-TEACH 数据入口。数据集文件、原始动作、预训练权重、派生缓存均不随本代码仓库分发。资源目录和模型文件位置见[数据与模型清单](data-and-models.md)及[`resources/manifest.json`](../resources/manifest.json)。

## HumanML3D：本仓库的主要数据集

HumanML3D 是文本与三维人体动作配对数据集。上游说明其动作来自 AMASS 和 HumanAct12；动作按 20 FPS 处理，单段约 2–10 秒，并提供自然语言描述。上游 README 报告 14,616 段动作和 44,970 条描述，并说明镜像增强会增加动作样本。实际训练样本数还会受数据版本、官方划分和本项目过滤规则影响，不应仅根据 README 中的总数推断。

- 官方代码、数据格式和处理说明：[EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D)
- 官方数据获取章节：[How to Obtain the Data](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data)
- HumanML3D 原论文：[Generating Diverse and Natural 3D Human Motions From Text](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html)
- AMASS 获取与许可：[下载页](https://amass.is.tue.mpg.de/)、[许可条款](https://amass.is.tue.mpg.de/license.html)

### 获取和生成数据

HumanML3D 上游说明受 AMASS 的分发政策限制，不直接分发 AMASS 动作数据，并提供从来源数据重建 HumanML3D 的脚本。请先阅读 HumanML3D 上游说明及 AMASS 当前许可，再从 AMASS 官方页面注册并获取有权使用的原始数据；HumanML3D 上游说明还提到 HumanAct12，具体重建所需的每项输入应以选定的上游版本和 notebook 为准。重建时按上游所列顺序使用 [`raw_pose_processing.ipynb`](https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb)、[`motion_representation.ipynb`](https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb)、[`cal_mean_variance.ipynb`](https://github.com/EricGuo5513/HumanML3D/blob/main/cal_mean_variance.ipynb)，并按上游要求单独取得相应身体模型。处理最好在独立、空的目录中进行；本仓库不提供数据自动下载器，也不要对正在使用的实验目录运行旧下载脚本。

AMASS 条款将授权限定在个人、非商业研究等用途，并限制向第三方复制或分发。HumanML3D 代码仓库中的 MIT 文件是代码仓库的许可证，不能据此推断 AMASS、HumanAct12、身体模型或生成的派生数据也可以按 MIT 重新分发。若需要提供派生动作特征、缓存或样例给评审者或公众，应逐项核对来源数据条款，必要时取得书面许可；不要把下载到的动作数据上传到本仓库、网盘或模型托管站。

重建结束后，将**与本项目输入表示一致**的处理结果放在项目根目录下：

```text
dataset/HumanML3D/
├── new_joint_vecs/
│   └── <motion_id>.npy
├── texts/
│   └── <motion_id>.txt
├── train.txt
├── val.txt
└── test.txt
```

本项目主要训练入口读取 `new_joint_vecs` 和 `texts`：动作特征应为有限值、形状 `(T, 263)`；文本与动作应使用相同 ID。HumanML3D 上游还会生成 `Mean.npy`、`Std.npy` 等统计文件，但本项目现有 VQ/GPT 流程从所选模型资源目录读取 `mean.npy`、`std.npy`。它们必须与该表示、评估器和 checkpoint 身份配套，不能把不同版本或 TMR 的统计文件互相替换。确切模型资源位置见[数据与模型说明](data-and-models.md)。

目前没有经过本项目重新验证的端到端数据重建脚本。上游 README 提到特定 Matplotlib 版本可能影响重建结果；因此不要随意重跑预处理后就把结果当作与历史实验完全相同。请记录 HumanML3D 上游 commit、AMASS 来源子集、处理依赖版本、输出文件哈希和数据 split。

### 数据格式和本项目中的处理注意事项

- **采样率和表示不可混用。**HumanML3D 输入按 20 FPS，当前训练数据类使用 22 关节、263 维特征；原始 SMPL 参数、关节坐标、渲染视频都不能直接替代 `new_joint_vecs`。
- **保留官方划分。**训练、验证、测试必须使用各自的官方 ID 文件。所有派生长度记录、特征和文本缓存都应保留来源 ID 和 split，并检查 split 之间无重叠。不要把 `train_val.txt` 当成验证集。
- **镜像动作要与文本一同处理。**上游以 `M` 前缀标识镜像动作，并对左右、顺逆时针等词进行对应调整。新建划分时应将原始动作与镜像版本留在同一 split，避免近重复动作跨训练/测试集。
- **文本行包含时间标记。**`texts/<id>.txt` 的每行由原始描述、词性标记、起始时间和结束时间组成，以 `#` 分隔。起止时间为 `0, 0` 表示整段动作；非零时间表示只描述动作片段。不要把这些字段删掉或改写格式。
- **AMG/TMR 配对准备只取整段描述。**`preprocess/prepare_amg_tmr.py` 默认只将起止时间均为零的描述作为整段配对；非零时间的局部描述会记录为排除项。缺失 TMR token embedding 也要查看 audit，不能静默丢弃后仍声称使用了完整训练集。
- **长度特征有额外过滤。**`preprocess/prepare_amg_lengths.py` 默认要求动作至少 64 帧，并仅保留整段描述；输出应与 train/val/test split 和 `audit.json` 一起管理。
- **VQ 训练另有长度筛选。**`src/datasets/dataset_tokenize.py` 会排除少于 40 帧或达到 200 帧及以上的动作，再按 `unit_length` 向下取整并随机采连续片段。当前长度控制入口的上限为 196 帧（49 个动作 token）。报告数据量时应注明实际过滤后的样本数。
- **缓存需锁定来源身份。**TMR token embedding 的文本索引、TMR motion/text 权重、归一化统计、VQ token 缓存和 GPT checkpoint 必须互相匹配。不能仅凭文件名判断一致；记录上游版本和 SHA256，运行已有 audit 并保留输出。

### HumanML3D 引用

使用 HumanML3D 时至少引用其原论文。若重新处理或使用 AMASS 来源动作，还应按实际使用的 AMASS 子数据集及许可要求引用；AMASS 官方页面提供其组成数据集的 BibTeX 清单。

```bibtex
@InProceedings{Guo_2022_CVPR,
  author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
  title     = {Generating Diverse and Natural 3D Human Motions From Text},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2022},
  pages     = {5152--5161}
}
```

AMASS 的论文引用（仍需按实际使用的 AMASS 组成数据集补充对应引用）：

```bibtex
@InProceedings{Mahmood_2019_ICCV,
  author    = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  title     = {{AMASS}: Archive of Motion Capture as Surface Shapes},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2019},
  pages     = {5442--5451}
}
```

## 代码中保留的其他数据入口

### KIT-ML

KIT-ML 是可选数据集，不是当前文档化 AMG 主流程的默认数据。官方数据页提供版本化下载和引用信息：[KIT Motion-Language Dataset](https://motion-annotation.humanoids.kit.edu/dataset/)。代码中的旧数据入口期望 HumanML3D 流程处理后的 `dataset/KIT-ML/new_joint_vecs`、`texts`，并使用单独的 KIT 统计量和 checkpoint；原始 KIT `.c3d` 或 `.mmm.xml` 文件不能直接放入该路径代替处理特征。KIT 使用 21 关节、251 维表示，采样率约 12.5 FPS，与 HumanML3D 的 22 关节、263 维、20 FPS 设置不可混用。

使用 KIT-ML 时引用原始数据集论文：

```bibtex
@article{Plappert2016,
  author    = {Plappert, Matthias and Mandery, Christian and Asfour, Tamim},
  title     = {The {KIT} Motion-Language Dataset},
  journal   = {Big Data},
  publisher = {Mary Ann Liebert Inc},
  year      = {2016},
  month     = {December},
  volume    = {4},
  number    = {4},
  pages     = {236--252},
  doi       = {10.1089/big.2016.0028}
}
```

下载时选定并记录官方发布版本；KIT 官方页面目前列有多个日期版本，样本和标注计数不同。不要在未记录版本的情况下对比结果或报告数据统计。官方页面可下载不等于本项目获得再分发许可，仓库不托管 KIT 数据。

### BABEL 与代码中的 `BABEL_TEACH`

代码还保留 `BABEL_TEACH` 的历史数据路径（`../Dataset/BABEL_TEACH`，位于仓库根目录之外），但它不是官方 BABEL 原始文件的直接目录。现有代码预期 `BABEL_TEACH_joint_vecs_2`、`BABEL_TEACH_texts`、派生文本目录以及 `train_12103.txt`、`val_2163.txt`、`test_2000.txt` 等自定义 split 文件；当前仓库没有记录这套数据的确切来源版本、转换脚本、split 身份和权利范围，也没有把它纳入 HumanML3D 资源清单。因此不要把官方 BABEL 下载解压后直接改名成 `BABEL_TEACH`，也不要将该入口描述为已经验证的复现流程。

如需使用官方 BABEL，请从[官方数据页](https://babel.is.tue.mpg.de/data.html)获取，并完整阅读[官方许可](https://babel.is.tue.mpg.de/license.html)。BABEL 许可将用途限定在非商业研究等范围，并限制向第三方分发；不能据此上传数据、派生数据或权重到 GitHub。官方数据分为 dense 标注和 `extra_train`/`extra_val` 附加标注，`extra_test` 未公开；若使用附加标注，论文或报告必须说明。使用 BABEL 应引用原论文：

```bibtex
@InProceedings{Punnakkal_2021_CVPR,
  author    = {Punnakkal, Abhinanda R. and Chandrasekaran, Arjun and Athanasiou, Nikos and Quiros-Ramirez, Alejandra and Black, Michael J.},
  title     = {BABEL: Bodies, Action and Behavior With English Labels},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2021},
  pages     = {722--731}
}
```

## 数据准备记录建议

每次实验至少保存以下信息：

- 数据集名称、官方来源 URL、下载日期、上游 commit 或官方发布版本；
- 原始来源（例如 AMASS 的具体子数据集）、许可接受记录，以及预处理代码和依赖版本；
- 官方 split 文件的哈希、样本 ID 数、过滤规则、过滤前后计数；
- 使用的动作表示、FPS、维数、镜像策略、均值/标准差文件 SHA256；
- 派生缓存的输入文件哈希、生成命令、`audit.json`、输出哈希；
- 论文表格、评估 checkpoint 和每个结果对应的数据 split。

如数据来源、许可或派生处理无法确认，应在资源清单中标记为未核实，不应将数据上传或声称可由他人直接下载复现。
