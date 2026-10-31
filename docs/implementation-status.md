# 源码范围与复现入口

2026-10-31 已将本地项目的源码基准收录到 [reproduction/local/](../reproduction/local/README.md)。本地模型、数据加载器、VQ/GPT 单卡及多卡训练、正式评估、长度模块、TMR 辅助入口、渲染源码、依赖与运行指南按原目录结构保存。

复现本地工程时，从 `reproduction/local/` 运行并使用该目录的指南；仓库根目录 `src/`、`preprocess/`、`tools/` 为整理版工作流，按 [workflow.md](workflow.md) 从仓库根目录运行。两套目录的源码版本与导入方式不应交叉使用。

本轮来源与内容身份记录在 [source-manifest.json](../reproduction/source-manifest.json)。[审查报告](local-source-sync.md) 说明收录范围、已发现的历史入口/资源路径问题，以及本轮实际执行的检查。

原快照中未收录的 LibMTL、SMPL 相关渲染、TMR 辅助入口等源码，现在包含在本地复现基准中；各文件原有许可头保留。所有数据集、模型/身体模型权重、先验、缓存和运行结果均由使用者单独准备。资源获取入口见 [数据说明](datasets.md) 与 [本地源码 README](../reproduction/local/README.md)。

本轮执行源码、路径与发布内容检查，没有重新训练模型或计算论文指标。历史命令记录中的缺失入口、空文件与重复入口均逐项记录在审查报告中。
