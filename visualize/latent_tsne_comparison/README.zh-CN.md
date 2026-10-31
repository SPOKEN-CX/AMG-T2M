**VQ 潜空间分析源码**

本目录保留普通 VQ 与 TMR 引导 VQ 的特征分析、样本筛选、t-SNE 绘制及核验工具。`artifacts/` 内的动作样例、缓存、探索性权重和内嵌动画页面未包含在私有代码快照中。

主配置见 `config.json`，入口为 `run.sh`。先安装仓库根目录的主环境依赖 `requirements.txt`，依法准备数据、统计量和指定 checkpoint，再检查命令参数；可用 `MOTION_TSNE_PYTHON` 指定解释器。

```bash
bash visualize/latent_tsne_comparison/run.sh --help
```

配置中的历史训练批量、迭代数和路径只描述被选定的工程对照，不代表本次发布重新测得结果。关键词自动筛选的类别需要核验；探索性选择不能替代冻结协议和正式统计。每张拟公开的图应绑定样本清单、配置、模型哈希及实际渲染脚本。依赖旧 artifacts 的补充脚本需要先重建对应输入。
