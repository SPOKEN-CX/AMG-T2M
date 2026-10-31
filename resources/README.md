**外部资源目录**

[manifest.json](manifest.json) 汇总当前HumanML3D链路使用的38个资源类别或具体路径。它描述目录与入口之间的关系，供资源准备和迁移时查阅；当前不提供自动下载或运行时验收。

`origin` 中，`external` 表示需单独合法获取，`external-or-regenerated` 表示外部缓存或按上游流程重建，`generated` 表示由准备脚本生成，`trained` 表示训练产生。`stages` 用于导航，基础数据还需与相应训练/评估阶段一起准备。未列出的辅助文件应继续参照具体入口。HumanML3D、KIT-ML 与 BABEL 的引用、获取、许可和数据处理注意事项见[数据集说明](../docs/datasets.md)。

`working_reference.sha256` 表示选定工作文件的参考身份。均值/方差来自现有小文件的实际哈希；CLIP和冻结VQ保留已有manifest中的记录，本轮没有重新读取大权重计算。它们均不自动证明上游下载档身份，也不替代资源许可。

权重、动作/文本数据、token或embedding缓存、实验输出均不放入本目录。训练后的GPT通过调用参数选择，并应保留其对应的 `vq_identity.json`。尚未建立公开下载渠道的资源不填虚构链接。完整准备说明见 [数据与模型](../docs/data-and-models.md) 与 [脚本顺序](../docs/workflow.md)。
