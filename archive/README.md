# 原项目归档

`vista4d/` 保存切换至官方 Manifold4D 前的完整项目快照，包括 Vista4D、FlowLong、DiffSynth、DA3、Pi3、SAM3、相机 UI、旧 JEPA 代码、旧 Manifold4D 移植实现及其测试。

- [原 README](vista4d/README.md)
- [原长视频说明](vista4d/READMEv2.md)
- [原流水线说明](vista4d/README_pipeline.md)
- [原 Manifold4D 移植实现说明](vista4d/docs/manifold4d_baseline.md)
- [原配置](vista4d/configs/)
- [原依赖](vista4d/requirements.txt)
- [快照清单](vista4d/snapshot.json)：归档前 Git commit，以及原有全部受 Git 跟踪文件的 SHA-256。

归档文件保持原内容。归档用于历史查阅与后续按需迁移，内部命令和路径对应原项目布局；不作为当前 baseline 的安装、训练或推理入口。

当前研究文档仍保留在根目录 `docs/`。本地 `datasets/`、`checkpoints/` 及论文下载目录保持原位置，不复制进归档。归档中的测试不参与默认 pytest 搜索。
