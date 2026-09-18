> 历史开发记录。当前 GitHub 安装步骤见 [模型说明](docs/MODELS.md)，复算输入及代码入口见 [训练说明](docs/TRAINING.md)。历史交接包中的私有数据与实验输出不随 GitHub 源码发布。

# G1 香蕉搬运阶段模型

新增 profile `g1-stage-v1`。在录像审核页“进展”面板的模型菜单中选择“G1 香蕉搬运 · 双视角阶段模型（研究版）”。现有 WARP 模型继续保留。

输入为头部左目、右腕相机的冻结 DINOv2-S/14 特征。验证集选中的双视角 MLP 输出阶段分类和阶段条件进度，三个种子预测平均，并用验证集温度校准。研究同时训练了两层 Transformer 和倒放增强候选。详见 [研究与实验报告](research/model_upgrade_20260917/模型升级报告.md) 和 [论文取舍](research/model_upgrade_20260917/论文研究与落地决策.md)。

五种状态为：接近左桌、抓取香蕉、持物搬运、向篮子放置、释放后观察。最后一种状态不等于成功入篮。进度为阶段标注对齐的代理值，不是成功概率，也不使用 WARP 的速度倍数单位。缺相机/缺帧会保留未知，阶段模型不支持 WARP 动作块权重导出。

实际训练使用 8 段录像，验证 2 段，留出测试 5 段；另 3 段为任务变体与恢复样例。标注来自助手对稀疏原始画面的视觉检查，尚未独立人工验收。同一场次测试的阶段准确率为 90.0%（63/70），不代表跨场景效果或任务成功率。

代码路径：`warp_progress/stages.py` 模型及通用推理，`stage_worker.py` 实际 MCAP 分析，`progress_service.py` 队列及数据约定，`static/stage-data.mjs` 和 `progress-view.js` 页面。

重新提取和训练（从项目外层目录运行，原始 MCAP 或已提取特征需存在）：

```bash
robot-data-studio/workspace/warp-env/bin/python research/model_upgrade_20260917/extract_features.py
robot-data-studio/workspace/warp-env/bin/python research/model_upgrade_20260917/train_stages.py --output research/model_upgrade_20260917/reproduction_run
```

训练程序核验冻结协议与标注、检查源文件内容在集合之间没有重叠，只按验证集选型。现有测试报告不会被覆盖。重新训练只是复算同一开发实验，不能把已见过的测试数据重新称为独立验证。研究目录中的完整特征缓存来源、数据校验值、模型权重、每个种子的训练记录和原始报告均可追溯。
