> 历史开发记录。当前 GitHub 安装步骤见 [模型说明](docs/MODELS.md)，复算输入及代码入口见 [训练说明](docs/TRAINING.md)。历史交接包中的私有数据与实验输出不随 GitHub 源码发布。

# v4 论文扩展研究：继续使用已验证的 v3

本轮测试了跨录像对比学习、区间边界与片段约束、两者组合、阶段与进度梯度隔离。没有候选达到事先冻结的升级门槛，所以 **没有安装 `g1-stage-v4`**。实际评分请继续选择“G1 香蕉搬运 · 运动感知模型 v3（研究版）”。

开发五折宏 F1：v3 为 95.14%；前两个单项仍为 95.14%；组合为 94.33%；梯度隔离为 93.03%。完整依据见 `research/stage_v4_20260917/实验与验证报告.md`，论文与实现范围见同目录 `论文研究与落地决策.md`。这些是现有小样本上的研究指标，不能作为新场景准确率或最终成功率。

## 当前可用功能

v3 的阶段曲线、人工复核提示、视频定位与原有 WARP 功能继续保留。新增的神经网络辅助目标和 v4 推理接口属于研究基础设施，没有通过登记向日常评分暴露未达标权重。没有自动修改录制成功标签或人工审核。

## 实验复算

从项目外层目录，在装有 NumPy、PyTorch 的环境中运行：

```bash
robot-data-studio/workspace/warp-env/bin/python research/stage_v4_20260917/verify_experiments.py
robot-data-studio/workspace/warp-env/bin/python research/stage_v4_20260917/verify_delivery.py
```

交接包不包含虚拟环境，移动到其他机器时使用当地安装好依赖的 Python。这两个脚本始终从当前包内读取特征、状态数组和模型，不访问原始录像；后者同时核对现行 v3 的全部 18 段输出。图像重提取和完整 UI 视频播放仍需要原始数据。

初始研究脚本为 `research/stage_v4_20260917/train.py`；后续梯度隔离实验为 `research/stage_v4_decoupled_20260917/train.py`。现有选择结果已经冻结，脚本会拒绝覆盖。重新开展研究应使用单独的输出目录和明确的新协议，保留原实验记录；不要删除旧选择结果后无痕覆盖。

原始 `training/selected.pt` 只是该研究内部的候选检查点，其 `promotion_eligible` 为 false，不是已发布模型。正式使用和复训 v3 的说明仍见 [STAGE_MODEL_V3.md](STAGE_MODEL_V3.md)。
