# 开源质量规则来源与融合范围

本项目没有将外部项目描述为经过行业认证的 A–D 分级标准。HFlow 提供可复用测量与可配置策略；trajlens 主要评估数据结构/完整性；robomimic 提供不同熟练度与成功/失败示范，用于检验质量假设。

## 来源

| 项目 | 已核对实现 | 采用或保留的内容 |
|---|---|---|
| [HFlow](https://github.com/Hebbian-Robotics/hflow) | `src/hflow/checks.py`，Git blob `296c68ed52aaf186d1ce6d2f69b4e44eb1b11a8b` | 原始时间间隔测量（默认 10 ms 容差、3 倍周期长间隔）、有限数值/重复值/未变化维度、轨迹速度与变化量。运动指标仅作参考 |
| [trajlens](https://github.com/Kunal-Somani/trajlens) | `checks/temporal.py`，blob `f14ce1c7a56a490ecb1597436f1f6bf1107fcafb` | 固定帧率间隔检查：偏差 >0.1 ms 提醒，超过一个帧周期严重异常；仅对导出网格使用 |
| trajlens | `report/trust_score.py`，blob `4c7180405b4f4661a0105c7417b390d17a121c2a` | 完整性参考分：FAIL 每项 30、最多 60；WARN 每项 5、最多 20；ERROR 每项 10；质量层的运动提醒不计分。只在实验中使用，不替换正式 A–D |
| [robomimic](https://github.com/ARISE-Initiative/robomimic) | 官方 `docs/datasets/robomimic_v0.1.md`，blob `5943540c867330ddabf418fb08e740364c4eaaa2` | 使用公开 Can Multi-Human 与 Paired 数据进行验证，不把操作者熟练度当作逐条任务成败标签 |

## 有意做的改进

1. **完整扫描时间轴**：trajlens 的单个 spacing 检查遇到首个轻微警告即停止当前 episode 的扫描；融合实现检查全部间隔并优先报告后续严重异常。这不代表整个 trajlens 缺少其他可能发现异常的检查。
2. **区分数据阶段**：固定帧率导出的 0.1 ms 容差不用于原始 MCAP。原始多传感器时间测量借鉴 HFlow 的 10 ms 容差，同时保留实际周期和长间隔数量。
3. **不跨缺测重新连接轨迹**：运动计算屏蔽坏值和超过 100 ms 的间隔；时间倒退不提供误导性运动数值。100 ms 来自本项目现有对齐约束，并非 HFlow 的默认阈值。
4. **未知不算满分**：没有测量成功的完整性检查时返回“未测量”，不返回 100。部分检查跳过时注明覆盖不足。
5. **重复值不直接删除**：HFlow 的精确重复与未变化维度测量被保留为事实；G1 静止腿部、保持抓握、量化手部读数与恒定目标均可能合理，因此不把它们直接判断成采集故障。
6. **运动平滑度不决定优劣**：公开实验显示单一平滑度分档不能可靠跨操作者泛化；候选方案未并入正式数据等级。

## 当前未采用的内容

HFlow 还公开了相机黑屏/冻结检查，推荐策略包含黑帧比例不超过 50%、冻结总时长不超过 2 s，检测参数另包括 98% 黑像素占比、亮度阈值 17 和 -60 dB 冻结噪声容差。这些参数针对其 FFmpeg/视频亮度处理流程，不可直接当作 RGB 像素阈值套到 JPEG。

本轮下载的是无图像的 low-dimensional 公开数据，未完成这一相机策略的迁移验证，因此本轮没有把黑屏、模糊或冻结判定加入正式分级。也没有调用 HFlow 的托管模型服务、发送本地记录或连接模型 API。

## 许可与复现

HFlow 和 trajlens 的代码许可均为 Apache-2.0，完整许可保留在 `static/licenses/hflow.txt` 与 `static/licenses/trajlens.txt`。`quality_evidence.py` 说明了改写范围。本地原文快照与测试用 AST 函数载入均核对 Git blob SHA，实验只运行明确列出的函数，不导入或启动上游托管服务。

`experiments/quality_validation/sources.json` 保存十份源码、测试和许可证的下载地址/哈希；`fetch_sources.py` 可恢复同一批快照。数据版本、长度与上游 LFS SHA-256 在同目录的 `datasets.json`。实验过程与边界见根目录 `QUALITY_VALIDATION_REPORT.md`。
