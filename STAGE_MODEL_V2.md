> 历史开发记录。当前 GitHub 安装步骤见 [模型说明](docs/MODELS.md)，复算输入及代码入口见 [训练说明](docs/TRAINING.md)。历史交接包中的私有数据与实验输出不随 GitHub 源码发布。

# G1 多模态时序阶段模型 v2

在“进展”面板的模型列表选择 **G1 香蕉搬运 · 多模态时序模型 v2（研究版）**。现有 18 段 G1 录像已评分；新录像先解析，再点击计算进展。旧版 `g1-stage-v1` 和 `open-dinov2-pilot` 继续保留。

任务限定为从左桌抓香蕉模型，搬运到右桌并放入篮子。模型识别接近、抓取、搬运、放置、释放后观察五个阶段。“释放后观察”不等于任务成功，阶段进度也不是成功概率或 WARP 速度。输入覆盖、低把握或模型分歧提示帮助定位需要查看视频的片段，正式审核仍由人保存。

模型结合冻结 DINOv2 的头部左目与右腕特征、29 个身体关节、双手各 6 个实测状态，以及最近 8 个 2 Hz 采样点。两层 Transformer 和阶段交叉注意力联合预测阶段及阶段内进度。融合 SARM 和 FACT 的部分思路，属于针对本项目的实现，不是原论文完整复现。

当前五折开发验证宏 F1 为 93.68%，同条件重训的双视角 MLP 为 90.34%。历史 5 段回归的 70 个阶段点中，正确数由 63 提高至 65，但仍有两个高把握度错误。没有新的独立测试录像；研究标注未通过独立人工验收。详情与消融见 [升级验证报告](research/stage_v2_20260917/升级验证报告.md)。

## 输入缺失和缓存

关节按当前或更早的最近值对齐，100 ms 以前的状态视为缺失。名称或单位不匹配会报错，防止把错误的关节顺序送入模型。一个相机或关节状态缺失时可输出降级估计，但始终提示复核；两个相机都无有效图像时不输出阶段。

模型配置、源录像指纹、关节缓存指纹、相机/状态描述共同参与 v2 缓存身份；推理前后检查源数据是否变化。v1 与 WARP 的缓存接口保持兼容。阶段模型继续禁止用作 WARP 动作块权重。

## 复算和复现

命令从项目根目录 `人形移动操作demo`（交接包解压后的根目录）执行。依赖版本见研究目录 `environment.json`；需要已有的 `robot-data-studio/workspace/warp-env` 环境，交接包不附带虚拟环境。

仅复算已保存特征，无需原始 MCAP：

```bash
robot-data-studio/workspace/warp-env/bin/python research/stage_v2_20260917/verify_delivery.py
```

完整训练实验已冻结，脚本拒绝覆盖已完成的选型。复训时复制该研究目录为新的兄弟目录，例如 `stage_v2_reproduction`，保留 `protocol.json`、`freeze.json`、`state_manifest.json` 和 `states/`，为新目录创建空的 `training/`，然后运行新目录的 `train_cv.py`。原始研究目录及其 `training/` 不应删除或改写。脚本会校验旧研究标签和特征 SHA256，完成 90 次交叉验证拟合，再选型并作 3 次最终拟合。历史回归仅在选型固定后评估。

`prepare.py` 只用于从原始解析缓存重建输入与冻结协议；已有冻结协议时会拒绝覆盖。`finalize_calibration.py` 是本轮初次进程的校准顺序修正记录，修正已纳入最终 `train_cv.py`，新复训不需要再次运行它。

工程入口：`warp_progress/stages_v2.py`、`state_features.py`、`stage_worker_v2.py`；评分调度在 `progress_service.py`。模型登记由研究目录 `register_model.py` 完成，它拒绝静默覆盖相同模型 ID。

项目测试（在 `robot-data-studio` 目录）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. workspace/warp-env/bin/python -m pytest tests -q
node --test tests/*.test.mjs
```

生成图表使用系统 Python 的 Matplotlib：`python3 research/stage_v2_20260917/plot_results.py`。不需要 Hugging Face 在线登录或 DINOv3 授权。
