"""Render a reviewable report from saved experiments and acceptance evidence."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1]
def read(p):return json.loads(Path(p).read_text())

def main():
    cv=read(ROOT/'training/cross_validation.json');r=read(ROOT/'training/report.json')
    selection=r['selection'];diagnostics=read(ROOT/'diagnostics.json');profile=read(ROOT/'installed_model.json')
    runtime=read(ROOT/'runtime_verification.json');assert runtime['passed']
    checks={k:read(APP/'workspace/evidence/stages-v2'/k/'report.json') for k in ['live','v1-regression','warp-regression']}
    assert all(v['passed'] for v in checks.values())
    names={'baseline_mlp':'双视角 MLP（同条件重训基线）','baseline_temporal':'双视角时序 Transformer',
        'fusion_mlp':'双视角＋关节状态 MLP','fusion_temporal':'双视角＋关节状态时序 Transformer',
        'fusion_tokens':'多模态时序＋阶段交叉注意力（选中）','fusion_tokens_aug':'上述模型＋缺输入/速度增强'}
    table='\n'.join(f"| {names[k]} | {100*v['overall']['macro_f1']:.2f}% | {100*v['overall']['accuracy']:.2f}% | {v['overall']['nll']:.4f} |" for k,v in cv.items())
    missing='\n'.join(f"| {label} | {v['n']}/70 | {100*v['accuracy']:.2f}% | 全部提示复核 |" if v['accuracy'] is not None else f"| {label} | 0/70 | 不输出阶段 | 保留未知 |"
        for key,label in [('head','整段移除头部图像'),('wrist','整段移除右腕图像'),('state','整段移除关节状态'),('all_cameras','移除全部图像')]
        for v in [r['controlled_missing_inputs'][key]])
    ci=diagnostics['recording_bootstrap']['percentile_95_interval']
    text=f'''# G1 多模态时序阶段模型 v2：实装与验证

已在现有 HoloCurate 中安装 `g1-stage-v2`，并完成 18 段原有录像的实际评分。新版用双视角图像、29 个身体关节和双手各 6 个状态通道判断香蕉搬运阶段；保留旧阶段模型和 WARP 速度模型。升级改善了这批录像上的阶段识别，但尚不能可靠判断香蕉是否真正入篮，也没有证明下游机器人策略成功率提高。

## 采用的论文思路

- [SARM，ICLR 2026](https://qianzhong-chen.github.io/sarm.github.io/)：[正式论文](https://openreview.net/pdf/a003594857c5bc6d40ac553e082c79c1dcd1f921.pdf)。延续阶段分类、阶段条件进度和训练集时长先验，本轮进一步加入实测关节/手部状态与时间历史。原文使用冻结 CLIP 和更大的网络；本项目使用合法可用的冻结 DINOv2。
- [FACT，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html)。借鉴视频帧与动作 token 交叉注意力，在本任务中用 5 个可学习的阶段 token 与近期历史交互。没有实现原论文完整帧分支、匹配损失、训练设置或基准复现。
- 原来的 [WARP-RM](https://arxiv.org/abs/2606.28320) DINOv2 方法适配继续保留。v2 阶段进度不是 WARP 的速度，也不作为 WARP 动作块权重导出。DINOv3 官方授权问题没有被绕过。

这是对论文方法的任务适配与有对照实验的工程迭代，不是上述论文原始成绩的完整复现。

## 实际网络与训练

```text
头部左目 → 冻结 DINOv2-S/14 → 384 维 → 64 维 ┐
右腕相机 → 冻结 DINOv2-S/14 → 384 维 → 64 维 ├→ 96 维融合
29 身体关节＋12 手部状态 → 32 维＋有效性标记 ┘
  → 最近 8 个采样点的两层 Transformer（4 个注意力头）
  → 5 个阶段 token 与历史的交叉注意力
  → 阶段分类头 ＋ 依赖阶段概率的阶段内进度头
  → 3 个独立训练模型集成、把握度与分歧提示
```

采样率 2 Hz，8 个点覆盖当前到 3.5 秒前；启动时用首点补齐历史。输入不包含绝对时间、录像总长或采集成功标记。状态按“当前或更早的最近值”取样，最大允许过期 100 ms，单位或关节顺序不符会拒绝评分。模型不读取未来帧。

每个可训练网络 {r['final_training'][0]['parameters']:,} 个参数，DINOv2 保持冻结。本轮拟合 6 个候选 × 5 折 × 3 个种子，另对选中模型作 3 次最终拟合，共 93 次训练；每次 300 步、批量 64、AdamW、学习率 0.0003。损失为阶段交叉熵＋0.5×阶段内进度 MSE。固定平均第 200、250、300 步权重，不用留出折挑训练步数。

## 验证边界与选型

按用户要求只使用已有录像。标签沿用上一轮稀疏视频抽查形成的研究标注，未追加模型伪标签、未获得独立人工验收。18 段同一采集时段录像中：10 段开发数据做五折交叉验证，5 段旧测试录像只作历史回归，3 段变体/恢复录像保留作压力样例。

每折 8 段训练、2 段留出，整段录像不跨训练和留出集合；归一化及阶段先验仅拟合训练录像。139 个共同有效标注点用于比较全部候选，原始内容哈希排除了同内容重复录像。协议与标签哈希在训练前冻结。

预定升级门槛：新多模态模型宏 F1 至少超过重训 MLP 0.02，且不能落后纯视觉时序模型超过 0.01。按汇总宏 F1 选型，同分比较 NLL。没有用旧测试录像反选模型。

| 候选 | 五折开发宏 F1 | 开发准确率 | NLL |
|---|---:|---:|---:|
{table}

选中模型比同条件 MLP 基线提高 **3.34 个百分点**；比纯视觉时序模型提高 **0.57 个百分点**。单独增加关节状态的 MLP 没有改善；主要收益来自时间建模，不能把全部提升归功于关节或交叉注意力。缺输入/速度增强候选没有胜出，部署模型没有采用该候选的增强训练权重。

以整段录像重采样 5,000 次，宏 F1 差值的描述性 95% 区间约为 **[{100*ci[0]:.2f}, {100*ci[1]:.2f}] 个百分点**，包含零。样本少、同场景、标签暂定，而且在同一开发集上选择多个候选，结果可能乐观；这不是跨场景或独立测试的可靠提升证明。

温度仅由开发折外预测确定，最终为 {selection['temperature']:.1f}。首轮进程的校准实现把温度施加在集成平均之后，而运行时施加在各成员之后再求平均；已用开发集修正计算顺序，选型与所有训练权重不变。原始报告保存在 `training/before_calibration_correction/`，修正记录在 `training/calibration_correction.json`。没有用历史回归调温度。

## 旧录像回归结果与未解决问题

| 同一批 5 段、70 个共同标注点 | 已安装 v1 | 新 v2 |
|---|---:|---:|
| 阶段正确数 | 63/70 | 65/70 |
| 准确率 | 90.00% | 92.86% |
| 宏 F1 | 89.52% | 90.40% |
| episode_000137 | 14/14 | 12/14 |
| episode_000138 | 13/14 | 14/14 |
| episode_000139 | 13/14 | 14/14 |
| episode_000140 | 9/14 | 11/14 |
| episode_000141 | 14/14 | 14/14 |

这 5 段以前已经检查过，不能再次称为全新盲测。最终 v2 用 10 段开发录像拟合，旧版 v1 用 8 段拟合，二者训练预算也不同；上表表示产品版本回归，架构公平对照应看五折表。上一轮某些未部署的视觉时序候选曾在这 5 段取得更高成绩；本轮没有据此反向选型。

v2 的 5 个错误中，3 个被复核提示覆盖，**另 2 个仍是高把握度误判**：137 的 24.5 秒（搬运→放置，约 89%）和 140 的 3.5 秒（接近→抓取，约 99%）。研究标注本身仍有稀疏边界不确定性。新版并非每段都改善，137 退步；140 的快速放置过程仍然困难。不能将较好的整体指标等同于每次判断可信。

阶段标签中的“释放后观察”只表示动作阶段；空抓、掉落、篮外放置、篮沿挂住的最终成功判别仍未解决。127 的恢复录像和 141 的篮沿终态不被自动写成成功或失败。当前技术质量总分仍由原有质检体系计算，本轮没有把阶段置信度混成质量分。

## 缺输入与复核行为

对历史回归录像人工移除整路输入，使用同一模型和 70 个原本有效标注点，得到以下受控结果。它们是有限压力检查，并非真实传感器故障下的独立验证。

| 受控条件 | 有输出的标注点 | 阶段准确率 | 行为 |
|---|---:|---:|---|
{missing}

界面分别显示头部、右腕和关节状态覆盖率。少一路相机或状态缺失时，可尝试阶段估计，但强制提示复核；全部图像缺失时保留未知。正常输入下，置信度低于 80% 或三个成员有分歧也会提示复核。所有最终入篮结论仍显示“待核验”。

## 项目验证与交付入口

- 原始 18 段实际评分完成，运行时与研究推理的阶段、复核标记及有效性掩码逐段一致。概率/进度最大浮点差 {runtime['max_float_abs_error']:.3g}。
- 18 段原始源文件指纹和 18 份关节缓存 SHA256 不变；正式审核记录与审核日志未改写。
- 102 项 Python 回归检查、8 个前端测试文件通过；真实录像浏览器验收：v2 {len(checks['live']['checks'])} 项、v1 {len(checks['v1-regression']['checks'])} 项、WARP {len(checks['warp-regression']['checks'])} 项全部通过。
- 在应用“进展”面板选择 **G1 香蕉搬运 · 多模态时序模型 v2（研究版）**。旧版和 WARP 仍可选择。
- 模型文件：`robot-data-studio/workspace/warp/models/{profile['checkpoint']}`。
- 模型 SHA256：`{profile['checkpoint_sha256']}`。
- 训练、校准、结果与输入摘要：本目录 `training/`、`states/`、`state_manifest.json`；真实评分索引 `local_stage_report.json`；程序校验 `runtime_verification.json`。
- 可复算图表：[comparison.png](comparison.png)、[comparison.svg](comparison.svg)。完整复现操作见应用内 `STAGE_MODEL_V2.md`。

当前最有价值的下一步是独立核验阶段边界与失败终态，再检验模型分数能否改善实际策略训练的数据选择。已有成绩支持继续研究，尚不足以证明通用数据质量判断或机器人任务成功率提高。
'''
    (ROOT/'升级验证报告.md').write_text(text)
    (ROOT/'交接说明.md').write_text('''# HoloCurate 多模态时序模型 v2 交接

本包在原 WARP 和阶段模型 v1 基础上加入 `g1-stage-v2`。主要说明见 `research/stage_v2_20260917/升级验证报告.md`，使用与复现见 `robot-data-studio/STAGE_MODEL_V2.md`。

保留三个已安装的公开权重模型及对应来源。包中包含源代码、模型、冻结的视觉/状态特征、研究标注和验证结果，不包含原始 MCAP、虚拟环境、登录缓存、浏览器配置或人工审核数据库。若仅复算保存特征上的模型预测，不需要原始录像；完整重解析和提取图像需要原始 MCAP。

模型用于阶段辅助复核。“释放后观察”不表示入篮成功，阶段进度不作为 WARP 速度权重。新模型为同场景研究版，缺少独立新录像与正式人工标注的验收。

在已有依赖环境中运行解包目录中的 `research/stage_v2_20260917/verify_delivery.py`，可核对全部 18 段保存特征上的预测。该脚本固定读取包内相对路径，不回读原工作区的特征或原始录像。完整性清单见包根目录 `ARTIFACT_MANIFEST.json`。
''')
    print('Wrote report and handoff notes')

if __name__=='__main__':main()
