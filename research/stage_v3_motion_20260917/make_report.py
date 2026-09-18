"""Build the Chinese release report directly from the completed experiment evidence."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1];V2=ROOT.parent/'stage_v2_20260917'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    first=read(ROOT.parent/'stage_v3_20260917/training/cross_validation.json')
    second=read(ROOT/'training/cross_validation.json');report=read(ROOT/'training/report.json')
    old=read(V2/'training/report.json')['historical_regression'];guard=read(ROOT/'review_guard/report.json')
    installed=read(ROOT/'installed_model.json');runtime=read(ROOT/'runtime_verification.json')
    assert runtime['passed']
    ui={}
    for folder in ['live','v2-regression','warp-regression']:
        r=read(APP/'workspace/evidence/stages-v3'/folder/'report.json');assert r['passed'];ui[folder]=len(r['checks'])
    candidates=[('v2 参考',first['v2_reference']),('v2 + 软标签',first['v2_rw']),
        ('长短上下文',first['multiscale']),('长短上下文 + 软标签',first['multiscale_rw']),
        ('显式运动变化',second['motion']),('显式运动变化 + 软标签（采用）',second['motion_rw'])]
    table='\n'.join(f"| {n} | {r['overall']['macro_f1']*100:.2f}% | {round(r['overall']['accuracy']*139)}/139 | {r['overall']['nll']:.4f} |" for n,r in candidates)
    controls='\n'.join(f"| {label} | {round(report['controls'][key]['accuracy']*report['controls'][key]['n'])}/{report['controls'][key]['n']} |" for key,label in [
        ('head','去掉头部图像'),('wrist','去掉右腕图像'),('state','去掉关节状态'),('mid_start','从记录中间开始'),('history_shuffle','打乱历史，保持当前帧不变')])
    curves=read(ROOT/'training/regression_curves.json');released=read(ROOT/'review_guard/regression_curves.json')
    stage_names=['接近左桌','抓取','搬运','放置','释放后观察'];errors=[]
    for ep,c in curves.items():
        for i,y in zip(c['annotation_indices'],c['annotation_stages']):
            if c['stage'][i]!=y:
                errors.append(dict(episode=ep,time_s=c['times_s'][i],label=stage_names[y],prediction=stage_names[c['stage'][i]],
                    confidence=c['confidence'][i],review=released[ep]['review'][i]))
    error_table='\n'.join(f"| {e['episode']} | {e['time_s']:.1f} | {e['label']} → {e['prediction']} | {e['confidence']*100:.1f}% | {'是' if e['review'] else '否'} |" for e in errors)
    old_curves=read(V2/'training/regression_curves.json')
    old_unflagged=sum(c['stage'][i]!=y and not c['review'][i] for c in old_curves.values()
                      for i,y in zip(c['annotation_indices'],c['annotation_stages']))
    before=guard['historical_before'];after=guard['historical_after'];final=read(ROOT/'finalization.json')
    out=f'''# HoloCurate v3 升级与验证记录

本轮完成了运动感知阶段网络的训练和项目接入。相同 139 个开发标注点上的五折宏 F1 由 **93.68% 提高到 95.14%**，阶段正确数由 **131 到 133**；历史 5 段录像仍为 **65/70**。这是一项有开发集证据的小幅改进，尚不能确认新录像上的收益。

现有 18 段录像已由实际评分程序重新计算，阶段与复核输出全部通过离线一致性核对。模型在界面中登记为 **{installed['label']}**，ID 为 `g1-stage-v3`。本轮没有获得新录像，也没有新增独立人工验收的标签。

![开发比较和历史回归](comparison.png)

## 实际加入的模型改动

**运动变化输入。** 继承 v2 的双相机与关节融合网络，在每个视觉特征和关节位置旁加入当前与前一采样的差分。头部、右腕特征各由 384 维扩展为“当前 + 变化”768 维；41 维实测位置扩展为 82 维。差分前后两端都有效才使用，缺失输入不会变成虚假的大幅运动。这里的关节变化是 2 Hz 网格上的位置差，未使用原始 `state.dq`，不能称为实测速度。

**阶段过渡软标签。** 保留原有研究标注及同标签区间的硬监督，只在训练录像的已知阶段锚点之间，利用视觉与状态差异计算链图随机游走的吸收概率，形成软标签。未知过渡区不被强行当作某一阶段的确定真值；不向锚点外推，也不跨输入缺口传播。最终拟合有 493 个硬监督网格样本与 149 个弱监督样本，来自 10 段原有开发录像。这些网格样本并不是 642 次独立人工标注。

**不确定提示保护。** 概率校准可能把原本低把握的结果抬高。发布版保留原始温度和校准后任一模型集成触发的复核标志，界面用“阶段证据不足”解释新增提示；阶段、概率、把握度和进度均保持分类器原有输出。该规则增加复核量，不能保证覆盖全部错误。

架构为：冻结的 DINOv2-small 双视角特征 + 实测位置及各自差分 → 分流投影与融合 → 两层、四注意力头、宽度 96 的 Transformer → 五个阶段 token 的交叉注意力 → 阶段分类与阶段内进度。最近 8 个 2 Hz 采样点覆盖 3.5 秒，单成员 353,610 个训练参数，集成三个种子模型。视觉编码器没有微调。保留当前/过去输入约束，不使用整段时长或采集成功标签作为输入，不强制阶段单向推进。

## 论文依据与采用范围

| 来源 | 借鉴与本项目结果 |
|---|---|
| [WARP-RM, arXiv:2606.28320](https://arxiv.org/abs/2606.28320)；[作者固定版本代码](https://github.com/uynitsuj/WARP-RM/blob/26f9894abdaf883af8db78a38e164764e5bf7c00/warp_rm/models/aggregators/transformer.py) | 依据作者实现中的 `use_temporal_diffs` 引入显式视觉差分，并扩展到关节位置。原有时间扭曲相对进展模型仍保留；v3 是任务阶段模型，不等同于原 WARP 奖励模型。 |
| [Random Walks for Temporal Action Segmentation With Timestamp Supervision，WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Hirsch_Random_Walks_for_Temporal_Action_Segmentation_With_Timestamp_Supervision_WACV_2024_paper.html) | 借鉴稀疏标注图上的随机游走及边界不确定性，采用一维链的训练软标签。未复现原文完整图结构、全部损失或推理平滑系统。 |
| [How Much Temporal Long-Term Context is Needed for Action Segmentation?，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Bahrami_How_Much_Temporal_Long-Term_Context_is_Needed_for_Action_Segmentation_ICCV_2023_paper.pdf) | 测试局部密集 + 较长稀疏上下文的因果改编，长分支覆盖 14 秒。结果未达到本轮升级门槛，未采用为发布模型。 |

v2 原有的阶段注意力与阶段进度设计继续保留，来源与实现范围见上一轮报告。本轮不把论文名直接当作效果保证，也不声称完成原论文全部模型或机器人策略训练的复现。DINOv3 授权限制不妨碍吸收方法，实际交付仍使用公开 DINOv2。

## 数据划分与全部候选结果

任务是用户确认的“走到左桌、抓香蕉模型、搬到右桌、放入篮子”。数据为同次采集的 18 段 G1 录像；标签是先前由助手看视频产生的稀疏研究标注，未经独立人工验收。本轮未改标签或源录像。

开发 10 段为 126、128–136；五个验证折为 (126,132)、(128,134)、(129,135)、(130,136)、(131,133)。每次只用其余 8 段训练和计算归一化、图边尺度、进度先验及软标签，整段录像隔离。139 个有效稀疏点用于开发指标。137–141 的 70 个点是已看过的历史回归，124、125、127 保留为压力样例，均未加入本轮分类器训练。

两个实验顺序推进：先做长短上下文和软标签，再因初步开发收益不足做显式运动变化。每次先冻结协议，按开发数据选型。门槛固定为宏 F1 至少超过 v2 **1 个百分点**，且 NLL 不能增加超过 0.03。第一轮最佳提升只有 0.93 个百分点，未通过；第二轮最佳提升 1.45 个百分点，通过门槛。开发集反复用于研究，存在选择乐观偏差。

| 候选（选型时温度 T=1） | 五折汇总宏 F1 | 正确阶段点 | NLL，越低越好 |
|---|---:|---:|---:|
{table}

v2 参考直接复用已校验 SHA256 的同划分折权重。新候选均为 300 次优化、硬样本 batch 64、相同三个种子、AdamW、学习率 0.0003；取第 200/250/300 步参数平均。软标签候选每步额外加入 32 个弱样本、损失权重 0.3，因此它们使用更多样本和计算，不能将改善全部归因于网络结构。新模型在前两个验证折改善，另外三个折的分类指标与 v2 相同。

最终温度 0.6 仅用开发折外预测选择，对每个成员先校准再平均。正式拟合只用 10 段开发录像。上述 F1 是用于选型的未校准指标；发布概率和以下历史指标使用最终温度。

## 历史录像与复核成本

| 版本 | 阶段正确数 | 宏 F1 | 提示复核的点 | 错误但未提示复核 |
|---|---:|---:|---:|---:|
| 已发布 v2 | 65/70 | 90.40% | 9/70 | {old_unflagged} |
| v3 分类器，加入保护前 | 65/70 | 90.40% | {before['reviewed']}/70 | {before['unflagged_mistakes']} |
| v3 发布版 | 65/70 | 90.40% | {after['reviewed']}/70 | {after['unflagged_mistakes']} |

历史 NLL 从 {old['nll']:.4f} 到 {report['historical_regression']['nll']:.4f}，但 ECE 从 {old['ece']:.4f} 到 {report['historical_regression']['ece']:.4f}，校准误差更大。发布版的复核量高于 v2，而未提示错误数与 v2 相同。因此，不能宣称历史阶段识别或全面置信度可靠性已超过 v2。

复核保护是在分类器选定、发现校准与复核冲突后追加的独立记录实验，**不是新的盲测**。固定规则先通过开发折外检查：未提示错误 3 → 0，复核点 11 → 23/139；增加 8.63 个百分点，满足预先记录的“漏提示减少且复核比例增加不超过 15 个百分点”门槛。然后才运行历史检查，得到上表 3 → 2。只要原始输出与校准输出之一要求复核就保留提示，没有搜索新的置信度阈值。

曾测试“阶段切换及其后一个采样也提示复核”：在 v3 开发中复核点 11 → 25，未提示错误仍为 3；v2 备选检查也未减少漏提示。因此该规则没有启用。所有失败实验与选择记录均保留。

发布版历史错误点如下，时间为模型 2 Hz 采样位置：

| 录像编号 | 秒 | 研究标签 → 预测 | 把握度 | 提示复核 |
|---|---:|---|---:|---|
{error_table}

这些错误集中在阶段过渡处，稀疏标注边界本身也存在不确定性。137 的 5.0 秒和 140 的 3.5 秒仍有高把握度的提前抓取判断，应优先由人核对。需要更密集且独立验收的边界标注，才能进一步区分模型误差与标注误差。

## 条件扰动检查

下表是分类器选定后的历史诊断，不用于重新选择模型；复核保护不改变其阶段预测。

| 改动 | 阶段正确点 |
|---|---:|
{controls}

缺少任一模态的受测点全部提示复核；双相机均无效时不输出阶段。去掉头部偶然多判对一个点，说明小样本存在波动和视角依赖，不能证明缺头部更好。从中间开始的指标只覆盖后半段 40 个点；打乱历史同时保持当前视觉和状态不变，仅是一次敏感性诊断。124、125、127 的阶段曲线随包保留，但压力样例中的篮外放置、重新抓取和人工复位仍需视频复核，尚未形成可靠的失败分类器。

## 接入、验证与复现证据

v3 的视觉特征约定、关节顺序、单位和缺失处理受检查；模型、源录像及状态缓存指纹参与评分身份。新增复核原因与曲线写入版本化评分结果。旧 v2、v1、WARP 保持原模型权重与使用入口；本轮不改技术质量总分、不自动写入最终成功、不自动生成动作权重。

工程测试为 109 项 Python 测试通过，8 个 Node 测试文件通过。真实视频浏览器检查：v3 {ui['live']} 项、v2 {ui['v2-regression']} 项、WARP {ui['warp-regression']} 项。18 段实际 worker 输出与保存输入的独立复算中，阶段和复核标志完全一致，概率/进度最大绝对差 {runtime['max_float_abs_error']:.3g}；源文件指纹、18 份状态缓存 SHA256 及人工审核数据库保持不变。

已学习参数与原选定分类器逐张量完全一致；发布检查点只增加已通过检查的复核元数据。原分类器 SHA256 为 `{final['classifier_checkpoint_sha256']}`，发布 SHA256 为 `{final['release_checkpoint_sha256']}`。

使用与复训步骤见 [STAGE_MODEL_V3.md](../../robot-data-studio/STAGE_MODEL_V3.md)。`verify_delivery.py` 只读包内视觉和状态数组，可在没有原始 MCAP 的情况下核对全部 18 段评分；解包后的实际复算结果另附于交付包的 `.offline-check.json`。完整重新提取图像仍需原始录像。

关键证据：`protocol.json` / `freeze.json`、`training/cross_validation.json`、`training/selection.json`、`training/report.json`、`review_guard/`、`finalization.json`、`runtime_verification.json`。首轮未采用的完整实验在 `../stage_v3_20260917/`。全部代码、权重、结果和图表来源随交接包保存。

下一步的主要验证缺口是正式阶段边界标注、独立同任务新录像和空抓/掉落/篮外放置等明确失败样例。在这些条件补齐前，v3 适合辅助定位待复核片段；当前证据不足以承诺跨场景、最终成功判别或机器人策略训练收益。
'''
    (ROOT/'升级验证报告.md').write_text(out)
    (ROOT/'release_evidence.json').write_text(json.dumps(dict(model_id=installed['id'],checkpoint_sha256=installed['checkpoint_sha256'],
        report_inputs={p:sha(ROOT/p) for p in ['training/cross_validation.json','training/report.json','review_guard/report.json','runtime_verification.json','finalization.json']},
        historical_errors=errors,browser_checks=ui),ensure_ascii=False,indent=2)+'\n')
    print('Report generated from completed evidence')
if __name__=='__main__':main()
