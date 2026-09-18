"""Generate the negative-result report directly from frozen experiment evidence."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parent;FOLLOW=ROOT.parent/'stage_v4_decoupled_20260917';APP=ROOT.parents[1]
def read(p):return json.loads(Path(p).read_text())
def main():
    first=read(ROOT/'training/cross_validation.json');second=read(FOLLOW/'training/cross_validation.json')
    decision=read(ROOT/'promotion.json');assert not decision['promoted']
    runtime=read(ROOT/'runtime_verification.json');replay=read(ROOT/'reproduction_check.json');assert runtime['passed'] and replay['passed']
    evidence=APP/'workspace/evidence/stages-v4'
    suites=ET.parse(evidence/'regression.xml').getroot().findall('testsuite')
    tests={k:sum(int(s.attrib[k]) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert tests['failures']==tests['errors']==tests['skipped']==0
    ui={n:len(read(evidence/n/'report.json')['checks']) for n in ['v3-regression','warp-regression']}
    assert all(read(evidence/n/'report.json')['passed'] for n in ui)
    candidates=[('v3 原参考',first['v3_reference']),('跨录像对比学习',first['contrastive']),('区间边界 + 片段约束',first['boundary_structure']),
        ('对比学习 + 边界及片段约束',first['combined']),('阶段/进度梯度隔离',second['detached_progress'])]
    table=[];budget=[]
    for label,row in candidates:
        s=row['overall'];d=s['bracket_diagnostic']
        table.append(f"| {label} | {s['macro_f1']*100:.2f}% | {round(s['accuracy']*s['n'])}/{s['n']} | {s['nll']:.5f} | {d['matched']}/{d['brackets']} | {d['extra_switches']} |")
        runs=[r for fold in row['folds'] for r in fold['runs']]
        if runs:budget.append(f"| {label} | {runs[0]['parameters']:,} | {len(runs)} | {sum(r['processed_examples'] for r in runs):,} |")
    reg=read(ROOT/'training/report.json')['historical_regression'];proxy=read(FOLLOW/'training/progress_proxy_audit.json')['scores']
    text=f'''# 论文扩展实验与项目验证记录

**本轮没有找到超过现行 v3 的候选，项目继续使用 v3。** 新方法的实现、训练权重、对照结果和推理接口已保留在项目中，未登记任何 v4 候选供日常评分使用。这次是完成了一轮研究验证，不能称为已取得模型识别效果提升。

## 研究了什么，实际做了什么

在原来的双视角 DINOv2、关节状态、显式差分和时序网络上，完成四组新方案的真实训练：

| 依据 | 本项目实际适配 |
| --- | --- |
| [TSCL，IEEE RA-L 2024](https://ieeexplore.ieee.org/document/10598312/) | 同阶段、不同训练录像的表征进行对比学习；增加训练用投影头。未复现原论文完整边界估计程序。本轮取得的是官方摘要及出版信息。 |
| [Boundary Supervision + Segment-Level Regularization，CVPR 2026 Workshop](https://arxiv.org/html/2604.01859v1) | 增加边界输出和片段内部的 CDF 约束。把原文的精确边界监督改成稀疏锚点间“至少一次变化”的区间监督；单独测试以及与对比学习组合测试。 |
| [SARM2，2026 年预印本](https://arxiv.org/html/2606.10305v1) | 借鉴阶段与价值分开优化，将本项目近似进度回归的梯度与阶段编码器隔离。没有实现原文的多任务专家路由或机器人闭环策略训练。 |

另外检索了动作分段、对象关系奖励、姿态辅助和目标对比奖励等工作。完整来源、出版类型与未采用原因见 [论文研究与落地决策](论文研究与落地决策.md)。这些是论文思路的任务适配，不是下载四个论文原版模型后直接比较。

## 完整结果

18 段原录像保持不变。10 段开发录像采用原有五折划分，每个候选每折训练三个种子网络。统计对象是同一组 139 个有效稀疏阶段标注点；F1 是分类宏 F1，不是任务成功率，也不是论文中的分段 F1@IoU。下表都使用选型时的温度 T=1。

| 方案 | 阶段宏 F1 | 阶段正确点 | NLL（低更好） | 落入锚点区间的切换 | 额外切换 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

原 v3 的 95.14% 是**开发交叉验证宏 F1**，不能解释为对所有新录像有 95.14% 准确率。边界列只检查预测的阶段变化能否落入两个不同阶段锚点之间；锚点之间可能有未标注动作，因此它只是稀疏标签诊断，不能当作精确边界真值。

升级门槛在各自训练前冻结：F1 不下降、NLL 至多增加 0.02，且至少满足 F1 增加 0.5 个百分点、NLL 降低 0.01，或正确区间切换多两个且额外切换不增加。四组都没有通过。单独对比学习的 NLL 只降低约 0.00035，远小于门槛，分类与切换结果没有改善。

SARM2 启发的后续试验在看到首组实验的部分开发结果后提出，使用单独的冻结协议；它仍使用同一开发集，不是独立验证。其进度代理 MAE 从 {proxy['v3_reference']['mae']:.5f} 到 {proxy['detached_progress']['mae']:.5f}，但阶段分类退步，MSE 也从 {proxy['v3_reference']['mse']:.5f} 到 {proxy['detached_progress']['mse']:.5f}。这个进度标签本身带有时间插值近似，不能据此宣称真实任务进度估计提高。

开发结果相对较好的对比学习候选还进行了预定的历史回归：阶段正确 **{round(reg['accuracy']*reg['n'])}/{reg['n']}**，现行 v3 为 **65/70**。历史结果没有用于重新调参或重选；该候选在查看历史结果前就已未通过开发升级门槛。另三组未进行新的全开发最终拟合，不能给它们虚构历史测试数值。

## 训练与工程交付

四组新方案共完成 60 次折内网络训练，另外完成 3 次对比学习候选的全开发训练。原 v3 的 15 个折内参考网络直接复用旧权重。每次都是 300 次优化；数据划分、种子、优化器与视觉特征保持一致。

| 新方案 | 每成员训练参数 | 折内训练次数 | 折内累计输入窗口次数 |
| --- | ---: | ---: | ---: |
{chr(10).join(budget)}

边界与片段方案额外读取监督区间，计算量较大；这些窗口次数包含重复采样，不是新增录像或独立标签。投影头与边界头只增加了实验容量，未证明有实际收益。

新增内容包括 `warp_progress/stages_v4.py`、四种可复现训练配置、区间切换诊断、梯度隔离、冻结门槛核对和版本化推理支持。v3 仅增加可选的表征返回接口，默认计算保持兼容。由于没有候选通过门槛，模型列表继续保留现行版本，阶段、进度和复核提示不变。阶段模型仍不自动写入最终成功，不参与 WARP 动作权重导出。

完成 {tests['tests']} 项 Python 测试、8 个 Node 测试文件；浏览器验证 v3 {ui['v3-regression']} 项、WARP {ui['warp-regression']} 项。全部 18 段现行评分的阶段及复核标志与独立复算完全一致，浮点最大绝对差 {runtime['max_float_abs_error']:.3g}。原始录像指纹、18 份原生状态缓存和人工审核记录未改动。

另从保存的视觉与状态数组重新加载了 25 个折检查点、共 75 个成员网络，复算基线和四组候选的全部折外输出，与实验保存值最大绝对差为 {replay['max_float_abs_error']:.1g}。这些检查不读取原始 MCAP 或原生缓存。

## 可以得出的判断

当前证据支持保留 v3。四组失效不等于这些论文普遍无效，也不能证明本项目已到性能上限；它说明这些具体适配与超参数在当前样本上没有显示收益。数据来自同次采集，标注未经独立人工验收，开发集已反复用于模型迭代，不能当成新场景盲测。

下一步优先假设是补强已有录像中香蕉、手与篮子的关系证据，并独立核准阶段过渡标签。这个方向尚未完成实验，不能预先承诺提升。继续增加损失项或扩大网络，目前没有获得这批数据的支持。

使用说明见 [STAGE_MODEL_V4.md](../../robot-data-studio/STAGE_MODEL_V4.md)。关键文件为 `promotion.json`、两组 `protocol.json` / `freeze.json`、`training/cross_validation.json`、`training/selection.json`、`reproduction_check.json` 和 `runtime_verification.json`。交接包包含本轮失败候选权重，供追溯研究使用；现行模型仍为 `g1-stage-v3`，SHA256 `{decision['checkpoint_sha256']}`。
'''
    (ROOT/'实验与验证报告.md').write_text(text)
    (ROOT/'release_evidence.json').write_text(json.dumps(dict(promoted=False,active_model=decision['active_model'],
        checkpoint_sha256=decision['checkpoint_sha256'],python_tests=tests,node_test_files=8,browser_checks=ui,
        new_cv_member_fits=60,new_final_member_fits=3,experimental_replay=replay),ensure_ascii=False,indent=2)+'\n')
    print('Generated source-backed negative-result report')
if __name__=='__main__':main()
