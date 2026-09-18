> 历史开发记录。当前 GitHub 安装步骤见 [模型说明](docs/MODELS.md)，复算输入及代码入口见 [训练说明](docs/TRAINING.md)。历史交接包中的私有数据与实验输出不随 GitHub 源码发布。

# RoboCurate × WARP-RM 工程接入记录

更新日期：2026-09-17。目标是复现 WARP-RM 的任务进展方法，接入现有审核与训练导出流程。

**已完成可运行的方法融合版本：公开 DINOv2 特征、WARP 时间扭曲训练、相对进度预测、审核候选和动作块权重导出。** 30 段真实公开视频的小规模训练与留出测试已完成，4 段本地 G1 视频已生成真实模型评分。官方 DINOv3 授权只限制原配置的严格复现；当前结果不能用于声称论文全部实验或 G1 策略提升已复现。

原配置的模型访问申请被作者拒绝，已停止自动轮询。严格 DINOv3 复现保留为独立入口；当前方法适配使用官方公开、Apache-2.0 的 DINOv2-S/14，并重新训练进度模型。登录凭据不写入项目或交接包。

## 已接入的功能

1. 审核页新增“进展”面板与同步时间轴：查看进展速度、定位视频，按正向/停滞/回退筛选候选。默认只显示持续至少 1 秒的候选，也可选择 0.4 秒或 2 秒，减少短时抖动带来的审核干扰。候选只填入片段编辑范围，需审核人员加入并保存。
2. 离线评分任务：隔离 ML 环境、后台队列、取消、失败反馈、重启恢复、源文件和模型变更后的缓存失效。多个进程通过工作区文件锁协调任务管理。
3. 原始双目图像先明确选择左目/右目，再做与模型训练一致的裁剪、缩放和归一化。特征缓存记录时间、原始帧时间、有效性与处理参数。
4. 导出窗口可附加 WARP 动作块权重，预检显示评分覆盖、完整动作块和保留数量。原有技术有效性掩码、人工审核条件与数据划分仍独立保存。
5. 完整训练入口：冻结特征上的时间扭曲采样、相对标签、C51 双热点交叉熵、验证、检查点与训练来源。支持将本地训练模型登记到审核页。
6. 两条模型路线：原配置的 DINOv3 推理对照，以及公开 DINOv2 特征上的重新训练。骨干、修订、散列和预处理必须与进度模型的训练来源一致；显式迁移试算另行标注。

生产工作区没有导入测试合成曲线。浏览器验收图中会明确显示“测试曲线（非模型输出）”，这些数据只存在于隔离测试工作区。

## 论文和代码基线

- 论文：[arXiv 2606.28320 v4](https://arxiv.org/abs/2606.28320v4)。本次依据 2026-09-06 修订版。
- 作者代码：[uynitsuj/WARP-RM](https://github.com/uynitsuj/WARP-RM/tree/26f9894abdaf883af8db78a38e164764e5bf7c00)，固定提交 `26f9894abdaf883af8db78a38e164764e5bf7c00`。
- 仿真奖励模型：[warp-rm-sim-bottles-sss15](https://huggingface.co/uynitsuj/warp-rm-sim-bottles-sss15)，修订 `6417be056ed0bc2b2549f7f414fed76d4ca8501f`。
- 奖励模型 SHA256：`9c74aa3934b12dd6b169f8945dc2501a8c625ef0becb9f27e4f297725e2c775f`。已下载、校验，并严格加载其参数执行前向测试。
- 图像骨干：[DINOv3 ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)，冻结、768 维池化特征。下载后会固定实际修订与文件 SHA256。
- 数据集：[sim-bottles-mjwarp-v1](https://huggingface.co/datasets/uynitsuj/sim-bottles-mjwarp-v1)，固定修订 `897b86679272a47897cc5ecca78c2205586a29d5`。
- 选取公开视频文件 `videos/top_camera-images-rgb/chunk-000/file-00007.mp4`，433,905,625 字节，SHA256 `14ceccf471beb2b97ab4cd04dc6b5a0d986659afa17bc374323c7ad8d963fe4d`。已完整校验。首个验证对象为逻辑 episode 2316，共 885 帧。

`vendor/warp_rm` 保留作者的 MIT 许可及未修改实现，逐文件来源见 `vendor/warp_rm/provenance.json`。奖励模型与数据的许可分别遵循其上游发布条款；DINOv3 使用 Meta 的模型条款，交接包不分发该权重。

## 方法实现与差异

| 环节 | 实现约定 |
|---|---|
| 图像 | 224×224，ImageNet 归一化；公开视频使用完整顶视图和 `squash` |
| 窗口 | N=32，原始帧率 30 Hz，公开仿真参考步长 SSS=15，特征步长 1 |
| 标准跨度 | 15.5 秒；短记录依作者实现减小间隔并校准速度 |
| 采样 | AR(1) 对数速度，α=0.5，σ=ln(2)，Poisson 反转次数均值 1，整体反转概率 0.5 |
| 路径预算 | 固定中心 1.5 秒、半宽 1.0 秒；不随 SSS 自动缩小 |
| 标签 | `(index[j]-index[0])*feature_stride / ((N-1)*SSS)` |
| 时序模型 | 作者 Transformer，12 层、8 头、768 维；拼接图像特征及相邻差分 |
| 损失 | 30 个分类支持点覆盖 [-3,3]，邻接双热点交叉熵；不启用未训练的绝对进展头 |
| 速度 | 对窗口相对预测求差，乘 N−1，按覆盖区间进行重叠平均 |
| 训练优化 | AdamW，weight decay=0.001，梯度裁剪 1；继承作者训练器的更新后 warmup 与余弦调度约定 |

部署中显式保留没有评分的区间，避免将缺帧解释为停滞。作者参考代码会插值未覆盖位置并对积分结果做归一化；审核界面不显示这种归一化曲线为“完成百分比”。真实公开视频对照脚本只在双方实际覆盖的位置比较速度，另外保存作者参考输出供审计。

采样随机数改为显式传入 NumPy Generator，使重复实验可控。已对 100 个随机种子、4 种序列长度、3 种特征步长逐项对照作者 ARSampler，采样索引完全一致。

公开复现指南与旧 recipe 文档有默认值差异，本实现固定上述路径预算。原公开代码目前默认 batch=1024、lr=4e-4、steps=15000；本训练入口的参考配置显式为 batch=256、lr=1e-4、steps=20000，所有实际参数写入 run.json。不能将不同训练预算的结果视为同一次实验。

公开元数据已核对：2438 段、2,228,979 帧，按 30 Hz 为约 20.64 小时。按作者“物体数量分层、保留最短 25%、从最短 15% 池中抽取 20 段验证”的流程，得到 590 段训练、20 段验证；清单见 `research/warp_rm_20260916/public_selection.json`。这只是数据选择清单，目前没有对这 610 段执行完整训练。论文表格的 17.5 小时口径与公开元数据仍需进一步解释。

## 本机使用

当前应用：`http://127.0.0.1:8421`。打开记录后选择“进展”，模型就绪后点击“计算进展”。官方模型在 G1 头部图像上标为“迁移试算，尚未验证任务判别效果”。

所有命令在 `robot-data-studio` 目录执行。当前独立环境已安装完毕；重新部署时：

```bash
python3 -m venv workspace/warp-env
workspace/warp-env/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
workspace/warp-env/bin/python -m pip install -r requirements-warp.txt
```

GPU 机器需要安装与其驱动匹配的 PyTorch CUDA 构建。本次机器没有可用 NVIDIA GPU，未使用付费外部算力。

本机终端代理曾返回 `Unknown scheme socks://...`。`bash scripts/hf_login.sh` 会仅对本次命令使用已经确认可用的 `http://127.0.0.1:7897`。其他机器可用 `HOLOCURATE_HF_PROXY` 指定自己的 HTTP 代理；没有该代理的机器直接运行 `workspace/warp-env/bin/hf auth login`。

### 当前推荐：公开模型的方法适配

不需要 DINOv3 访问权限，也不需要为此重新登录。依次准备公开模型、固定的数据切片，运行小规模实验：

```bash
workspace/warp-env/bin/python scripts/warp_prepare_open.py
workspace/warp-env/bin/python scripts/warp_fetch_public.py
workspace/warp-env/bin/python scripts/warp_open_pilot.py
```

下载命令仅在当前进程处理代理配置；若系统只有不兼容的 `socks://` 配置，可在命令前加 `HOLOCURATE_HF_PROXY=http://127.0.0.1:7897`，端口按实际代理修改。模型和数据来自无需账号的公开库，SDK 使用 `token=False`。

该实验使用 DINOv2-S/14 的 384 维冻结特征，输入 224×224；时间模型为 4 层、128 维、4 头，batch=32、lr=3e-4、1600 步，feature_stride=3、SSS=15。保留论文的 AR 时间扭曲、32 帧相对标签、双热点交叉熵和速度融合。上述骨干、模型规模、采样密度与训练预算是明确的方法适配差异。

本次实际训练 1600 步，按验证交叉熵选择第 1400 步检查点，SHA256 为 `a17751a7a566da2a60f976452c2d45a2ceb37142ab8c9e3752ee3904ef4658f4`。测试集结果：

| 测试输入 | 相对标签 RMSE（越低越好） | 正反方向平衡准确率 |
|---|---:|---:|
| 真实视频特征 | 0.3243 | 71.09% |
| 打乱帧顺序 | 0.7940 | 52.04% |
| 固定为单帧 | 0.7663 | 57.84% |
| 恒零进度基线 | 0.7615 | 不适用 |

相对恒零基线，RMSE 降低 57.41%。这支持模型利用了真实画面变化，但样本只有 5 个测试 episode；320 条时间扭曲共享这些源视频，不能当作 320 个独立机器人试验。模型规模和数据覆盖均不足以支撑通用 G1 任务质量结论。

数据来自公开第 7 个视频文件，在原论文训练候选池内固定随机抽取 30 段，按完整记录分成 20 段训练、5 段验证、5 段测试。原论文验证记录未参与此实验训练。验证集选择检查点，测试集随后评估 320 条时间扭曲序列，并比较真实特征、打乱帧顺序、固定单帧和恒零进度基线。报告衡量时间扭曲预测，不等同于真实任务完成率。

实验计划、原始视频切片来源、特征 SHA256、训练记录、测试控制和真实曲线保存在 `research/warp_rm_20260916/open_pilot`；`status.json` 表示运行状态，`report.json` 只在训练和测试完成后生成。中断时已完成的特征可复用。训练目录已有未完成运行时需保留它并使用新的实验输出目录。

将训练模型显式登记为 G1 迁移试算：

```bash
workspace/warp-env/bin/python scripts/warp_register_model.py \
  --checkpoint research/warp_rm_20260916/open_pilot/train/best.pt \
  --id open-dinov2-pilot --label 'WARP 方法适配 · DINOv2 / 公开小样本' \
  --transfer-camera head --transfer-view left
workspace/warp-env/bin/python scripts/warp_score_local.py \
  --model-id open-dinov2-pilot --episodes episode_000124 episode_000125
```

迁移试算会保留训练相机与目标相机的差异，默认状态为 `transfer_unvalidated`。它使真实视频评分流程可运行，但公开瓶子任务的模型仍需在本地任务上标定。不能把换用视觉模型与数据域迁移混为已经验证的性能提升。

本机已评分 episode 000124、000125、000133、000141，覆盖分别为 99.71%、99.35%、99.46%、99.34%。其中 000125 的操作员结局标记为失败；四段平均预测速度均较低（约 0.017–0.074×），目前不能据此可靠区分本地任务成功或失败。当前价值是完整可运行的学习、评分、复核与加权导出流程；本地任务适配仍需明确任务及成功示范。真实评分报告见 `research/warp_rm_20260916/open_pilot/local_g1_report.json`。

### 原配置复现入口

模型获批后手动续跑：

```bash
workspace/warp-env/bin/python scripts/warp_resume.py
```

限定等待窗口后自动续跑：

```bash
workspace/warp-env/bin/python scripts/warp_resume.py --wait-minutes 120
```

运行状态写入 `workspace/warp/workflow/status.json`。流程为：下载并校验模型 → 公开第 2316 段推理与作者代码对照 → 本地 episode 000124、000125、000133、000141 评分。评分不修改原始文件或人工审核。两小时后若仍未获批，流程停止等待，状态保留为 awaiting_access，之后可再次执行。若模型作者明确拒绝申请，则立即停止并记录 access_rejected；本次属于这一情况。

真实公开对照结果生成于 `research/warp_rm_20260916/public_inference/report.json` 和 `predictions.npz`。在文件实际生成且通过检查前，不应宣称真实视频复现已完成。

## 本地任务训练

本批 18 段录制来自同一会话，原始任务描述泛化，未保存人工审核。训练输入需先在现有审核页确认成功示范、填写一致的具体任务指令，并明确完整任务边界。当前入口不将带裁剪的记录自动视为完整成功示范。

先只检查训练条件，示例中的任务指令与记录划分必须按实际审核修改：

```bash
workspace/warp-env/bin/python scripts/warp_extract_local.py \
  --task '填写与审核页完全一致的实际任务指令' \
  --train episode_000124 episode_000126 \
  --validation episode_000133 \
  --output workspace/warp/training/task-a/manifest.json --plan-only
```

计划文件会逐条列出缺少审核、任务不一致、来源过期、失败结局或记录缺失等原因。满足条件后去掉 `--plan-only` 提取真实特征。训练与验证按完整 MCAP 内容划分，重复源内容或重复特征跨集合会被拒绝。同会话验证仍不等于独立会话泛化。

完整架构、CPU 小规模训练示例：

```bash
workspace/warp-env/bin/python -m warp_progress.train \
  --manifest workspace/warp/training/task-a/manifest.json \
  --output workspace/warp/training/task-a/run-001 \
  --device cpu --batch-size 8 --steps 200 --warmup-steps 20 --eval-every 40
```

这 200 步只是本地试验。完整参考预算显式设为 `--device cuda --batch-size 256 --lr 1e-4 --steps 20000 --warmup-steps 1000`，并使用相应完整数据清单和算力。训练输出 latest.pt、best.pt 与 run.json，记录数据散列、划分、参数和验证指标。验证指标衡量合成时间扭曲的预测能力，不是机器人任务成功率。

登记真实本地训练模型：

```bash
workspace/warp-env/bin/python scripts/warp_register_model.py \
  --checkpoint workspace/warp/training/task-a/run-001/best.pt \
  --id task-a-v1 --label '任务 A · 本地模型'
```

登记时检查受支持的 DINO 骨干、相机、裁剪和时间尺度的训练来源，并默认标记效果尚未验证。之后在审核页刷新模型列表即可使用。

已有本地审核数据后，可以直接用公开骨干提取特征并从头训练，无需先有进度模型：

```bash
workspace/warp-env/bin/python scripts/warp_extract_local.py \
  --backbone facebook/dinov2-small --feature-stride 3 --source-standard-stride 45 \
  --task '填写实际审核任务指令' --train episode_000124 episode_000126 \
  --validation episode_000133 --output workspace/warp/training/local-open/manifest.json
```

训练命令必须匹配 `--feature-stride 3 --source-standard-stride 45`。审核条件与数据划分约束保持一致，不自动把操作员成功标记当作已完成的人工审核。

## 导出与 BC 对接

训练包导出窗口展开“任务进展权重”，选择已评分的模型、速度阈值、动作块帧数和模式。当前实现两种权重：

- continuous：`v[t+H-1] × 1(v[t+H-1] > threshold)`。
- binary：`1(v[t+H-1] > threshold)`，对应仿真筛选后的二值权重。

每个动作块要求 H 帧都有效；不跨片段边界，不对片段末端补齐。速度阈值是严格大于。H 是导出帧率下的帧数，导出元信息另存 H/fps 的秒数。

原始 `valid` 保留技术有效性；新增 `warp.velocity`、`warp.valid`、`warp.eligible`、`warp.weight`。模型 SHA256、处理配置、分数签名、覆盖及保留数量随 manifest 保存。没有当前有效评分或没有保留动作块时，导出预检会提示。

```python
from dataset_reader import RobotDataset
from warp_progress.bc import weighted_bc_loss

dataset = RobotDataset('解压后的训练包', split='train',
                       sequence_length=30, curation='warp')
# DataLoader 组批后，策略用起始观测预测 H 帧动作。
# prediction = policy(batch['observation.state'][:, 0], ...)
# loss = weighted_bc_loss(prediction, batch['action'], batch['sample_weight'])
```

`sequence_length` 必须与导出时的 horizon 一致。示例只展示损失接线，项目没有凭空增加一个经过任务验证的 G1 策略。

论文仿真对比的 31.5% 保留率包含尾部补齐的动作块；本地保守导出使用不同候选定义。公开视频复现脚本同时保存两套权重作比较，不能直接用本地导出的保留率声称复现了论文策略实验。评分模型本身使用整段录制的双向上下文；动作块不跨片段并不意味着它适用于在线实时奖励。

## 验证与边界

作者已发布轨迹的离线复算已完成：128 组配对场景的官方 `--self-test` 共 23 项全部通过；512 组场景也与发布表格吻合。这里运行的是原始 `score_world` 和汇总函数，按场景并行调度，没有重新训练或执行 BC 策略。

| 公开轨迹 | 原始 BC 每场数量 | WARP-BC 每场数量 | 原始 BC 吞吐量 | WARP-BC 吞吐量 |
|---|---:|---:|---:|---:|
| 128 对 | 3.9766 | 4.6719 | 243.02 /h | 301.28 /h |
| 512 对 | 3.8848 | 4.5332 | 236.59 /h | 290.40 /h |

512 对中 2 对（4 个文件）含非有限状态，数据散列与作者发布文件一致。主结果保留原始评分口径；另行排除这 2 对时，510 对的平均数量为 3.9000→4.5510，差值仍为正。异常形态与仿真发散一致，但未取得更高精度的源状态，不能确定具体原因。原评分器按 30 Hz / 60 秒计时，而发布控制步长为 0.034 秒；复算保持发布口径，并明确记录这一差异。原代码的 p 值采用正态近似，报告另存 SciPy 的配对 t 检验值。

轨迹修订、文件树 SHA256、逐场得分、数据质量检查、敏感性分析及原评分器输出见 `research/warp_rm_20260916/trace_replay`。这些公开数字不代表 RoboCurate 或 G1 已获得同等提升。

- Python：92 项通过，覆盖原有回归与新增采样、融合、权重、来源失效、任务生命周期、骨干兼容和真实 MCAP 导出往返测试。
- JavaScript：7 个测试文件，包括空值断线、片段候选和原有播放器/审核逻辑。
- 浏览器：新增进展面板与权重预检 16 项；原有回放、轨迹、审核和布局 38 项；真实已评分 G1 页面另有 10 项，包含原生相机帧到达所选时刻。
- 页面稳定性：运动数据异步加载完成时，不再重复创建已挂载的进展面板，避免打断候选筛选控件。
- 训练优化：明确标注的合成特征测试，8 维特征、2 层 32 维小模型、80 步；验证交叉熵 3.5828→0.8454。此数字仅证明训练代码能够优化，不能作为真实视频效果。
- 原应用进程不导入 torch；模型依赖与耗时任务在独立环境运行。

详细证据位于 `workspace/evidence/warp`。重跑测试时需关闭本机 ROS 自动加载的无关 pytest 插件：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 workspace/warp-env/bin/python -m pytest tests -q
node --test tests/*.test.mjs
PYTHONPATH=. .venv/bin/python tests/check_progress_ui.py
PYTHONPATH=. .venv/bin/python tests/check_trajectory_ui.py
PYTHONPATH=. .venv/bin/python tests/check_progress_live.py --episode a06a7cb4e03d --model-id open-dinov2-pilot
```

尚未完成：DINOv3 原配置的真实视频推理对照、针对 G1 的任务训练与人工标定、完整论文策略训练和新执行的机器人/仿真吞吐量对照。当前 DINOv2 路线已完成真实 G1 评分。正式审核记录目前仍为 0；下一阶段需先明确本地任务指令、成功示范和独立验证记录，再训练本地任务模型并测量候选命中率、审核时间与策略效果。
