# 论文模型身份核验与软坐标尺度诊断交接说明

核查日期：2026-09-11。命名依据为当前 [Overleaf SpikePose_Draft_V2.6](https://www.overleaf.com/project/6a843f589d98cde3ea0ef8bd) 的方法部分及表 1、4、5。

## 交接任务与执行边界

本文件不只是名称映射。后续执行者应先确认“论文方法—实验配置—代码版本—checkpoint—评估协议—论文数值”是同一条可追溯链，再分析 softmax 软坐标是否造成运动估计尺度失配。

当前交付范围是：在本地建立本文档，提交到 GitHub `OanaHuang/Msc-Project` 的 `main`，由 mitkof 的 `/extra2/yunhao/MSc_Project` 拉取并核验文件一致。同步文档不等于启动训练、修改推理算法或改写论文结果。

### 后续诊断执行顺序

1. **确认工作目录。** 在 mitkof 使用 `/extra2/yunhao/MSc_Project/scripts/thesis_experiment`；先记录 `pwd`、Git HEAD、工作区差异及 Python 环境。不要以仓库根 README 的历史模型家族、NTU-18 规划、NTU-25 preview 或另一个 worktree 代替当前 v2.6 的 16 关节论文实验。
2. **核对模型身份。** 读取下述正式评估的 `evaluation_provenance.json`、训练 resolved config、lineage 和 checkpoint。对 SpikePose 核对本文列出的 SHA-256、epoch=13、seed=42；对基线和其他消融分别读取各自 provenance，不能沿用主模型的 epoch。配置中核对 16 关节布局、64×64 热图、sigma=2、单帧一次 SNN 更新、MAM 配置和初始化来源。
3. **确认代码实现。** 区分当前代码、训练时代码和 2026-09-09 评估代码。评估记录有未提交修改，Git commit 不足以重建全部源码。保存关键文件内容及哈希，重点检查 builder 的空间热图到 MAM 路径、MAM softmax、位移残差、warp、状态延续和 DARK 解码；如不能还原历史代码，明确记录该限制。
4. **固定评估口径。** 当前论文 NTU60 使用 4,563 段完整视频、419,968 帧、MPII-style 16 关节、每段完整视频固定 tube crop；SNN 每帧重置，MAM 跨推理 chunk 保留并在源视频边界重置。不要混入旧的逐窗口 reset、逐窗口 crop、sampled-2x16 测试或 NTU-25 的结果。需要小规模诊断时，固定并保存验证视频名单，保持坐标和 crop 定义一致；小样本结果不冒充完整论文测试。
5. **先做只读推理诊断。** 使用已核实的 checkpoint，记录进入 MAM 前原始热图的值域、峰值和背景分位数、softmax 最大概率和熵、软坐标与 GT 误差、DARK 与 GT 误差、粗位移和最终位移与 GT 位移误差、残差幅值及饱和比例。按真实运动大小分组，只用相邻帧有效关节；统一在热图坐标系比较，记录有效样本数。额外记录零位移预测作为参照。
6. **开展同权重对照时标明诊断性质。** 原实现、推理时关闭对齐、GT 位移 oracle 和温度变体分开输出。GT 仅用于显式 oracle 诊断。温度仅在验证集选择。更换软坐标会改变 offset/gate 输入分布，未经训练适配的变体不能直接作为修复后的正式模型结果。
7. **依据证据决定修复。** 输出“已证实、尚未证实、下一步”三类结论。若真实热图导致显著位移压缩，提出坐标提取修正、训练适配和受影响消融重评方案；不能仅靠改公式宣称问题解决，也不能仅凭合成高斯反例判定全部 PCK 无效。

诊断产物至少包含：运行命令和环境记录、模型/代码/数据清单及哈希、分组指标 CSV、位移对比图、结论 Markdown。写入新的诊断目录，保留现有 checkpoint 和评估产物。

### 已完成与尚未完成

- 已完成：读取当前 Overleaf 方法定义和主表；对齐主模型/基线指标；核对本地主模型 checkpoint 哈希；检查两个本地工程的 softmax/残差代码；复现合成高斯例子。
- 尚未完成：mitkof 当前源码与历史运行源码的一致性核验；真实 checkpoint 原始热图和运动估计诊断；任何修复、训练或正式重评。
- 合成例子：64×64、sigma=2、峰位置 (16,24)→(20,24)，完整高斯粗位移为 0.0320867 像素，3σ 截断高斯为 0.0320372 像素；加入每轴至多 2 像素残差仍不能恢复 4 像素水平位移。这仅证明该输入尺度下的失配。
- 历史第 20 轮日志的采样残差均值约 0.0447 像素、饱和比例为 0。它不代表第 13 轮 best checkpoint 的全验证集统计，也不能作为粗位移正确的证据。

论文明确将 **SpikePose** 定义为完整的单步 SNN 编码器与 MAM 系统，将 **SpikePose w/o MAM** 定义为逐帧 SNN 变体。后续诊断报告、图例和说明采用以下显示名称；实验 ID、checkpoint 路径继续作为追溯标识。

## NTU60 主模型与基线

| 论文显示名称 | 实验 ID | 旧报告/内部别名 |
|---|---|---|
| SpikePose | `mamv2_fullcs20` | Core MAM、MAM、mam |
| SpikePose w/o MAM | `mamv2_fullcs_p00_source` | SpikePose Frame、Frame baseline、p00 |
| SpikePose–ANN w/o MAM | `pilot20_ntu_spikepose_ann` | SpikePose-ANN、spikepose_ann |
| SpikePose–ANN + MAM | `pilot20_ntu_spikepose_ann_mam` | ANN + MAM、spikepose_ann_mam |
| SpikeYOLO–Frame | `pilot20_ntu_spikeyolo` | SpikeYOLO、spikeyolo |
| ResNet-50 | `pilot20_ntu_simplebaseline_r50` | SimpleBaseline ResNet-50、simplebaseline_r50 |
| HRNet-W32 | `pilot20_ntu_hrnet_w32` | hrnet_w32 |

`(Ours)` 是表格归属标注，不是独立模型名。MAM 单独使用时指 motion-aligned heatmap memory 模块；`MAM V2` / `mam_v2` 是实现版本标识。

主模型与逐帧基线的映射已由 `Outputs_Thesis_Pilot20/reset_policy_validation/20260909_table4/` 下的 evaluation provenance 核实。其他表 1 基线的实验映射见 `tools/run_fullvideo_evaluation_queue.py` 的 `MODELS`。

### 主模型身份复核

本地存在主目录 `scripts/thesis_experiment` 和工作树副本 `.worktrees/ntu25-pilot/scripts/thesis_experiment`。论文主模型评估记录指向服务器 `/extra2/yunhao/MSc_Project/Outputs_Thesis_Pilot20/runs/ntu60_cs/mam_v2_fullcs/mamv2_fullcs20/seed_42/checkpoints/best.pt`，epoch 为 13。

本地同相对路径 `best.pt` 的实测 SHA-256 为 `4ea67827c0f7ee837e0f8544a98ade41466f234b465c42479ebba3aaeb908468`，与论文对应评估记录的 `checkpoint_sha256` 完全一致。

`mam/video_reset/summary.json` 的 PCK=83.8055987786%、MPJVE=3.4992037571、MPJAccE=5.4979759527，四舍五入后就是当前论文 83.81 / 3.50 / 5.50。`matched_video_crop/p00/summary.json` 为 83.6309805483% / 4.8685491130 / 8.0213571577，对应论文逐帧基线 83.63 / 4.87 / 8.02。

上述确认了 checkpoint 和评估结果的身份。评估 provenance 同时记录了代码有未提交修改，因此不能仅凭其 git commit 宣称当前本地全部源码与当时服务器源码逐字一致。两个本地副本的 MAM V2 源文件并不完全相同，但两者均直接对原始热图使用未缩放 softmax，并采用有界 tanh 位移残差。

## MAM 消融

表 4 用组件勾选表示变体。下列名称是将其组件定义展开后的诊断显示名称，并非声称表 4 原文逐字使用这些行名。

| 诊断显示名称 | 实验 ID | 实际含义 |
|---|---|---|
| SpikePose w/o motion alignment | `mamv2_p06_noalign` | 关闭历史热图 warping 和位移监督 |
| SpikePose w/o adaptive gate | `mamv2_p07_fixed_decay` | 保留可学习、与当前输入无关的逐关节 retention；并非将 retention 永久固定为 0.08 |
| SpikePose w/o residual fusion | `pilot20_mamv2_noresidual` | 关闭当前热图直接残差融合路径；仍保留位移残差分支 |

前两项正式结果使用 `variants/setups_full/seed_42`，不能仅凭实验 ID 误取默认 S010 开发运行。核查依据是对应 `no_alignment`、`fixed_retention`、`no_residual` 目录的 `evaluation_provenance.json`。

特别区分两个概念：

- **Residual offset**：`coarse + residual` 中用于位移修正的残差，配置为 `memory_use_residual_offset`。
- **Residual fusion**：`H + gamma * (M - H)` 的输出融合路径，配置为 `memory_use_current_direct_path`。

`pilot20_mamv2_noresidual` 的后者为 false，前者为 true，因此不能把该实验称为“去掉残差位移”。

## 诊断结果命名与范围

- 原始模型结果使用 `SpikePose` 和 `SpikePose w/o MAM`。
- 同一 checkpoint 临时关闭对齐时，标为 `SpikePose — alignment disabled at inference (diagnostic)`，与单独训练的 `SpikePose w/o motion alignment` 区分。
- GT 对齐标为 `SpikePose — GT alignment (oracle diagnostic)`。
- 温度变体标为 `SpikePose — softmax temperature τ=… (diagnostic)`。
- 表 5 的 `SP w/o M` 是 `SpikePose w/o MAM` 的已定义缩写；滤波器报告可展开为 `SpikePose w/o MAM + EMA / One Euro / SG`。
- KPA、TPA、KTP 扩展不属于当前表 1 的 SpikePose 主模型，不将其 checkpoint 混入主模型诊断。
- 同一模型在不同 crop、state reset、数据子集和评估版本下的指标必须另标协议；名称一致不代表不同评估版本的指标可混用。

## 当前论文仍存在的名称歧义

方法部分定义 SpikePose 为完整 encoder–MAM 系统，但 MPII 表 2 使用 `SpikePose (T=1)`、`SpikePose (T=2)` 表示空间编码器，邻近正文另用 `SpikePose–SNN`。MPII 表 2 的 `SpikePose–ANN` 同样是空间编码器，不能据名称误认为包含 MAM。

建议后续编辑论文时将 MPII 两行明确为 `SpikePose w/o MAM (T_snn=1)` 和 `SpikePose w/o MAM (T_snn=2)`，ANN 行明确为 `SpikePose–ANN w/o MAM`，并统一相邻正文。这是待处理的论文建议；本次仅建立实验命名对应表，未修改 Overleaf。
