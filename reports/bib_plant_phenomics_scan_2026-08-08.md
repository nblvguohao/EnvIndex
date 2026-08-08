# BIB / Plant Phenomics 近一年新发表调研（2026-08-08）

**目的**：响应"走方法类研究型，走 Plant Phenomics 或 BIB 期刊看起，看看近一年有没有我们这个方向的新发表论文，有没有我们可以做的"。

## 1. 两本刊的定量对比

| 期刊 | 中科院分区 | IF（2025/2026） | 出版方 |
|------|-----------|------------------|--------|
| **Plant Phenomics** | **农林科学大类一区** | 8.2 | Elsevier（2025年1月起脱离 AAAS 独立出版） |
| **Briefings in Bioinformatics (BIB)** | 大类（生物学）二区非 top，**小类（生物信息学）一区** | 7.3–7.7 | Oxford Academic |

两者数字都好看，但下面的文献检索结果显示**实际投稿适配度差异很大**。

## 2. Plant Phenomics：数字诱人，但范围风险真实存在

检索其近一年含"genotype × environment"关键词的文章，命中的是：
- *High-Throughput Yield Prediction of Diallele Crossed Sugar Beet … Using **UAV-Derived** Growth Dynamics*
- *Interaction of Genotype, Environment, and Management on Organ-Specific Critical Nitrogen Dilution Curve in Wheat*（田间生理曲线实测）

期刊官方 Scope 虽然写了"连接 phenomics 与 genomics/统计/建模/计算科学"，但**实际发表的 G×E 相关文章无一例外都绑定了新的高通量表型采集**（无人机影像、田间生理实测曲线），没有找到一篇纯计算型、复用已有产量试验数据做环境×基因型预测的论文。我们的工作不产生新的表型采集数据（用的是已有产量试验记录 + 气象重提取），**投 Plant Phenomics 存在真实的"范围不符"退稿风险**，尽管分区数字最好看。不建议作为首选，除非愿意承担这个风险，或能找到令人信服的角度把"阶段感知环境特征提取"包装成"环境表型分析（envirotyping as phenomics）"。

## 3. BIB：找到直接的近一年同赛道论文，且有明确可填补的缺口

检索到 **EXGEP**（*Briefings in Bioinformatics*, 2025, vol 26 issue 4）——一篇发表仅数月的直接同赛道论文：

> *EXGEP: a framework for predicting genotype-by-environment interactions using ensembles of explainable machine-learning models*

细读后的对比：

| 维度 | EXGEP（BIB 2025） | 本项目 |
|------|---------------------|--------|
| 作物覆盖 | 仅玉米（美国 109 环境训练 + 23 环境 2022 测试，另加中国 5 地点验证） | **四作物**（玉米/小麦/燕麦/大麦，1211 环境） |
| 基线 | BRR、LightGBM、XGBoost、GBDT、RF、CNN、GEFormer | **缺失 Finlay-Wilkinson/反应范数基线**——这正是本项目发现"复杂模型打不过"的那个基线 |
| 环境距离/外推分析 | **未涉及**——报告了 23 个测试环境的逐环境表现，但未刻画"预测增益如何随环境相异度变化" | **本项目 H2 的核心内容**，且做了 BH 校正、bootstrap CI、预注册 MDE |
| 结果性质 | 全正面（"比 BRR 提升 17–42%"），无阴性/零结果讨论 | 诚实报告 H2 零结果、G3 方向分裂 |
| 泄漏/统计规范性 | 基本未讨论（自陈气象特征处理"仅用播种到收获的均值"这一局限，未讨论时间泄漏） | 全流程折内拟合、聚类稳健推断、N_eff 校正 |

**这给了一个非常具体、可写进引言的定位句式**："近期 BIB 发表的 EXGEP（2025）在单一作物、23 个测试环境上报告了集成学习相对经典基线 17–42% 的提升，但未包含反应范数基线、未检验该增益是否随环境相异度衰减、且局限于单一作物——本文在四作物、防泄漏、预注册统计框架下系统检验了这些问题，发现 Finlay-Wilkinson 反应范数本身就是极强基线，且相对该基线的增益在四作物间不存在稳健的距离依赖模式。"

这既证明了 BIB **确实接受这个方向的论文**（不是我们臆测），又给出了明确、有说服力的差异化空间。

## 4. 额外发现：G2F 竞赛论文（必引，且有一个可执行的加分项）

检索到 **Genomes to Fields (G2F) Genotype-by-Environment Prediction Competition** 系列论文（2022/2023/2024 三届，分别发在 *BMC Research Notes* 和 *Genetics*）——用的正是我们玉米队列同源的 G2F 数据，是这个数据集上事实标准的社区基准竞赛。核心发现"diverse modeling strategies deliver satisfactory results"（多种建模策略效果相近，无一支独大）与我们自己的发现（FW 反应范数与复杂模型相当甚至更优）方向一致，是很好的独立佐证。

**可执行的加分项（如果决定投入）**：把我们的玉米 269 环境 PCC(G∘z)/PCC(FW) 结果，与 G2F 竞赛已公开的排行榜结果做一次直接对照——同一数据源、社区已知基准下的横向定位，比自说自话的基线墙更有说服力。工作量：中等（需拉取竞赛公开结果 + 核对环境集是否重叠/可比），非必须但性价比较高。

## 5. 检查代码库后发现的一个真实缺口：GEFormer 未实现

协议 §5 基线墙写"沿用 Paper 1：… LightGBM、XGBoost、**GEFormer**、MeNet"，但检查 EnvIndex 和 nc 两个仓库代码，**GEFormer 和 MeNet 均未在任何脚本中实现**（`grep -rli geformer` 全仓库零命中）。GEFormer 恰好是 EXGEP 论文里被列为"近期 SOTA"直接对比的方法（Molecular Plant, 2025）——如果投 BIB，审稿人大概率会问"和 GEFormer 比过没有"。这是协议文本"声称已具备"但实际不存在的一个具体缺口，与 v1.0 冻结时"W0 清单诚实反映代码状态"的教训是同一类问题。

## 6. 建议

1. **BIB 是目前证据最充分的方法类研究首选**——分区、近一年真实同赛道论文、明确差异化空间三者都对得上；Plant Phenomics 数字更好看但范围风险是真实的，不建议作为主投目标。
2. **补实现 GEFormer 基线**（至少在玉米队列上跑一次），补上这个已知会被问到的缺口——工作量取决于是否需要自己实现（论文本身发过代码的话可以直接复用其开源实现，需要确认）。
3. **可选加分项**：拉取 G2F 竞赛公开基准做横向对照，性价比较高但非必须。
4. 引言部分应明确引用 EXGEP 和 G2F 竞赛论文，把本文的贡献定位为"该赛道近期工作遗漏的三个维度（反应范数基线、距离依赖检验、多作物）"。
