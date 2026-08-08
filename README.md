# Protocol: 学习环境索引与 G×E 可预测性边界（多作物）

- **协议版本**：v1.1（首次冻结后修订，2026-08-08）
- **冻结日期**：2026-08-04（v1.0）
- **冻结前身**：`Paper2实验设计_学习环境索引与GXE可预测性边界.md` v1.1（经 6 透镜敌对评审修订）
- **状态**：✅ 已冻结。v1.1 为 2026-08-05～08-07 期间 10 份 dated amendment 的正文并入（见附录 C），此后修改仍须通过 dated amendment 机制（见末尾 §A）
- **v1.1 摘要**：四作物全量重跑完成（玉米 269/小麦 800/燕麦 79/大麦 63，共 1,211 环境）。**H2（可预测性边界）经严格重估未获支持**——四作物 BH 校正后均无可检出的 Δ–dist 依赖（见 §4.4.1、附录 C）；**G3 门（学习索引 vs PCA 对照）通过**但方向随作物分裂、与监督数据量相关而非 PCA 强度；论文叙事按 §11.1 第三行落地："H1+H3 通过，H2 失败"。基线墙扩展（GBLUP、LightGBM、pooled SelectionGain、seed-ensemble UQ）与"可靠性门控"（gate science，附录 E）分析已启动
- **与 Paper 1 的关系**：Paper 1（泄漏安全残差 G×E 基准，玉米 G2F）提供基础设施与严格性框架；本文档的阶段感知环境特征从 Paper 1 的建设性章节升级为方法核心。两篇共用同一套 harness，边际成本递减
- **设计评审**：v1.0 草案经 6 透镜敌对评审（统计、泄漏、可行性、创新、生物学、设计一致性），13 agents × 298k tokens。3 致命 + 7 重要问题已在冻结前修复。评审报告存档于 workflow `wf_8be77c7c-e0d`。

---

## 1. 一句话主张与可检验假说

**工作标题（英）**：*When Is Genotype-by-Environment Interaction Predictable? A Learned, Stage-Aware Environmental Index Defines the Boundary Across Crops*

**一句话主张**：G×E 是否可预测、显式建模何时有收益，不是经验问题而是可刻画的一般性规律——它由"环境能否嵌入一个低维、生理可解释的索引空间"以及"训练集对该空间的覆盖度"共同决定；阶段感知的学习环境索引给出这个空间，并且其结构在多个作物间部分共享。

**可检验假说**：

| 编号 | 假说 | 证伪条件 |
|------|------|----------|
| H1 | 环境可嵌入低维（d≤32）空间，且嵌入的近邻结构与已知 mega-environment 划分一致 | 嵌入检索一致性不显著高于原始气象特征的随机基线 |
| H2 | 显式 G×E 建模相对反应范数基线的 LOEO 增益（Δ_FW，定义见 §4.4.1，v1.1 起替代原 Δ(G+E)——见附录 C 方法学修正），是"目标环境到训练集的嵌入距离"的单调递减函数（剂量-反应曲线），且在 ≥3 种作物上方向一致。**关键区分检验**：z_e-based 距离产生的 Δ-dist 曲线必须显著不同于 PCA-based（无学习、纯描述性降维）距离产生的曲线——拒绝"任何结构化嵌入均产生递减曲线"的零假设 | 曲线在 ≥2 种作物上不存在或方向相反；**或效应不存在**——逐环境统计量在预注册 CI/MDE 下不显著（BH 校正后），即"无可检出依赖关系"，与"方向相反"作为并列的独立证伪结果类别；或 PCA-based 对照曲线在 ≥2 种作物上呈现相同定性模式（递减且显著）——此时 H2 被结构性证伪。**现状（v1.1）：四作物 BH 校正后 Δ_FW–dist 逐环境相关均不显著（附录 C，2026-08-06 方法学修正 §7.2(3)）→ H2 按"效应不存在"类别证伪** |
| H3 | 阶段感知编码（DAP 对齐）在跨作物 LOEO 上系统优于日历窗口与原始日序列编码 | 增益 < MDE 或仅在玉米成立 |
| H4 | 在一种作物上训练的环境编码器，经特征空间对齐后应用于另一作物，其嵌入距离与表型距离在**作物内部**保留显著正相关（Partial Mantel 检验，以作物指示矩阵条件化），表明跨作物环境空间结构部分共享 | 作物条件化的 Partial Mantel 检验不显著；或未经条件化的整体 Mantel 显著但完全由作物间对比驱动 |

H2 是主图假说（可预测性边界）；H1 是方法成立的前提；H3 承接 Paper 1；H4 是影响力放大器（generality 的直接证据）。

---

## 2. 与 Paper 1 的接口

**直接复用（已存在或 PR #1 已补）**：

- LOEO 评估 harness、环境级 block bootstrap、paired Wilcoxon + BH、MDE 功效分析模板
- FoldPreprocessor 泄漏安全协议（train-only 拟合、manifest 记录、`dap_alignment_mode` 可追溯）
- 天气对齐审计（`scripts/63_audit_weather_alignment.py`，新增作物时作为入库门禁）
- DAP 对齐的阶段窗口机制（DEFAULT_STAGE_WINDOWS 的生成逻辑）
- 基线墙框架（GBLUP 三件套、FW 反应范数、LightGBM/XGBoost、深度模型两条）

**新增（本文档范围）**：

- 多作物数据 schema（§3.2）
- 环境索引模块 EnvIndex（§4.2）
- 四臂环境表示对比协议（§4.1）
- 可预测性边界分析（§4.4）
- 跨作物迁移评估（§4.5）
- DNA 基础模型辅助特征管道（§4.6，不承担核心主张）

**多作物就绪约束**：Paper 1 代码库当前为玉米专属（`DEFAULT_STAGE_WINDOWS`、GDD 基温硬编码；`crop` 字段不存在）；PR #1 将实现 `crop` 为 schema 一级字段、阶段窗口/GDD 基温/性状单位全部按作物参数化。此项改造为 W0 启动清单最高优先级项——在协议冻结前必须完成并推送到可验证仓库。

---

## 3. 数据

### 3.1 候选数据集矩阵

置信度说明：✅= 已核实公开可得且有明确访问路径；🔍= 公开但访问细节/环境元数据需 W1–W2 普查确认；⚠️= 备选，拼装成本高或覆盖不足。

| 作物 | 数据集 | 环境数 | 基因型 | 气象/播期 | 置信度 |
|------|--------|--------|--------|-----------|--------|
| 玉米 | G2F 2014–2023 | 272（自有 pipeline） | 5,027 杂交种 | 站点级日气象 + 播期 ✅ | ✅ 已在手 |
| 小麦 | T3 Wheat（USDA WheatCAP/Breedbase） | 数百试验，可用环境待普查 | 多育种项目，PHG 填充 1.3M–2.9M 标记 | 试验含地点与播期字段；气象走 NASA POWER 重提取 | ✅ 公有领域，需免费注册 [^1^][^2^] |
| 大麦/燕麦 | T3 Barley / T3 Oat | 待普查 | 同上 | 同上 | 🔍 同平台附加作物 [^1^] |
| 大豆 | SoyNAM | ≥8（产量全群体），最多 18（农艺性状子集） | 5,600 RIL（39–40 个双亲家系） | 2011–2013 年站点×年份；站点坐标公开，气象可重提取 | ✅ 公开 [^3^] |
| 小麦（国际） | CIMMYT IWIN / ESWYT / SAWYT | 数十年 × 每年 100–400 个国际站点 | 精英品系 | 站点元数据公开程度待普查 | 🔍 数据获取路径需确认 [^4^][^5^] |
| 水稻 | 3K RG + 公开表型 | 多环境覆盖有限 | 3,000+ 品种 | ⚠️ 环境数偏少 | ⚠️ 备选 |
| 高粱 | SAP / TERRA-REF | TERRA-REF 以单站点高频表型为主 | SAP 多环境研究分散 | ⚠️ 拼装成本高 | ⚠️ 备选 |

**环境协变量统一管道**：NASA POWER 日气象重提取（沿用 Paper 1，确保 G2F 与其他作物同源同口径）；SoilGrids 土壤静态变量；管理变量仅保留公开可得字段（播期、灌溉标记），缺失即缺失，不做臆造填充。

### 3.2 Go/No-Go 数据门（W3 末评审）

**通过条件（全部满足）**：≥3 种作物；每种 ≥50 个可用环境（有播期或可推算 DAP 的锚点）；跨作物合计 ≥400 个环境；每个环境可通过坐标+日期重提取完整生长季日气象；**每个环境至少确认开花日期（或等价物候：小麦抽穗期、大豆 R1 日期）可用，或文档化退化方案指定在无物候观测情况下以 GDD 阈值校准阶段边界**。

**当前预期**：玉米（272）+ 小麦 T3（待普查，预期 100+）+ 大豆 SoyNAM（8–18）已大概率满足；SoyNAM 环境数偏少，**不计入 H2 的 ≥3 作物最低要求（≤18 环境无法贡献稳定的剂量-反应数据点）**，作为"家系结构遗传背景"的补充臂而非主力。若小麦 T3 环境元数据不足，降级路径为 CIMMYT ESWYT 子集或加大麦 T3。

### 3.3 数据 schema（多作物扩展）

沿用 Paper 1 的 parquet 规范，一级变更：

```
phenotype.parquet    : plot_id, crop, environment_id, genotype_id, trait, value, ...
env_meta.parquet     : environment_id, crop, site_id, lat, lon, year, planting_date,
                       harvest_date, irrigation, mega_environment(可选), ...
weather_daily.parquet: environment_id, date, day_after_planting, tmax, tmin, tmean,
                       precipitation, solar_radiation, relative_humidity, vpd, gdd, ...
genotype.parquet     : genotype_id, crop, marker_biallelic_codes / marker_id + allele_dosage
env_stage_features   : environment_id, crop, stage_name, feature_name, value
                       （阶段窗口按作物参数化，见 §3.4）
```

所有新作物入库前必须通过 `63_audit_weather_alignment.py` 的生殖期覆盖检查（作物特异 DAP 区间）。

### 3.4 作物特异阶段窗口（协议参数，冻结前需文献确认初值）

| 作物 | 阶段定义锚点 | GDD 基温 | 生殖期 DAP 窗口（初值，普查后修订） |
|------|-------------|----------|-------------------------------------|
| 玉米 | 沿用 Paper 1 六阶段 | 10 °C | 166–240 |
| 小麦（春/冬分开建模） | 出苗–拔节–抽穗–开花–灌浆–成熟 | 0 °C | 待定，抽穗期记录辅助校准 |
| 大豆 | 出苗–营养期–R1 开花–R3 结荚–R5 灌浆–R7 成熟 | 10 °C | 待定，MG 组分层 |

冬小麦越冬期处理：以"有效积温日"替代日历 DAP，避免越冬段稀释窗口。**有效积温日定义（冻结后修改需 dated amendment）：**
- **冷驯化期**（日均温 < 5°C 持续）：GDD 贡献 = 0（停止活跃生长）
- **春化满足判定**：累计 0–7°C 冷量小时数达到基因型无关阈值（如 ≥50 天等效春化温度），标记春化满足标志位
- **返青后**：恢复正常 GDD 累积（基温 0°C）
- 若 T3 小麦包含冬小麦且春化数据不可得，将冬小麦降级为补充分析，仅以春小麦支持主要声称。此条写入协议附录。

---

## 4. 方法设计

### 4.1 四臂环境表示对比（核心实验结构）

| 臂 | 表示 | 说明 |
|----|------|------|
| R0 | 原始日序列 | 气象变量日序列直接入模（现状上界） |
| R1 | 阶段感知摘要 | Paper 1 的 stage_summary 特征（每阶段 mean/min/max/sum/std + 热日/雨日/干日） |
| R2 | 学习环境索引 | 本文核心：编码器输出 d 维环境向量 z_e（§4.2） |
| R1' | 无监督 R1 编码（对照） | R1 特征 → 与 R2 相同架构的编码器 → 仅辅助任务 A+B 训练（**无产量监督信号**）→ 同一预测头。**目的：分离"编码器架构的表示容量"与"产量监督学习信号"的贡献。** 若 R2 超越 R1'，增益来自产量监督；若 R2 ≈ R1'，增益仅来自架构容量 |

**预测头共享约束**：R0 产生可变长度日序列、R1/R1'/R2 产生固定维度向量，预测头不可原样共享。采用统一方案：R0 序列通过简单均值池化聚合为固定向量后进入共享预测头（§4.3），并在敏感性分析中对比注意力池化与 LSTM 池化以界定该聚合选择的混淆效应。

### 4.2 EnvIndex 模块（R2）

**输入**：环境 e 的阶段感知特征矩阵（n_stages × n_features，R1 输出）+ 静态协变量（土壤理化属性、气候区分类、≥100km 网格单元空间特征——**不含精确经纬度**，避免站点身份泄露）。

**编码器**：2 层 Transformer 或 MLP-Mixer（参数量 < 2M，A100 单卡分钟级训练），输出 z_e ∈ R^d，d ∈ {8, 16, 32} 作为消融维度。

**训练信号（多任务）**：

1. **主任务**：z_e 条件化产量预测（与 §4.3 预测头联合训练）；
2. **辅助任务 A（结构正则）**：环境均值预测——从 z_e 线性读出环境产量均值（该量在折内仅用训练环境估计，无泄漏）；
3. **辅助任务 B（对比）**：同 mega-environment / 同年份的环境对在嵌入空间中更近（InfoNCE，正样本对定义仅用训练集元数据；**明确排除"同站点"作为正对条件**——站点作为随机效应在编码器外部按折估计；mega-environment 仅以环境与土壤协变量定义，例如 Köppen-Geiger 气候区 × 土壤质地类别，不得从产量表现模式反推）；
4. **消融**：主任务-only（无辅助）、主任务+A、主任务+A+B 三档。消融框架为强制项而非可选——用于量化各辅助任务的增量贡献。

**解释性读出**：对 z_e 各维做与阶段×气象变量的置换重要性分析，输出"哪些生育阶段×变量驱动环境差异"——这是生物学注释章节的素材，挂公共 QTL/基因数据库（如已知开花期、抗旱位点）做富集。

### 4.3 预测头

| 头 | 公式骨架 | 用途 |
|----|----------|------|
| G+E 基线 | ŷ = g(G) + e(z_e)，**与 G∘z 共享相同的 z_e** | ⚠️ **v1.1：环境内 PCC 下与纯基因型均值基线数学恒等**（`ge_pred = env_mean + (geno_mean − global_mean)`，env_mean 在单个留出环境内为常数，PCC 对平移不变）——不再作为 H2 主对照臂，保留仅作跨环境池化指标（SelectionGain@10%/RMSE）的参照 |
| **Finlay–Wilkinson 反应范数**（v1.1 新增主对照臂） | ŷ = μ_i + b_i·Ê_e，逐基因型斜率 b_i，环境指数 Ê_e = 训练折内拟合、留出环境由环境协变量预测（**不可用留出环境实测均值**，复用辅助任务 A 的读出机制） | **H2 的参照系（v1.1 起）**——不退化，逐基因型重排序，见附录 C 方法学修正 §7.1"M3" |
| G∘z 低秩交互 | + (U G)·(V z_e)，秩 r ∈ {1, 2, 4}，参数计数 ≤ G+E 基线的 1.2× | 显式 G×E 建模臂 |
| FiLM 条件化 | z_e → γ, β 调制基因型编码器各层 | 深度臂 |
| 树基线 | LightGBM/XGBoost，特征 = 基因型标记 + 环境表示 | 严格性地板（Paper 1 教训：树必须先被打败）；v1.1 已实现四作物版 `scripts/lightgbm_loe.py`（玉米另加 top-32 标记 PC） |

**关键约束（v1.1 修订）**：Δ_FW(e) = PCC(G∘z, e) − PCC(FW, e) 仅测量"显式交互建模相对反应范数基线"的增益。G+E 加性基线不再用于此对比——环境内 PCC 指标下它与纯基因型均值基线完全恒等，无法承载"消除编码质量混淆"的原设计意图（详见附录 C）。另设消融分析：G+E 基线使用 R0/R1 特征 vs 使用 z_e，以量化编码质量差异的独立贡献（该消融的解读需同时受上述退化限制约束）。

### 4.4 可预测性边界分析与前瞻性验证（H2，主图）

#### 4.4.1 描述性分析

**定义（v1.1 修订）**：对每个留出环境 e，**Δ_FW(e) = PCC(G∘z 模型, e) − PCC(Finlay-Wilkinson 反应范数, e)**（原 `Δ(e) = PCC(G∘z) − PCC(G+E)` 定义已废弃，见 §4.3 与附录 C 方法学修正）。附指标 Δ_rank(e)：环境内 PCC 对"无 G×E 零模型（固定基因型排序）"的对比，作为总价值参考，不作 H2 的主判据。

**解释变量**（全部仅用训练集计算，无泄漏）：

- **嵌入距离** dist(e)：**当前实现**为 e 的 R1 阶段摘要特征（标准化后）到训练集环境特征的欧氏 k-NN 平均距离（k=5）——**注意这不是 z_e 嵌入空间距离**（早期 z_e 余弦距离在弱编码器下塌缩，已改用 R1 特征空间距离，见开发记录）；若后续改回 z_e 距离需以 dated amendment 记录并重算；
- **覆盖度**：训练集嵌入空间的凸包体积 / 主成分解释率；
- **嵌入质量**：训练集内的检索一致性得分（§4.2 辅助 B 的留出评估）。

**分析**：Δ_FW(e) 对 dist(e) 的剂量-反应曲线（LOESS + 分箱均值 ± 环境级 bootstrap CI），分作物绘制后叠加。**趋势统计量的强制口径（v1.1，附录 C 修正规则 R1-R3）**：一切相关系数/斜率必须逐环境（非分箱均值）计算；分箱均值对比仅用于可视化和预注册判别器（§4.4.2），报告时必须同时给出 bootstrap CI 与预注册 MDE；跨作物对比表必须使用统一聚合口径。

#### 4.4.2 关键区分检验（PCA 对照，H2 的强制性诊断）

**动机**：X 轴 dist(e) 和 Y 轴 Δ(e) 都源于同一个学习到的嵌入 z_e。需排除以下替代解释——"任何结构化嵌入表示（无论是否经产量监督学习）都产生形态相同的递减曲线"。

**检验方案**：将 z_e 替换为原始阶段摘要特征矩阵的前 d 个 PCA 主成分（PC1−PCd，d 与 z_e 维度相同，无产量监督，纯描述性降维），重新训练相同的 G∘z 预测头，计算 PCA-based Δ-dist 曲线。对比检验：

1. z_e-based 曲线的斜率（或整体效应量）是否在统计上显著不同于 PCA-based 曲线（bootstrap 双曲线差异检验）；
2. 在 ≥3 作物上，z_e-based 曲线是否一致更陡峭或更早出现拐点。

若 PCA-based 在 ≥2 作物上呈现与 z_e-based 相同的定性模式（递减且显著），H2 被结构性证伪——递减不是学习型环境索引的专属属性，而是任何结构化嵌入 + 过参数化交互模型的必然结果。**PCA 对照为 H2 的强制性诊断项，不可降级为可选。**

#### 4.4.3 前瞻性边界验证（H2 的决策规则检验）

**动机**：§4.4.1 的描述性曲线是回顾性的——在观测到所有环境的表型后拟合。若声称"边界"而非"趋势"，必须证明在未见环境下该曲线具有预测校准度。

**方案**：

1. **分层预留**：按 dist(e) 分位数分层，预留 20% 环境作为前瞻验证集（确保覆盖各距离区间）；
2. **曲线拟合**：在剩余 80% 环境上拟合 Δ–dist 关系（LOESS 或单调样条）；
3. **前瞻性评估**：对每个预留环境，基于其 dist(e) 预测 Δ̂(e)，检验：
   - 校准度：Δ̂(e) 与真实 Δ(e) 的偏离（MSE 分解为偏差+方差）；
   - 区分度：Δ̂(e) 对 Δ(e)>0 的 AUC / 方向一致性；
   - 阈值估计：若关系支持单调递减，估计 Δ 降至 0 的临界距离 d_crit 及其 bootstrap CI；
4. **报告**：验证集上的全部指标，不 cherry-pick。

若此前瞻性验证不成立或方向不一致，主图标题从"可预测性边界（Boundary）"降级为"可预测性与环境距离的关系（Relationship）"，报告为描述性发现而非决策规则。

**已知先例支撑该问题真实存在**：环境外推增益被报道依赖于 mega-environment（高粱温带 vs 亚热带）且随目标环境观测数增加而改善——但文献中只有案例观察，无人给出跨作物的定量边界曲线 [^6^][^7^]。

### 4.5 跨作物迁移（H4）

玉米上训练的 EnvIndex 编码器，经特征空间对齐（per-variable z-score 标准化至目标作物分布）后应用于小麦/大豆环境特征（不微调编码器权重）。

**主检验（强制）**：
1. 在每种**作物内部**定义表型距离（环境均值产量的秩相关距离）；
2. 检验作物内嵌入距离与表型距离的 Mantel 相关性；
3. **Partial Mantel 检验**（以作物指示矩阵条件化）作为决定性诊断——分离"跨作物共享的环境结构"与"作物间基线差异驱动的伪相关"。

**解读矩阵**：

| Mantel（整体） | Partial Mantel（条件化） | 含义 |
|----------------|--------------------------|------|
| 显著 | 显著 | 真正的跨作物环境结构共享 |
| 显著 | 不显著 | H4 失败——相关性由作物间差异驱动 |
| 不显著 | 不显著 | H4 失败——无迁移结构 |

显著正相关 + Partial Mantel 通过 = 环境空间结构跨作物部分共享。若 Partial Mantel 不通过，H4 如实报告失败，主图不受影响。此外，若 H4 的零样本迁移不成立，讨论中考虑轻量领域适配（在目标作物少量环境上微调编码器）作为后续分析。

### 4.6 DNA 基础模型辅助特征（辅助分析，不承担核心主张）

按 Paper 1 冻结协议的"pretraining as an auxiliary analysis"条款延续：用 PDLLMs / PlantCaduceus（消费级显卡可推理，参数 20M–225M）对基因区序列提取嵌入，作为基因型侧的辅助特征加入树模型与 FiLM 头 [^8^][^9^]。报告其边际贡献与置换对照；若无效，作为阴性辅助结果如实报告——不影响主线。

### 4.7 算力预算

EnvIndex 编码器 < 2M 参数，四臂 × 全 LOEO（400+ 环境）× 5 seed：A100×2 约 1–2 周；FM 嵌入提取为一次性推理任务（5070Ti 可承担 PDLLMs 规模模型）；树基线 CPU。无算力风险项。

---

## 5. 基线墙（在 Paper 1 八条基础上扩展）

1–8. 沿用 Paper 1：GBLUP-G、GBLUP-G+E、GBLUP-G∘E（Hadamard）、FW 反应范数、LightGBM、XGBoost、GEFormer、MeNet
9. **envRtype-EC**：文献标准 enviromic 核（EC 矩阵 → 反应范数核），代表手工 envirotyping 路线
10. **因子回归/MegaLMM-lite**：潜性状 × 环境协变量回归的简化复现，代表潜变量环境建模路线（完整 MegaLMM 引用对比，不必全量复现）[^7^]
11. **AMMI**：每作物经典参照
12. **随机环境嵌入对照**：z_e 替换为同维随机向量（R2 的 negative control，必含）
13. **PCA 嵌入对照**（强制，H2 诊断项）：z_e 替换为 R1 阶段摘要特征的 PCA 前 d 个主成分 → 相同 G∘z 预测头。H2 的 Δ-dist 曲线必须在 ≥3 作物上显著不同于此对照的曲线
14. **R1 无监督编码对照**（强制，R2 消融项）：R1 特征 → 与 R2 相同编码器架构 → 仅辅助任务 A+B（无产量监督）→ 预测头。分离编码器架构容量 vs 产量监督学习信号的贡献
15. **跨作物均值基线**：预测每个环境的产量为该作物的总体均值（零成本校准项，所有 R² 和 PCC 的锚点）

CGM-GP（作物生长模型×基因组预测）路线：引用 Heslot/Technow/Messina 等结果做讨论定位，不复现——其需要作物模型参数化专家投入，超出纯公开数据+二人算力的范围 [^10^]。论文标题和声称范围明确限定在"统计 envirotyping 范式"内，将 CGM-GP 的扩展留作讨论中的未来工作。

**v1.1 实现进度**：GBLUP（`scripts/corn_gblup.py`，VanRaden GRM，玉米队列专属，h²=0.5 敏感性）、四作物 LightGBM（`scripts/lightgbm_loe.py`，环境协变量+折内基因型目标编码，玉米另加标记 top-32 PC）、pooled SelectionGain@10%（`scripts/pooled_selection_gain.py`，依赖 `run_loe(dump_preds=True)` 的逐 plot 原始尺度预测）、seed-ensemble 认知不确定性（`scripts/seed_ensemble_run.sh`，train-seed 1/2 复算，`--seed` 恒为 0 以保留 §7.2 权威队列）均已落地。

### 5.1 可靠性门控（Gate Science，附录 E，v1.1 新增支柱 A'）

G4 门未通过（H2 边界主张不成立）后新增的探索性分析：能否在**不知道真实标签**的前提下，逐环境预测"该信神经网络臂（G∘z）还是 FW 基线"，为部署场景提供实用价值（即使总体上无法建立距离依赖的边界曲线）。

- **朴素距离门控**：以 dist(e) 预测赢家——**失败**（AUC 0.57，证实 dist(e) 与哪个模型赢无关，与 §4.4.1/G4 的零结果一致）
- **oracle 上限**：`negative_transfer_predictor.py` 给出理论上限 +0.04～+0.06 PCC/作物（若有完美门控）
- **富特征门控**（`gate_rich_features.py`）：折前可得特征（FW 岭回归杠杆值、环境索引岭回归 R²、FW 斜率异质性）→ AUC 0.77（大麦）/ 0.70（燕麦），AUC 本身经 bootstrap 确认显著（详见附录 E.1）；**但部署增益经 bootstrap+BH 校正后四作物无一显著**（2026-08-08 完成，`gate_bootstrap_significance.py`）——排序信息真实但样本量不足以支撑可部署收益，不构成论文正面主张，作为附录如实报告
- 定位：附录分析，不影响 §11.1 的主叙事门控判定

---

## 6. 评估协议（沿用并强化 Paper 1 严格性条款）

- **主协议**：LOEO 全量循环（跨作物 400+ 环境逐留一），指标为环境级 Pearson PCC、RMSE、SelectionGain@10%、NDCG@10%
- **选择指标定义**：SelectionGain@10% = 跨作物内所有测试折池化后选 Top-10% 基因型的平均表型 vs 全群体均值（以固定比例替代固定数量，避免因作物间基因型数差异不可比）；平局以随机打破处理并报告 10 次随机打破的均值 ± SD
- **统计**：paired Wilcoxon + BH 多重校正；**预注册分析家族**：指定主要终点（PCC）、次要终点（RMSE、SelectionGain@10%、NDCG@10%），每家族报告比较总数与校正后显著阈值；**前置 MDE 功效分析**——从 Paper 1 G2F 数据的 per-environment PCC 残差方差分布正式推导当前设计对 Δ=0.03 PCC 的检验功效（至少按作物×性状组合验证），功效 < 80% 则先扩环境数
- **Bootstrap**：环境级 block bootstrap（1,000 次）；**新增层级 bootstrap** 作为敏感性分析（年→站点→环境，按站点×年份聚类重采样），评估方差分量在各层级的分布
- **DAP vs GDD 澄清**：GDD 基温已在 §3.4 指定为阶段边界的主要机制；DAP 窗口仅用于跨作物天气对齐审计（`63_audit_weather_alignment.py`）。避免审稿人错误认为阶段边界基于固定日历天数
- **禁用条款**：任何 n=1 的确定性 split（forward_year / leave_year）不得作为结论性证据，仅作描述性附录（Paper 1 教训写入协议）
- **泄漏安全**：沿用 Paper 1 全部条款；新增——嵌入空间的任何拟合（含辅助任务）严格在训练折内，通过代码级强制（非文档约定）；所有环境级统计量（均值、方差、覆盖度指标）按折重新计算；`dap_alignment_mode` 写入每个结果的 manifest
- **阴性对照**：随机嵌入对照（§5-12）+ 环境标签置换对照 + PCA 嵌入对照（§5-13）各 1 组

---

## 7. 图表清单

| 编号 | 内容 | 对应假说 |
|------|------|----------|
| Fig. 1 | 概念图：阶段感知编码 → 环境索引空间 → 可预测性边界 | 全文 |
| Fig. 2 | 多作物数据矩阵与环境覆盖（地图 + 嵌入空间覆盖度） | 数据 |
| Fig. 3 | 学习环境嵌入结构：mega-environment 一致性、跨作物叠加 | H1, H4 |
| Fig. 4 | 四臂对比（R0/R1/R1'/R2）× 全基线墙 × 跨作物 LOEO | H3 |
| **Fig. 5** | **主图：Δ_FW–dist 剂量-反应曲线（Y 轴 v1.1 起为 Δ_FW，非原 Δ(G+E)；分作物 + 叠加 + PCA 对照曲线 + 前瞻性验证集标记）** | **H2** |
| Fig. 6 | 嵌入维度生物学注释：驱动环境差异的阶段×变量与已知位点富集 | 生物学章节 |
| Fig. 7 | 前瞻性边界验证校准图：Δ̂ vs 真实 Δ（预留 20% 环境） | H2 |
| Table 1 | 数据集与 census 审计汇总 | 数据 |
| Table 2 | 基线墙全结果（含 CI 与校正后 p） | H3 |
| Table 3 | 消融（辅助任务、d、秩 r、R1' 对照） | 方法 |

---

## 8. Go/No-Go 门与降级路径

| 门 | 时点 | 通过条件 | 未通过的降级路径 |
|----|------|----------|------------------|
| G1 数据门 | W3 末 | §3.2 全部满足（含新植物候记录要求） | 降为 2 作物 + 描述性第三作物；仍不足则退回单作物深度版（H2 仅玉米，期刊目标下调一档） |
| G2 嵌入质量门 | W6 末 | H1 成立（检索一致性显著优于随机）**且**站点泄露审计通过（嵌入距离不以站点 ID 为首要结构） | 放弃学习索引主线，R1（阶段摘要）升格为方法核心，论文重心完全转向边界分析 |
| G3 PCA 对照门 | W7 初 | z_e 与 PCA 对照的**逐环境配对 bootstrap 差异**（非分箱均值）95% CI 不含 0，经 BH 校正存活（v1.1 修订，附录 C 规则 R2）| **✅ 已通过（2026-08-07 重跑）**：四作物配对差全部 BH 存活，但方向随作物分裂（小麦+0.019、玉米+0.050 为正；燕麦−0.068、大麦−0.116 为负），与监督数据量（环境数）而非 PCA 对照强度相关 |
| G4 边界效应门 | W7 末 | Δ_FW–dist 剂量-反应曲线在 ≥3 作物方向一致，**且**逐环境相关系数 BH 校正后显著，**且**（若声称边界而非趋势）前瞻性验证集校准度可接受（v1.1 修订：明确要求逐环境统计量，非分箱均值） | **❌ 未通过（2026-08-07 重跑）**：四作物 BH 校正后 Δ_FW–dist 逐环境相关均不显著（附录 C §7.2(3)）→ 按总降级条款执行，边界主张不成立，主打多作物泄漏安全基准（§11.1 第三行已落地） |
| G5 生物学注释门 | 投稿前 | 至少 1 个可解释嵌入维度与已知生物学一致 | 删除生物学章节，纯方法+基准投稿 |

**总降级条款**（写入协议）：若 G2、G3 或 G4 未通过，论文框架为"多作物泄漏安全 G×E 基准 + 诚实的环境表示对比"，学习环境索引作为辅助分析保留——与 Paper 1 的 fallback 条款同构。门控结果到期刊的映射矩阵见 §11.1。

---

## 9. 行动计划

| 阶段 | 行动 | 产出 |
|------|------|------|
| W0（启动清单） | 创建 `63_audit_weather_alignment.py`（当前不存在）；完成 PR #1 多作物改造（crop 字段 + 阶段窗口/GDD 基温/性状单位参数化 + 消除所有硬编码玉米假设）并推送到可验证仓库 | 多作物就绪代码库 + 审计脚本 |
| W1–W3 | **并行 A**（数据普查）：T3 注册与试验清单导出、SoyNAM 下载、CIMMYT 访问路径确认；验证播期/物候记录/坐标/气象；**并行 B**（管道建设）：小麦/大豆入库管道，全部通过天气对齐审计；作物特异阶段窗口文献值确认（小麦抽穗/春化、大豆 MG 分层） | 普查报告 + 3 作物入库 parquet + 审计报告 + G1 门评审记录 |
| W4–W6 | EnvIndex v0：四臂框架（R0/R1/R1'/R2）跑通，玉米+小麦 pilot LOEO（子集 50 环境快循环）；PCA 嵌入对照纳入 pilot | G2 门评审记录 + PCA 对照初步结果 |
| W7 | 边界分析 pilot：玉米全量 LOEO 的 Δ–dist 曲线 + PCA 对照 + 前瞻性验证设计确认 | G3/G4 门预评审 |
| W8 | 全门评审 → 协议冻结（冻结稿 + dated amendment 机制沿用 Paper 1 惯例） | protocol_freeze_paper2.md |

**时间线说明**：若 PR #1 多作物改造遇到不可预见困难（64+ 脚本的级联变更），W0 可能延长至 2 周，整体时间线变为 10 周。

---

## 10. 风险登记册

| # | 风险 | 概率 | 影响 | 缓解 |
|---|------|------|------|------|
| R1 | 多作物数据拼装失败（播期缺失、坐标不全、访问受限） | 中 | 高 | G1 门前置；备选作物队列（大麦/燕麦/CIMMYT） |
| R2 | 撞车：envirotyping+ML 领域活跃（MegaLMM、envRtype、巴西 enviromics 团队） | 中 | 中 | 区分点=泄漏安全+跨作物边界曲线（文献仅有个案观察）；W1 做系统文献查新；速度优先 [^6^][^7^][^11^]。**2026-08-08 复查**：领域内已有呼吁"可学习动态环境表示"的综述 [^12^]，尚无人做本文式的严格外推检验；跨领域（遥感基础模型 LOCO 评估 [^14^]）已出现结构相同的阴性先例——巩固而非削弱本文定位，"阴性结果亦是贡献"的叙事有同期支撑 |
| R3 | 学习嵌入不增益（R2 ≤ R1） | 中 | 中 | G2 门降级路径；阴性结果在基准框架下仍可发表 |
| R4 | 跨作物迁移无结构（H4 失败） | 高 | 低 | H4 非核心；修订后加入 Partial Mantel 强制诊断；零样本大概率失败，需考虑领域适配 |
| R5 | 性状尺度/遗传力跨作物不可比，Δ 定义失真 | 中 | 中 | 作物内标准化（z-score per crop）；遗传力分层敏感性分析 |
| R6 | FM 辅助特征再次 null（重蹈 syntax 覆辙） | 中高 | 低 | 其定位即为辅助分析，自带置换对照，无效即如实报告 |
| R7 | PCA 对照证伪 H2（任何结构化嵌入均产生递减曲线，z_e 无额外信息） | 中 | 致命 | G3 PCA 对照门前置；若触发则框架降级为系统化消融 + 泄漏安全基准；H2 仅作为辅助描述性结果保留。**已实现（2026-08-07）：风险以另一种形式触发**——不是"PCA 与 z_e 同形"，而是二者均不呈现 dist(e) 依赖关系（H2 直接判定"效应不存在"，比原风险描述更彻底）；G3 本身通过（z_e 确实优于 PCA），但与 H2/G4 无关，见 §4.4.1、§8 |
| R8 | 站点身份泄露不可完全消除（粗粒度空间特征 + 无同站点 InfoNCE 正对后仍有残留结构） | 中低 | 中 | 站点泄露审计纳入 G2 门评审（嵌入距离不以站点 ID 为首要 PCA 成分）；讨论中显式报告残留泄露量级 |
| R9 | PR #1 多作物改造超出预期时间（64+ 脚本级联变更、测试覆盖不足） | 中 | 中 | W0 启动清单先行；若 W0 未完成则时间线延长至 10 周；诚实反映代码状态而非假定改造已完成 |

---

## 11. 期刊策略与叙事

### 11.1 门控结果 → 期刊映射矩阵

| 门配置 | 叙事重心 | 目标期刊 |
|--------|----------|----------|
| G1–G5 全部通过 | "G×E 可预测性边界" + 多作物一般性 + 泄漏安全严格性 | Nature Communications / Genome Biology |
| H1+H3+H4 通过，H2 仅描述性 | "多作物泄漏安全 G×E 基准 + 环境表示系统消融 + 跨作物环境结构共享证据" | Plant Communications / Molecular Plant |
| **H1+H3 通过，H2 失败** | **"多作物泄漏安全 G×E 基准 + 诚实的环境表示对比"** | **Plant Communications / GigaScience** |
| H1 通过但 H3 仅玉米 | 单作物深度版 + 学习环境索引方法论 | GigaScience / Briefings in Bioinformatics |
| G1 失败（<3 作物） | 玉米深度版：泄漏安全 G×E 基准 + 环境表示对比（即 Paper 1.5） | GigaScience / The Plant Genome |

**✅ v1.1 现状：第三行已实际触发**（G3 通过、G4 未通过，见 §8）。以下为该行叙事的定位强化（2026-08-08 文献查新）：

- **叙事顺序**：Paper 1 先投（泄漏审计建立可信度与引用锚点），Paper 2 于 2027 年中投出，引言直接引用 Paper 1 的量化结果作为动机（"单作物已证实的泄漏与统计陷阱，在多作物尺度上意味着什么"）
- **正面定位（v1.1 新增）**：2025 年一篇 Frontiers 综述明确呼吁"可学习、动态的环境表示"取代固定参数化/PCA 式环境索引 [^12^]，但未提供跨作物、防泄漏、按环境距离分层的严格外推检验——本文正是对该呼吁的首次此类实证检验，结果为诚实阴性。同期一项遥感基础模型嵌入在撒哈拉以南非洲玉米产量上的 Leave-One-Country-Out 评估得出结构相同的结论（学习嵌入相对工程特征无优势，归因于分布漂移而非表示质量），并明确将其定位为"可复现阴性基准"[^14^]——为本文的叙事与投稿定位提供跨领域的同期先例。基因组学领域另有先例表明深度学习不必然优于经典方法 [^15^]；Finlay-Wilkinson 反应范数作为环境外推预测的"奠基性方法"仍是本领域公认的强基线 [^13^]，支持 v1.1 将其确立为 H2 主对照臂（§4.3）的选择
- **时机考量**：2026 年综述已开始承认"环境外推有条件"但无人系统刻画 [^6^]——这个空位的窗口期估计 1–2 年，是本项目节奏的现实约束

---

## 12. 参考文献锚点

[^1^]: USDA-ARS, The Triticeae Toolbox (T3), public domain dataset. https://catalog.data.gov/dataset/the-triticeae-toolbox-cbfa8
[^2^]: T3 Wheat portal (Breedbase, WheatCAP). https://wheat.triticeaetoolbox.org ; https://triticeaetoolbox.org/
[^3^]: Genomic predictions of genetic variances and correlations among traits for breeding crosses in soybean (SoyNAM; Heredity, 2024). https://www.nature.com/articles/s41437-024-00703-3
[^4^]: Enhanced radiation use efficiency and grain filling rate … CIMMYT elite spring wheat yield trial (Scientific Reports, 2024). https://www.nature.com/articles/s41598-024-60853-6
[^5^]: Genetic Contribution of Synthetic Hexaploid Wheat to CIMMYT's Spring Bread Wheat Breeding Germplasm (IWIN/ESWYT/SAWYT). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6710277/
[^6^]: Integrating Envirotyping and Phenomics for AI-Enabled Multi-Environment Genomic Prediction in Crop Breeding (Agronomy, 2026)——"Environmental Extrapolation Remains Conditional". https://www.mdpi.com/2073-4395/16/10/1019
[^7^]: MegaLMM improves genomic predictions in new environments (latent regressions on environmental covariates). https://escholarship.org/content/qt1225h1m6/qt1225h1m6.pdf
[^8^]: Foundation models in plant molecular biology: advances, challenges, and future directions (Frontiers in Plant Science, 2025)——GPN/AgroNT/PDLLMs/PlantCaduceus 参数与算力对比. https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1611992/full
[^9^]: A foundational large language model for edible plant genomes (AgroNT; Communications Biology, 2024). https://www.nature.com/articles/s42003-024-06465-2
[^10^]: Genomic selection: Essence, applications, and prospects (The Plant Genome, 2025)——CGM-GP 整合综述. https://acsess.onlinelibrary.wiley.com/doi/10.1002/tpg2.70053
[^11^]: Envirotyping applications across crops (Elmerich et al. 2023 欧洲大豆 600+ 环境；Resende et al. 2024 巴西玉米 enviromics；Winans et al. 2024 美国高粱). https://innovationsagriculture.pensoft.net/article/157255/download/pdf/
[^12^]: Rethinking genomic selection under environmental uncertainty: toward learnable and dynamic environmental representations (Frontiers in Plant Science, 2025)——呼吁以可学习/动态环境表示取代固定参数化与 PCA 式降维索引；未提供跨作物、防泄漏、按距离分层的外推严格检验，是本文 v1.1 定位的直接对话对象. https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1754446/full
[^13^]: Regression approaches for modelling genotype-environment interaction and making predictions into unseen environments (arXiv, 2507.18125, 2025)——确认 Finlay-Wilkinson 反应范数为环境外推预测的奠基性方法，支持 v1.1 将其确立为 H2 主对照臂. https://arxiv.org/abs/2507.18125
[^14^]: Do Foundation Model Embeddings Improve Cross-Country Crop Yield Generalisation? A Leave-One-Country-Out Evaluation in Sub-Saharan Africa (arXiv, 2605.08113, 2026)——跨领域同期先例：遥感基础模型嵌入在严格 LOCO 评估下相对工程特征无优势，归因于分布漂移，明确定位为"可复现阴性基准". https://arxiv.org/abs/2605.08113
[^15^]: Deep learning methods may not outperform other machine learning methods on analyzing genomic studies (Frontiers in Genetics, 2022)——基因组学领域内深度学习不必然优于经典方法的既有先例. https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2022.992070/full

---

## 附录 A. Dated Amendment 机制

本协议冻结后的任何修改必须使用以下格式记录在 `amendments/` 目录下：

```
amendments/YYYY-MM-DD_<简短描述>.md
```

每份 amendment 须包含：
- 日期
- 修改范围（节/条款号）
- 修改前原文
- 修改后文本
- 修改理由
- 对已执行分析的影响评估（若已有结果受影响，需说明重跑范围）

原协议正文不做原位修改。所有 amendment 汇总于本文档的变更日志。

---

## 附录 B. 冻结待办清单（Outstanding Pre-Freeze TODOs）

以下项目在协议冻结时尚未完成，须在实验执行前解决。完成时以 dated amendment 记录结果。

- [x] **TODO-1**：PR #1 完成并推送到可验证仓库（多作物改造）—— 代码部分完成于 2026-08-04（见 amendments/2026-08-04_w0-multicrop-and-audit-script.md）；推送待办见备注
- [x] **TODO-2**：`63_audit_weather_alignment.py` 创建并测试 —— 完成于 2026-08-04
- [x] **TODO-3**：小麦/大豆阶段窗口文献确认（含春化/GDD 基温引用）—— 完成于 2026-08-04，见 amendments/2026-08-04_todo3-crop-stage-windows-literature.md
- [x] **TODO-4**：T3 物候记录可用性确认（G1 通过条件）—— 完成于 2026-08-04，见 amendments/2026-08-04_todo4-t3-phenology-census.md

**备注**：TODO-1 的代码改造提交于 `G:\cc\nc` 仓库 `bbb7e9c`（10 文件，110 测试通过）。该仓库当前有大量无关的未提交工作区状态，尚未推送 remote；是否推送及以何种方式推送由仓库所有者决定。EnvIndex 仓库本身（本文档）待 TODO-3/TODO-4 完成后单独提交。

---

## 附录 C. 变更日志

| 日期 | 版本 | 变更 | Amendment 文件 |
|------|------|------|----------------|
| 2026-08-04 | v1.0 | 协议冻结。冻结前已融入 6 透镜敌对评审反馈（3 致命 + 7 重要修复，详见设计文档 v1.1 修订日志） | — |
| 2026-08-04 | — | 完成 TODO-1 代码部分（多作物改造，`nc` 提交 `bbb7e9c`）+ TODO-2（`63_audit_weather_alignment.py`）。附录 B 勾选更新 | amendments/2026-08-04_w0-multicrop-and-audit-script.md |
| 2026-08-04 | — | 完成 TODO-3（小麦/大豆阶段窗口文献确认，`nc` 提交 `80d772f`）。附录 B 勾选更新 | amendments/2026-08-04_todo3-crop-stage-windows-literature.md |
| 2026-08-04 | — | 完成 TODO-4（T3 小麦 / SoyNAM 物候记录普查：抽穗/开花日期确认可得，G1 物候条款可满足）。附录 B 勾选更新 | amendments/2026-08-04_todo4-t3-phenology-census.md |
| 2026-08-04 | — | W1 数据普查实证：T3 小麦门户 9,139 试验；SDSU 春小麦 83 个环境带抽穗期物候；坐标缺失→NOAA 站点 ID；播期缺失→抽穗期锚定方案 | amendments/2026-08-04_w1-census-findings.md |
| 2026-08-05 | — | AgERA5 访问受限，改用 NASA POWER 重提取（CIMMYT 小麦） | amendments/2026-08-05_agera5-access-constraint.md |
| 2026-08-05 | — | 判别器预注册：数据驱动阈值推导（MDE、d_crit、N_eff） | amendments/2026-08-05_discriminator-preregistration.md |
| 2026-08-05 | — | 聚类稳健推断（N_eff 校正，ICC=0.346） | amendments/2026-08-05_cluster-robust-inference.md |
| 2026-08-05 | — | R1' 无监督对照的监督泄漏隔离修正 | amendments/2026-08-05_R1p-no-supervision-isolation.md |
| 2026-08-06 | — | H2 边界方向修订（**已于同日晚些时候被下一条 amendment 整体撤回，保留存档**） | amendments/2026-08-06_H2-boundary-direction-revision.md |
| 2026-08-06→08-07 | **v1.1 核心** | **方法学修正**：撤回上条方向修订；发现 M1（分箱相关系数误作趋势证据）/M2（预注册判别器缺失 CI）/M3（G+E 基线环境内 PCC 下与基因型均值恒等）/M4（小麦 PCA 对照为无损满秩旋转）四项统计/设计缺陷；发现并修复代码缺陷 D1（目标未标准化）/D2（基因型主效应默认初始化淹没信号）；Δ 指标重设计为 Δ_FW（Finlay-Wilkinson 反应范数对照）；四作物（玉米269/小麦800/燕麦79/大麦63）修复后全量重跑：**H2 按"效应不存在"证伪，G3 通过但方向分裂**。本 amendment 的"协议条款待更新"清单已于 v1.1 全部并入正文（§1/§4.3/§4.4.1/§7/§8/§11.1） | amendments/2026-08-06_methodological-correction-delta-dist.md |
| 2026-08-07 | — | 基线墙扩展（GBLUP、四作物 LightGBM、pooled SelectionGain@10%、seed-ensemble UQ）+ 可靠性门控（gate science）启动，见 §5.1/附录 E.1 | 未单独成 amendment，随 v1.1 一并记录（提交 `6686000`） |
| 2026-08-07 | — | 遗传力分层稳健性检验（`corn_heritability.py`）：H² 预测绝对精度但不预测 Δ_FW，见附录 E.2 | 未单独成 amendment，随 v1.1 一并记录 |
| 2026-08-08 | **v1.1** | **协议正文整体修订并入**（本次变更）：§1 H2 证伪条件类别、§4.3 预测头表（FW 替代 G+E 为主对照）、§4.4.1 Δ_FW 定义与 dist(e) 显式定义、§7 Fig.5 轴说明、§8 G3/G4 门判定结果、§11.1 现状标注、新增 §5.1 与附录 E；新增文献定位 [^12^][^13^][^14^][^15^]（2026-08-08 文献查新，详见下方"v1.1 文献定位"小节） | 本次修订即为 dated amendment 汇总的正文并入，无新增 amendment 文件 |

**v1.1 文献定位（2026-08-08 查新，支撑上表最后一行）**：确认 Finlay-Wilkinson 反应范数在文献中仍是环境外推预测的奠基性方法 [^13^]（佐证 v1.1 将其确立为 H2 主对照臂的选择）；确认领域内已有综述呼吁"可学习动态环境表示"取代固定/PCA 索引但未做本文式的严格外推检验 [^12^]；发现跨领域同期先例（遥感基础模型嵌入在 Leave-One-Country-Out 评估下无优势，明确定位为"可复现阴性基准"）[^14^]，与本文"H1+H3 通过、H2 诚实阴性"的叙事互相印证；确认基因组学领域内"深度学习不必然优于经典方法"已有先例 [^15^]。四篇均为 2025–2026 新增引用，纳入 §12 与 §10/§11.1 相应条目。

---

## 附录 D. 设计评审摘要

v1.0 草案于 2026-08-04 经 6 透镜敌对评审（workflow `wf_8be77c7c-e0d`），13 agents × 298k tokens。评审档案位于 workflow transcript 目录。

**评审发现的致命问题（冻结前已修复）**：

1. **剂量-反应曲线结构性循环论证**：X 轴 dist(e) 和 Y 轴 Δ(e) 同源于 z_e → 新增 PCA 嵌入对照作为 H2 强制性诊断（§4.4.2）
2. **站点身份泄露**：原始经纬度 + InfoNCE "同站点"正对 → 替换为粗粒度空间特征 + 排除同站点（§4.2）
3. **代码库玉米专属**：声称的多作物改造不存在 → 诚实反映状态 + W0 启动清单（§2, §9）

**各假说评审评估存活概率**：H1 85% / H2 35%（修复后 50%）/ H3 60%（修复后 65%）/ H4 10%（修复后 20%）。详细分析见评审报告。

---

## 附录 E. 可靠性门控与稳健性附加分析（v1.1 新增，2026-08-07～08-08）

G4 门未通过后的两条附加分析支柱，均为探索性/附录定位，不改变 §11.1 的主叙事门控判定。

**E.1 可靠性门控（Gate Science）**：见 §5.1。核心结论——dist(e) 与"哪个模型（G∘z vs FW）在该环境上更准"无关（朴素距离门控 AUC 0.57，与 H2/G4 零结果一致）；用折前可得的丰富特征（FW 岭回归杠杆值、环境索引岭回归 R²、FW 斜率异质性）可达 AUC 0.70–0.77（大麦/燕麦）。

**Bootstrap 显著性检验（2026-08-08 完成，`scripts/gate_bootstrap_significance.py`，结果 `data/t3/gate_bootstrap.csv`，详见 `reports/gate_science_bootstrap_2026-08-08.md`）**：环境级 bootstrap（n=3000）+ 四作物 BH 校正，分类门控与回归门控分开检验（避免逐次抽样取 max 的 winner's-curse 偏差）。**结论为二分**：(a) AUC 本身的 bootstrap CI 在大麦（0.765 [0.618,0.897]）、燕麦（0.701 [0.578,0.813]）、池化（0.568 [0.533,0.604]）上均不含 0.5，排序信息真实存在；(b) 但**部署增益（gated PCC − 单臂最优基线 PCC）四作物 BH 校正后无一存活**——大麦回归门控最接近（点估计 +0.036 PCC，原始 p=0.054），受限于样本量（63 环境）未能转化为显著收益。**判定：门控信号真实但当前样本量下不足以支撑可部署的显著增益，不构成论文正面卖点，作为附录如实报告**（呼应本文"一切声称须过 bootstrap/BH 关卡"的方法论立场）。至此该分析线已完结，无剩余开放项。

**E.2 遗传力分层稳健性检验**（`scripts/corn_heritability.py`，回应"H2 零结果是否仅是低质量/高噪声环境的伪影"这一预期审稿人问题）：以 G2F 玉米队列（269 环境，plot 级重复结构，随机截距混合模型 `y ~ 1 + (1|geno)` 估计逐环境广义遗传力 H²，中位数 0.450，范围 [0.000, 0.871]）为对象：

- H² 强预测**绝对**预测精度——各模型臂的 PCC(G∘z)/PCC(FW)/PCC(geno-mean) 与 H² 的 Pearson/Spearman r 均在 +0.54～+0.57（p<1e-4），符合预期（遗传力越高，任何模型都更容易预测）；
- H² **不**预测 Δ_FW（交互增益）——Spearman r=+0.058（p=0.34），Pearson r=+0.030（p=0.62），三分位分层后 Δ_FW 在低/中/高 H² 组分别为 −0.005/+0.018/+0.001，无单调趋势；
- **结论**：H2 的零结果不是噪声/低遗传力环境的统计伪影——它在遗传力充分的环境中同样成立。此为该零结果稳健性的独立证据，非仅依赖 §4.4.1 的 BH 校正
- **覆盖限制（如实报告）**：小麦 ESWYT（约 12% 重复率）与 T3 系（仅对照品种重复）不支持稳定的逐环境 H² 估计，此分析仅覆盖玉米队列，作为覆盖限制而非静默跳过
