# CIMMYT IWIN 气候口径归一化 Spec

- **日期**：2026-08-04
- **用途**：将 IWIN 清洗数据（`*_Obs_Sim_Yld_Phe_Climate_All.tab`）的气候变量映射到协议 §4.1 R1（stage_summary）特征口径，确保跨数据源的环境表示一致
- **关联**：protocol_freeze_paper2.md §3.1、§4.1、§6；amendment 2026-08-04_w1-cimmyt-download.md

---

## 1. IWIN 气候变量定义（数据实证 + 文档推断）

| 变量 | 含义 | 置信度 | 取值示例（ESWYT veg） |
|------|------|--------|------------------------|
| `sow` / `head` / `matu` | 播种 / 抽穗 / 成熟（天数） | 高 | head=70 → 播种后 70 天 |
| `tavg_{stg}` | 阶段日均温（°C） | 高 | 13.6 |
| `tdr_{stg}` | 热发育指数（**定义待 Xiong 文档确认**） | 低 | 11.1，与 tavg 相关 ~0.5 |
| `gdd30_{stg}` | **阶段累计 >30°C 热胁迫度-日** | 高 | 中位 0，max 344.6 |
| `rs_{stg}` | 阶段日均太阳辐射（MJ/m²/d） | 高 | 11.7 |
| `p_{stg}` | 阶段日均降水（mm/d） | 高 | 1.12 |
| `rh_{stg}` | 阶段日均相对湿度（%） | 高 | 60.1 |
| `vpd_{stg}` | 阶段日均饱和差（kPa） | 高 | 0.68 |
| `ws_{stg}` | 阶段日均风速（m/s） | 高 | 1.71 |

阶段：`_veg`（营养期）`_rep`（生殖期）`_gfi`（灌浆期）。

## 2. 与协议 R1（stage_summary）特征的口径差异

协议 R1 特征（`FoldPreprocessor` stage_summary，crop_profiles 阶段窗口驱动）：
- 每阶段 × 每气象变量：`mean/min/max/sum/std`
- 附加：`heat_days_tmax_gt{阈值}`（计数）、`rain_days`、`dry_days`
- 气象列：`tmax, tmin, tmean, precipitation, solar_radiation, relative_humidity, vpd, gdd`

**差异矩阵**：

| R1 特征 | IWIN 是否有 | 一致性 | 处理 |
|---------|------------|--------|------|
| tmean 各统计量 | 仅 `tavg`（=mean） | 部分 | 无 min/max/std |
| precipitation mean/sum | 仅 `p`（日均 mm/d） | 部分 | 无 sum/min/max/std |
| solar_radiation | `rs`（=mean） | ✅ | — |
| relative_humidity | `rh`（=mean） | ✅ | — |
| vpd | `vpd`（=mean） | ✅ | — |
| gdd（基温 0°C） | `tdr`（≈？） | ❓ | tdr 定义待确认 |
| heat_days_tmax_gt30 | `gdd30`（度-日累计） | 部分 | 连续 vs 计数 |
| wind_speed | `ws` | ✅ | 协议未用 wind |
| tmax/tmin 独立 | ❌ | ❌ | IWIN 无逐日 min/max |

## 3. 归一化策略（两级）

### 3A. 轻量级（直接用 IWIN 特征，描述性用途）
- CIMMYT 环境使用其**原生阶段均值特征**（`tavg/rs/p/rh/vpd/ws/gdd30` × 3 阶段）。
- **适用范围**：H2 可预测性边界分析的"远距离端"（CIMMYT 相对 T3 训练集的嵌入距离天然远），此时 CIMMYT 仅作环境，不做 R1-vs-R2 表示对比。
- **口径声明**：在论文方法中注明 CIMMYT 臂的气候特征为"阶段均值 + 热胁迫度-日"，与 T3 管道的完整 R1 统计量不同。

### 3B. 严格级（从 AgERA5 逐日天气重算 R1）
- 下载 AgERA5 IWIN 逐日天气（`IWIN_Weather_AgERA5_20210211.txt`，786 MB，hdl:11529/10548548），用 `FoldPreprocessor(weather_mode="stage_summary", crop="wheat")` 重算与 T3 完全一致的 R1 特征。
- **适用范围**：H3 跨数据源 R1 一致性检验；EnvIndex（R2）输入需要统一特征空间。
- **成本**：786 MB 下载 + 逐地点聚合（312 ESWYT 地点 × ~60 行/地点，轻量计算）。

### 3C. 基温/阈值统一
- 小麦 GDD 基温 **0 °C**（crop_profiles 已确认）；`gdd30` 的 30°C 阈值与 R1 的 `heat_day_tmax_threshold=25°C`（wheat provisional）**不一致**——需决策：
  - 若 CIMMYT 用 IWIN 原生特征：接受 `gdd30`（>30°C 度-日）作为其热胁迫指标，注释差异。
  - 若重算：统一用 25°C 热日阈值（wheat profile）。

## 4. 建议决策（供 W3 冻结）

| 用途 | 策略 |
|------|------|
| H2 边界曲线远距离端 | **3A**（原生特征，接受口径差异，注释） |
| H3 跨数据源 R1 对比 | **3B**（AgERA5 重算，统一口径） |
| R2 EnvIndex 输入 | **3B**（需要统一特征空间才能跨数据源共享编码器） |

**结论**：CIMMYT 臂的定位决定口径。若 CIMMYT 仅贡献 H2 远距离端 → 3A 足够；若 CIMMYT 参与 H3/R2 跨作物对比 → 必须 3B。推荐**先 3A 跑通，W3 决定是否升 3B**。

## 5. 待办

- [ ] 确认 `tdr` 定义（Xiong et al. codebook 或数据文档）
- [ ] 下载 AgERA5 逐日天气（786 MB）评估 3B 可行性
- [ ] 与 T3 管道联调：同一 `FoldPreprocessor` 处理 IWIN 天气
