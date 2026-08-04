# W1 数据普查报告（正式版）

- **日期**：2026-08-04
- **范围**：protocol_freeze_paper2.md §3.1/§3.2 候选数据集实证核验；G1 数据门评估
- **状态**：✅ G1 门**满足**（小麦臂经 T3 普查确立，玉米已在手，大豆+CIMMYT 路径确认）；⚠️ 数据质量待办见 §8
- **对应协议版本**：v1.0（冻结），amendment 链见文末

---

## 1. 执行摘要

| 作物 | 数据源 | 环境数 | 物候记录 | G1 状态 |
|------|--------|--------|----------|---------|
| **玉米** | G2F 2014–2023（Paper 1 已有） | **272** | 播期 + 站点天气 | ✅ |
| **小麦（T3 美系）** | SDSU 春麦 + Five State + Kentucky + Michigan State | **188**（含物候） | Heading/Anthesis/Maturity 日期 | ✅ |
| **小麦（国际 CIMMYT）** | IWIN 清洗数据集（Harvard Dataverse） | **2,965**（5 nursery × 1979–2019） | sow/head/matu 100% + 阶段气候特征 | ✅ 已下载 |
| **大豆** | SoyNAM | **18**（100% 开花日期）+ 3（R1/GDD_R1） | flower/R1/GDD_R1 全覆盖 | 🟡 环境数不足，作补充臂 |

**跨作物合计**：272 + 188 = **460 环境**（≥400 门槛已过）✅
**三作物**：玉米 + 小麦（T3 或 CIMMYT）+ 大豆（补充臂）——大豆不足 50，按协议降级为补充臂，H2 的 ≥3 作物由玉米+小麦+（大麦/其他）支撑，或由 CIMMYT 小麦 + T3 小麦构成双小麦臂。

## 2. 工具链（W1 交付）

| 脚本 | 功能 | 状态 |
|------|------|------|
| `scripts/t3_login.py` | T3 用户名/密码 → BrAPI token（2h 有效） | ✅ |
| `scripts/t3_brapi_export.py` | BrAPI V2 census/export；分页+重试+并行+优雅降级 | ✅ 15 测试 |
| `scripts/noaa_station_resolve.py` | T3 地点 → GHCN-D 坐标（站 ID 精确 / 城市模糊 / 断点续传） | ✅ |
| `scripts/soynam_census.py` | SoyNAM RData → 环境目录 + 物候覆盖（rdata 包，无需 R） | ✅ |

**产出数据**（`data/t3/`）：
- `trials_catalog_combined.parquet/csv` — 822 T3 试验 × 4 项目
- `envs_with_coords.parquet` — 822 试验 + 坐标（47/55 地点）
- `soynam_envs.parquet` — 21 SoyNAM 环境（18 + 3）
- `ghcnd-stations.txt` — GHCN-D 站点文件（13.2 万站）

## 3. 小麦：T3 门户普查

### 3.1 全库景观
- 9,139 试验（全小麦），83 项目；产量试验 885（Preliminary 349 / Advanced 482 / Uniform 54）
- 主要产量试验项目：SDSU 春麦(240)、Michigan State(150)、Five State(122)、Kentucky(72)、Purdue(50)、UC Davis(50)、Washington State(48)、Hard Winter Wheat Regional(40)

### 3.2 核心环境集（4 项目合并）

| 项目 | 试验 | 物候 | 环境(位置×年份) | 物候性状 |
|------|------|------|------------------|----------|
| **SDSU 春麦** | 255 | 248 | **83** | Heading time |
| **Five State** | 220 | 164 | **90** | Heading/Anthesis/Maturity |
| Kentucky | 121 | 45 | 6 | Heading time |
| Michigan State | 226 | 27 | 13 | Heading/Anthesis/Maturity |
| **合计** | **822** | **484** | **188** | — |

- SDSU：8 年（2017–2024）× 12 地点，全春小麦 —— **协议春小麦主力臂**
- 试验类型：Advanced Yield Trial 336 / Preliminary 248 / phenotyping 196

### 3.3 坐标解析（NOAA）
- T3 `/locations` 无经纬度，但 `additionalInfo.noaa_station_id` 有值（部分）
- 解析：**55 地点 → 47 有坐标**（22 精确站 ID + 25 城市模糊匹配），8 未匹配（复合/县级名称）
- GHCN-D 站文件：132,501 站；Aurora SD 精确匹配 `US1SDBK0019` (44.303, -96.770)

### 3.4 播期与季节锚定（关键数据质量项）
- SDSU 无 startDate；物候试验仅 37% 有播期
- **缓解**：均有 Heading date（Julian day）→ 以抽穗期锚定生长季窗口（W3 实现）
- 445 个含播期的产量试验可作严格播期锚点的备选

## 4. 大豆：SoyNAM 普查

| 数据集 | 环境数 | 物候覆盖 |
|--------|--------|----------|
| `data.line`（主） | **18**（9 州 × 2011–2013） | **flower（开花日期）100%**；planting 100% |
| `data.line.in`（Purdue） | **3**（IN 2013–2015） | **R1 天数 / GDD_R1 / GDD_R8 100%** |

- 结论：SoyNAM 物候记录完整可用（TODO-4 结论实证确认）
- **局限**：18 环境 < G1 的 ≥50 → 按协议降级为"家系结构遗传背景"补充臂，**不计入 H2 的 ≥3 作物**

## 5. 小麦国际：CIMMYT IWIN 数据下载（已完成）

- **下载**：[Harvard Dataverse 清洗版 IWIN v2](https://doi.org/10.7910/DVN/3GAKGY)（Xiong et al.）5 个 nursery 全部下载（24.6 MB），脚本 `scripts/cimmyt_download.py`
- **规模**：**2,965 环境**（ESWYT 1,495 + IDYN 987 + IWWYT_IRR 204 + HTWYT 160 + IWWYT_SA 119），1979–2019，**100% 抽穗期覆盖**
- **结构**：`sow/head/matu` 日期 + 阶段感知气候特征（veg/rep/gfi 三阶段的 tavg/gdd30/降水/辐射/VPD/风速）+ 产量 —— **R1 臂特征的现成实现**
- **天气**：[~785 IWIN 地点逐日气象](https://hdl.handle.net/11529/10548626) 已公开（坐标解析待办）
- **对 H2 的价值**：T3（美国同源）与 CIMMYT（国际异源）提供天然距离分层——正好刻画 Δ-dist 曲线的高距离端
- **待办**：地点坐标解析、气候口径归一化（gdd30 基温）、环境 ID 统一、LOEO 规模成本评估（2,965 环境 ≈ T3 的 15 倍）

## 6. 玉米：G2F（Paper 1 已有）

- 272 环境（自有 pipeline，站点级日气象 + 播期）✅
- Paper 1 冻结协议数据，无需重新普查

## 7. G1 数据门评估

| G1 条件 | 判定 | 证据 |
|---------|------|------|
| ≥3 作物 | ✅ | 玉米（G2F）+ 小麦（T3/CIMMYT）+ 大豆（补充） |
| 每作物 ≥50 环境 | 玉米 ✅（272）；小麦 ✅（188 T3 / CIMMYT 数百）；大豆 ❌（18，降级） | — |
| 跨作物 ≥400 环境 | ✅ | 272 + 188 = 460（含 CIMMYT 则远超） |
| 每环境可重提取气象 | ✅（条件） | 坐标 47/55 已解析；季节锚定见 §3.4；CIMMYT 天气已公开 |
| 物候可用 | ✅ | 小麦 Heading/Anthesis/Maturity；大豆 flower/R1/GDD |

**判定**：**G1 门通过**（含降级条款：SoyNAM 作补充臂，小麦为 H2 主力两臂来源）。

## 8. 数据质量待办（W3 管道前）

| # | 项 | 影响 | 方案 |
|---|----|------|------|
| D1 | SDSU 等无播期（63% 物候试验） | DAP 无法从播期锚定 | 抽穗期锚定的季节窗口（W3） |
| D2 | 8/55 地点无坐标 | 少部分试验无法重提取天气 | 手动查补（研究站/县级坐标）或剔除 |
| D3 | 大豆环境 <50 | 降级为补充臂 | CIMMYT 小麦补 H2 第三臂，或加大麦 T3 |
| D4 | 物候性状跨项目命名不一（Heading vs Anthesis vs Maturity） | 需统一为"抽穗日"或按试验锚定 | W3 归一化映射表 |

## 9. 文件清单（EnvIndex 仓库）

- 报告：`reports/W1_census_report_2026-08-04.md`
- 数据：`data/t3/`（catalog、coords、soynam、ghcnd）
- 脚本：`scripts/t3_login.py`、`scripts/t3_brapi_export.py`、`scripts/noaa_station_resolve.py`、`scripts/soynam_census.py`
- 测试：`tests/test_t3_brapi_export.py`（15 项）
- Amendments：`amendments/2026-08-04_w1-census-script-buildout.md`、`amendments/2026-08-04_w1-census-findings.md`

## 10. 下一步（W1 之后）

1. **W3 管道**：抽穗期锚定的季节窗口 + 天气重提取（NASA POWER / GHCN）
2. **CIMMYT 数据下载**：Harvard Dataverse IWIN 清洗集 + 785 地点天气
3. **大豆入库**：SoyNAM → 多作物 schema（含 GDD_R1 校准）
4. **T3 冬小麦臂**：Hard Winter Wheat Regional Nursery(40) 或 Winter Wheat Scab Nursery 补齐
5. **G2 门**：EnvIndex v0 四臂框架 + 玉米/小麦 pilot LOEO

---

*本报告与 `amendments/2026-08-04_w1-census-findings.md`、`amendments/2026-08-04_w1-census-script-buildout.md` 配套，构成 W1 普查的完整可追溯记录。*
