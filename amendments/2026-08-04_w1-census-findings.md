# Amendment 2026-08-04: W1 数据普查结果 —— T3 小麦环境集初步确立

- **日期**：2026-08-04
- **修改范围**：protocol_freeze_paper2.md §3.1/§3.2（候选数据集与 G1 门）的实际数据核验
- **性质**：记录 W1 普查实证结果；协议正文不原位修改
- **状态**：G1 门**条件性满足**（见 §4）

---

## 1. T3 小麦门户景观（实证）

- 全库 **9,139 个试验，全部为 Wheat**（`wheat.triticeaetoolbox.org` 门户）。
- 试验类型：phenotyping_trial 8,066 / Preliminary Yield Trial 349 / Advanced Yield Trial 482 / Uniform Yield Trial 54 / 其他 ~190。
- **产量试验合计 885**，全部含位置+年份；其中 **445 含播期字段**。
- 83 个育种项目，主要产量试验来源：SDSU 春小麦(240)、Michigan State(150)、Five State(122)、Kentucky(72)、Purdue(50)、UC Davis(50)、Washington State(48)、Hard Winter Wheat Regional(40) 等。

## 2. 春小麦主力环境集：SDSU Spring Wheat

| 指标 | 数值 |
|------|------|
| 试验总数 | 255 |
| **含物候性状（Heading time - Julian date）** | **248** |
| **唯一环境（位置×年份）** | **83** |
| 地点数 | 12（Aurora, Brookings, Volga, Faulkton, Groton, Miller, Watertown, Selby, …） |
| 年份 | 2017–2024（8 年，每年 ~30 试验） |
| 试验类型 | Preliminary Yield Trial 141 / Advanced Yield Trial 99 / phenotyping 8 |
| 物候性状 | **Heading time - Julian date (JD)**（248/248 一致） |

**结论**：单项目即提供 **83 个春小麦环境**，满足 G1 门的"≥50 环境/作物"。

## 3. 关键工程发现

### 3.1 播期缺失（SDSU）
- SDSU 试验的 `startDate`/`endDate` 为 null；无播期观测。
- **影响**：DAP 对齐无法从播期锚定。
- **缓解（已具备）**：试验含 **Heading date**（Julian day）观测 → 可用抽穗期锚定生长季（抽穗 ≈ 出苗+~50 天，成熟 ≈ 抽穗+~40 天，SD 春小麦典型 5 月播种–8 月收获）。这优于 G1 门要求的"GDD 阈值退化方案"——我们直接有物候锚点。
- 需在 W3 管道中实现"抽穗期锚定的季节窗口"替代"DAP-from-播期"。

### 3.2 坐标缺失 → NOAA 站点 ID
- `/locations` 对 SDSU 地点返回 `latitude/longitude: null`，但 `additionalInfo.noaa_station_id` 有值（如 `GHCND:US1SDBK0019`）。
- **缓解**：通过 NOAA GHCN 站点元数据解析坐标，或直接以站点 ID 取 GHCN 日天气（与 NASA POWER 口径需一致性审计）。
- 这是天气管道设计的重要输入，写进 W3 待办。

### 3.3 播期可用项目（445 产量试验）
- 若某项目需严格播期锚点，可优先使用"含播期的 445 个产量试验"（来自其他项目）。

## 4. G1 门评估

| G1 条件 | 状态 |
|---------|------|
| ≥3 作物 | 玉米 ✅（G2F 272 环境已在手）；小麦 🟡（SDSU 春麦 83 环境可，其他项目待补）；大豆 SoyNAM 待普查 |
| 每作物 ≥50 环境 | 小麦 ✅（SDSU 83）；玉米 ✅ |
| 跨作物 ≥400 环境 | 待补齐（玉米 272 + 小麦 ≥83 + 大豆 8–18 不足，需更多小麦或补充臂） |
| 每环境可重提取气象 | 🟡 坐标需经 NOAA 站点 ID 解析；季节窗口需以抽穗期锚定（无播期时） |
| 物候可用 | 小麦 ✅（Heading time）；大豆 ✅（SoyNAM R1/GDD_R1） |

**判定**：小麦臂 G1 条件性满足。**卡点**：(a) 大豆环境数不足（SoyNAM ≤18，需补 CIMMYT 或大麦臂）；(b) 播期/坐标缺失的退化方案需在 W3 落地。

## 5. 下一步

1. 跑 Michigan State / Five State / Kentucky 普查，扩充环境集并验证多项目一致性（后台进行中）。
2. 设计并实现"抽穗期锚定的季节窗口"（W3 管道）。
3. NOAA 站点 ID → 坐标解析脚本。
4. 大豆 SoyNAM 普查（R1/GDD_R1 确认）。
5. CIMMYT 数据路径确认（G1 缺口）。

## 6. 工具状态

census 工具已可端到端工作（12→15 项测试，真实连通修复记录见 amendments/2026-08-04_w1-census-script-buildout.md）。已修复：token 认证、基址归一化、分页位置、并行拉取、优雅降级、物候误报（trait 描述不再触发关键词）。
