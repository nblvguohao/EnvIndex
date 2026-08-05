# Amendment 2026-08-05: AgERA5 统一 R1 重算的数据访问约束

- **日期**：2026-08-05
- **修改范围**：protocol_freeze_paper2.md §3.1 CIMMYT 条目；specs/cimmyt_climate_normalization.md §3B
- **性质**：记录数据访问阻塞；改进2 的提取模块已完成，3B 重算执行待数据授权

---

## 1. 状态

| 项 | 状态 |
|----|------|
| `src/envindex/r1_unified.py`（统一 R1 提取模块） | ✅ 完成（stage_summaries 跨源一致语义） |
| 3B 路径：AgERA5 逐日天气重算 R1 | ⚠️ **被数据访问阻塞** |

## 2. 阻塞证据

- **AgERA5 IWIN 天气文件（786MB，hdl:11529/10548548）为受限数据集**：
  - data.cimmyt.org：下载中断（SSL UNEXPECTED_EOF，多次重试）
  - Harvard Dataverse：HTTP 401 Unauthorized（需数据使用协议）
  - 服务器 `100.112.165.109` 无法访问 data.cimmyt.org / dataverse.harvard.edu（URLError）
- 地点表 `IWIN_Locations_AgERA5_20210211.txt` 公开可下载（已在 W1 获取）——但**逐日天气**受限。

## 3. 影响与降级

- **CIMMYT 臂的 R1 特征**：暂用 IWIN 自带阶段均值特征（3A 路径，specs/cimmyt_climate_normalization.md 已记录口径差异）。
- **统一 R1 一致性（H3 跨源对比）**：需要 AgERA5 天气。**需向 CIMMYT 申请数据使用协议（数据共享协议/联系 IWIN 数据库）**后才能执行 3B。
- **玉米 G2F / 未来 T3**：可用本地天气直接跑 `r1_unified`（不依赖 AgERA5）。

## 4. 待办

- [ ] 向 CIMMYT IWIN 申请 AgERA5 天气数据使用（数据协议）
- [ ] 获批后：下载 → `r1_unified.build_r1_from_daily`（小麦 profile）→ 统一 R1 特征
- [ ] 若短期无法获批：在论文中如实声明 CIMMYT 臂用原生阶段特征（3A），H3 跨源 R1 对比限于玉米+小麦(T3)

## 5. 与规模化 LOEO 结果的关系

规模化 LOEO（reports/loe_scaled_results_2026-08-05.md）已用 IWIN 原生特征（3A）得到结果；统一 R1（3B）将作为后续增强，不阻塞当前机制验证。
