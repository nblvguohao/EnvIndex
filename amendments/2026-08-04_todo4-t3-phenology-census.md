# Amendment 2026-08-04: TODO-4 完成 —— T3 小麦 / SoyNAM 物候记录普查

- **日期**：2026-08-04
- **修改范围**：protocol_freeze_paper2.md 附录 B TODO-4（T3 物候记录可用性确认，G1 通过条件）；§3.2 G1 数据门新增的"开花日期可用性"条款之证据补充
- **性质**：记录普查结论；协议正文不原位修改
- **方法**：公开文献 + T3 官方 BrAPI 文档 + SoyNAM 官方 R 包文档交叉核实（本环境网络策略阻断 WebFetch 直连，采用 WebSearch 检索权威来源）

---

## 1. 结论

**G1 数据门新增的物候要求（每环境开花/抽穗日期可用，或退化方案）→ 可满足。** 两种关键作物的物候记录均已被权威来源确认存在；无需在普查阶段就触发 GDD 阈值校准退化方案。逐环境覆盖率的最终核实在 W1–W3 数据普查时完成。

## 2. T3 小麦（WheatCAP）

**物候性状存在 ✅**，多项同行评审研究明确声明数据源自 T3：

| 来源 | 证据 |
|------|------|
| Allelic Variation in Developmental Genes and Effects on Winter Wheat Heading Date in the U.S. Great Plains（PLoS One, 2016） | **"All phenological data are available from the The Triticeae Toolbox (T3) database"** —— 299 硬红冬麦品系 × 9 环境，含抽穗日期、春化基因（Vrn）与光周期基因（Ppd）等位变异 |
| Training population selection … historical USA winter wheat panel（Theor. Appl. Genet., 10.1007/s00122-019-03276-6） | GAWN 历史数据集含 **heading date = 50% 植株抽穗的日序日（Zadoks GS55）**，遗传力 0.49–0.57；数据"available for download at the T3 database" |

**访问路径**：
- T3 为 **BrAPI V2** 兼容（wheatcap.triticeaetoolbox.org 等实例），OIDC 认证
- 官方 R 包 `TriticeaeToolbox/BrAPI.R`（GitHub，含 TUTORIAL.md）
  - `/programs` → 育种项目（如 University of Nebraska）
  - `/studies` → 试验列表 + 播种/收获日期、地点、年份
  - `/observations` → 每小区×每性状观测，`observationVariableName` 标识性状名
- 性状为**试验级**清单：抽穗/开花日期是否记录因试验而异 → W1–W3 须逐试验普查确认

**注意**：以上证据主要来自冬小麦项目。WheatCAP 春小麦育种项目是否全部记录抽穗日期需在 W1–W3 逐试验确认；抽穗/开花是小麦育种标准性状，预期大部分记录。

## 3. SoyNAM

**物候性状存在 ✅**（官方 R 包 `SoyNAM` v1.6.2，CRAN/GitHub/soybase.org）：

| 数据集对象 | 物候字段 |
|-----------|----------|
| `data.line` / `data.checks`（主数据集） | **Flowering 日期**（如 701=7/1）、**Maturity 日期**（如 901=9/1）、R8（成熟天数）、**Planting date**（如 501=5/1） |
| `data.line.in` / `data.checks.in`（Purdue 产量构成子集，印第安纳 2013–2014） | **R1（开花天数）**、R8（成熟天数）、**GDD_R1**（开花积温）、**GDD_R8**（成熟积温）、**GDD_REP**（生殖期积温） |

**关键价值**：Purdue 子集直接提供 GDD 到 R1/R8，与协议 §3.4 的 GDD 校准需求精确匹配——即便个别环境缺日期字段，也可用 GDD_R1/GDD_R8 校准大豆阶段窗口。

**参考**：Diers et al. (2018)；Xavier et al. (2016, 2017, 2018)。

## 4. 对 G1 门与协议的影响

1. **G1 物候条款判定**：T3 小麦（抽穗/开花）与 SoyNAM（R1 开花）的物候记录**已确认可获得** → 协议 §3.2 中"或文档化退化方案"的分支**暂不触发**，退化方案保留为逐环境失败时的兜底。
2. **数据 schema 影响**：物候字段可落地到 `env_meta.parquet`（新增 `heading_date` / `flowering_date` 或作物特异列）；由 W1–W3 入库管道实现。
3. **阶段窗口校准**：T3 抽穗记录 → 校准小麦 DAP 窗口（协议 §3.4 "抽穗期记录辅助校准"）；SoyNAM GDD_R1 → 校准大豆 MG 分层窗口。
4. **待办闭环**：附录 B 四个 TODO 全部完成。下一步进入 W1–W3 数据普查（T3 注册 + 逐试验物候覆盖率核实 + SoyNAM 下载 + 入库管道）。

## 5. 方法限制（记录在案）

- 本环境网络策略阻断 WebFetch 直连 T3 / soybase / data.gov，结论基于 WebSearch 检索到的权威文献与官方包文档，未直接登录 T3 逐试验核对性状清单。**逐试验物候覆盖率核验仍须在 W1–W3 实际登录 T3 完成**——这是 G1 门评审的正式证据来源。
- T3 需免费注册（OIDC 认证），注册与导出是 W1–W3 的第一项实际工作。
