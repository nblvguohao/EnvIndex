# Amendment 2026-08-04: W0 启动清单完成（TODO-1 代码部分 + TODO-2）

- **日期**：2026-08-04
- **修改范围**：protocol_freeze_paper2.md 附录 B（冻结待办清单）中的 TODO-1（代码部分）与 TODO-2
- **性质**：记录已完成事项，协议正文不做原位修改

---

## 1. 完成内容

### TODO-1（代码部分）：多作物 schema 与参数化改造 —— ✅ 已完成

在 Paper 1 代码库 `G:\cc\nc`（`gxe_budget` 包）提交 `bbb7e9c`，改动 10 个文件：

| 文件 | 改动 |
|------|------|
| `src/gxe_budget/data/crop_profiles.py`（新增） | 作物注册表：maize / wheat / soybean 的 GDD 基温、阶段窗口、热日阈值、性状单位、生殖期 DAP 窗口；`get_crop_profile()` 带 maize 回退（保持 Paper 1 单作物调用不变） |
| `src/gxe_budget/data/schema.py` | `crop` 加入 `REQUIRED_PHENOTYPE_COLUMNS` |
| `src/gxe_budget/data/g2f.py` | 表型/环境/metadata 盖章 `crop="maize"`；GDD 基温与单位改用 `MAIZE_PROFILE` |
| `src/gxe_budget/data/weather.py` | `prepare_weather_daily()` 新增 `gdd_base_temp` 参数（默认 10.0=玉米）；manifest 记录所用基温 |
| `src/gxe_budget/data/preprocessing.py` | `FoldPreprocessor` 新增 `crop` 与 `heat_day_tmax_threshold`；作物档案驱动阶段窗口与热日阈值；特征名反映实际阈值（不再硬编码 `gt30`） |
| `tests/*`（3 个更新 + 2 个新增） | fixture 补 `crop` 列；新增 `test_crop_profiles.py`（5 项）与 `test_weather_alignment_audit.py`（4 项） |

**验证**：全量单元测试 `PYTHONPATH=src python -m pytest -m "not needs_outputs"` → **110 passed, 21 deselected**，无回归。

### TODO-2：`63_audit_weather_alignment.py` 审计脚本 —— ✅ 已完成

新增 `scripts/63_audit_weather_alignment.py`，功能：
- 输入：`weather_daily.parquet` + `env_meta.parquet` + `--crop` + `--min-coverage`（默认 0.80）
- 从作物档案读取生殖期 DAP 窗口，逐环境计算该窗口的天气覆盖度
- 列出所有环境（含 metadata 中无天气记录者，覆盖度记 0）
- 任一环境低于阈值 → 退出码 1（入库门禁失败）

**验证**：合成数据端到端通过（全覆盖环境 pass / 部分覆盖 fail / 退出码 1）；4 项单元测试通过。

---

## 2. 与协议正文的关系

- 协议正文 §3.2 G1 数据门的**物候记录要求**、§3.4 作物阶段窗口**数值**均未修改；仍待 TODO-3（文献确认）与 TODO-4（T3 物候普查）。
- `crop_profiles.py` 中小麦/大豆的阶段窗口与热日阈值为**临时占位值**，明确标注为 provisional，待 TODO-3 文献确认后以新 amendment 修订。
- 代码库 `G:\cc\nc` 尚有大量与本改动无关的未提交工作区状态（CLAUDE.md、baselines.py 等）；本 amendment 只对应提交 `bbb7e9c` 的 10 个文件。

---

## 3. 对已执行分析的影响评估

- Paper 1 已产出的结果文件（outputs/、Plant-Communications-submission_2026-08-04/）**不受影响**：本次改动对玉米路径保持数值等价（`MAIZE_PROFILE` 常量 = Paper 1 原常量，测试已验证）。
- 后续任何作物入库必须通过 `63_audit_weather_alignment.py`。
