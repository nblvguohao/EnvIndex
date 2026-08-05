# Amendment 2026-08-05: 聚类稳健推断（dated amendment）

- **日期**：2026-08-05
- **响应**：编辑意见"聚类结构下的有效样本量与相应功效：未计算"
- **修改范围**：protocol_freeze_paper2.md §6（统计、bootstrap、MDE）
- **性质**：本 amendment 冻结后生效；定量依据见 amendments/2026-08-05_discriminator-preregistration.md

---

## 1. 聚类结构定义

环境嵌套结构：
```
site（地点，如 Ithaca NE / Aurora SD）
  └── year（年份，2017, 2018, ...）  # 交叉因子
        └── environment = site × year
```
同一 site 的不同 year 环境**不独立**（共享土壤、气候带、管理实践），环境级 bootstrap 高估有效样本量。

## 2. 实测聚类强度（G2F）

- **ICC = 0.346**（G2F 环境均值按 location 的组内/组间方差分解）
- 含义：环境均值变异的 34.6% 可由 site 归属解释 → 强聚类

## 3. 有效样本量 N_eff

```
N_eff = N / (1 + (m − 1) × ICC)
```
- 玉米（N=200，ICC=0.346）→ **N_eff = 83.9**
- 小麦（N=396，ICC=0.346）→ **N_eff = 194.3**
- 名义 N 的有效性仅 **42–49%** ——编辑关切定量确认

**所有推断基于 N_eff**：
- MDE 功效分析（玉米 0.026 / 小麦 0.048，见判别器预注册）
- 判别器检验
- Δ-dist 曲线 CI

## 4. 层级 bootstrap（替代简单环境级）

| 层级 | 重采样单元 |
|------|-----------|
| 第 1 层 | **site**（有放回抽样） |
| 第 2 层 | site 内 **year**（有放回抽样） |
| 观测 | site×year 环境 |

- Δ-dist 曲线的分箱均值 CI 用此层级 bootstrap（1,000 次）
- 与简单环境级 bootstrap 的结果**并排报告**，标注差异

## 5. 统计检验选择

1. **Δ 跨作物方向一致性**：cluster-robust 方差估计（按 site 聚类的 sandwich SE），或混合模型 `Δ ~ dist + (1|site)`
2. **判别器（分箱均值对比）**：层级 bootstrap p 值 + cluster-robust SE
3. **MDE/功效**：一律基于 N_eff（第 3 节）
4. **敏感度**：主结果报告 N、N_eff、ICC 三列；结论稳定性的名义 vs 聚类两套对照

## 6. 报告要求

每张主表/主图附带：
- `N`（名义环境数）、`N_eff`（聚类校正）、`ICC`（实测或引用）
- CI 来源标注（层级 bootstrap / cluster-robust）
- 若聚类校正改变结论方向（p 跨 0.05），显式标注

## 7. 审计项

- [ ] 功效分析脚本改用 N_eff（`derive_thresholds.py` 已实现，接入正式协议）
- [ ] bootstrap 实现双层重采样（site → year）
- [ ] 混合模型 `(1|site)` 随机效应进 Δ-dist 回归
- [ ] 每图注的 N/N_eff/ICC 字段
