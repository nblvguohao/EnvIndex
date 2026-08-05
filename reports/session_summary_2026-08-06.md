# EnvIndex 项目会话总结报告（2026-08-04 ~ 2026-08-06）

- **日期**：2026-08-06
- **范围**：Paper 2 协议冻结 → W1 数据普查 → EnvIndex v0 → 规模化 LOEO → Δ-dist 曲线 → 编辑意见响应 → 阈值预注册
- **状态**：全部交付，36 测试通过，全部推送 GitHub

---

## 1. 协议体系

- **`protocol_freeze_paper2.md`**（v1.0 冻结）：4 假设（H1-H4）、四臂对比（R0/R1/R1'/R2）、5 门控（G1-G5）、dated amendment 机制
- **4 份 dated amendments**（响应编辑冻结前优先事项）：
  | 文件 | 内容 |
  |------|------|
  | `2026-08-05_R1p-no-supervision-isolation.md` | R1' 监督泄漏隔离（主任务权重恒 0 + 代码断言 + manifest） |
  | `2026-08-05_discriminator-preregistration.md` | 判别器定量预注册（数据推导阈值） |
  | `2026-08-05_cluster-robust-inference.md` | 聚类稳健推断（ICC=0.346，N_eff，层级 bootstrap） |
  | `2026-08-05_agera5-access-constraint.md` | AgERA5 数据约束 → NASA POWER 替代 |

## 2. 数据基座（W1 普查实证）

| 作物 | 环境数 | 物候 | 来源 |
|------|--------|------|------|
| 玉米 G2F | 272 | ✅ 播期 | Paper 1 已有 |
| 小麦 T3（美系） | 188 | ✅ Heading date | T3 BrAPI census |
| **小麦 CIMMYT** | **2,965** | 100% 抽穗期 | Harvard Dataverse IWIN |
| 大豆 SoyNAM | 18 + 3 | 100% flower/R1 | R 包解析 |

跨作物 ≥400 环境达成（272+188+2965）。

## 3. EnvIndex v0（协议 §4.2 核心）

- **编码器**：2 层 MLP-Mixer（<2M 参数），z_e ∈ R^d（d∈{8,16,32}）
- **多头**：G∘z 低秩交互（主）+ 环境均值（aux A）+ InfoNCE（aux B）
- **可学习基因型嵌入**（改进1）
- **pilot 验证**：玉米 Gz 0.307（200 环境，random-z 0.003 对照干净）

## 4. LOEO 规模化 + Δ-dist 曲线

### 服务器配置（2× A100-80GB）
- 折级并行：16 workers/2 卡（32 workers 过度订阅反而拖慢）
- batch 1024，距离分层抽样

### 结果

| 指标 | 小麦远距（396） | 玉米（150/200） |
|------|----------------|-----------------|
| PCC(G∘z) | +0.127 | +0.294 |
| PCC(G+E) | +0.156 | +0.269 |
| random-z 对照 | -0.001 | +0.002 |
| **Δ(e)** | **-0.030** | **+0.024** |

### 关键结论
1. **random-z 对照≈0 两作物验证**：学习嵌入是真实信号，非噪声
2. **Δ 分作物方向不同**（小麦负/玉米正）——H2"可预测性边界因环境类型而异"初步信号
3. **pilot 尺度无单调边界**：观测 Δ 在 MDE 内（0.026/0.048）→ 功效不足，非无边界
4. **斜率判别器结构性失效**（MDE 1.78-3.10）→ 需远距环境拉开 dist

## 5. 数据推导阈值（预注册）

| 阈值 | 玉米 | 小麦 |
|------|------|------|
| MDE 平均 Δ（N_eff 校正） | 0.026 | 0.048 |
| 判别器（分箱对比） | 0.024 | 0.047 |
| SelectionGain 决策 | 0.155 Mg/ha (1.6%) | 0.109 t/ha (2.2%) |
| d_crit 半宽（0.2×IQR） | 0.0005 | 0.0004 |
| N_eff / ICC | 83.9 / 0.346 | 194.3 / 0.346 |

## 6. 基础设施代码（36 测试通过）

| 类别 | 脚本/模块 |
|------|-----------|
| 数据获取 | `t3_login` / `t3_brapi_export` / `cimmyt_download` / `soynam_census` / `nasa_power_extract` |
| 坐标 | `noaa_station_resolve` / `cimmyt_coords` |
| 特征 | `corn_features`（GDD 阶段边界）/ `r1_unified`（统一 R1） |
| 模型 | `encoder`（MLP-Mixer）/ `train`（多任务）/ `sampling`（距离分层） |
| 分析 | `loe_pilot`（LOEO+并行）/ `delta_dist_curve` / `derive_thresholds` |

## 7. 关键排障教训

1. 32 workers 过度订阅（141>128 核）反而拖慢 → 16 最优
2. GPU 残留进程占显存（75GB）→ 按 `nvidia-smi` PID 清理
3. 余弦距离嵌入塌缩 → 环境级 R1 特征欧氏距离
4. 逐作物增量保存 → 慢作物不拖死快作物
5. `_init_worker` GPU 取模 bug → 32 workers 崩溃
6. ESWYT 129 NaN 产量 → loss NaN → strata 崩溃

## 8. 待办

- [ ] **H2 判别器定论**：跑 PCA 对照 LOEO，比较 z_e vs PCA 曲线（判别器测试）
- [ ] 统一 R1（NASA POWER）重跑 LOEO，对比 3A 原生特征
- [ ] H4 跨作物迁移、生物学注释章节
- [ ] 摘要/引言写作（编辑 AUDITOR_INPUT_NEEDED）
