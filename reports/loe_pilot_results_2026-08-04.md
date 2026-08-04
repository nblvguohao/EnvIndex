# 玉米+小麦 LOEO Pilot 结果

- **日期**：2026-08-04
- **执行环境**：服务器 `100.112.165.109`（2× A100-80GB），nohup 后台
- **配置**：50 小麦（ESWYT）+ 30 玉米（G2F）环境，150 epochs，d_embed=32，rank=4，strata 分层，learnable genotype embedding
- **日志**：`/data/lgh/envindex/repo/data/server_train.log`（~66 分钟完成）

---

## 1. 结果

### 小麦 ESWYT（50 留出环境）

| 模型 | PCC (mean ± std) |
|------|------------------|
| G∘z（学习嵌入） | +0.028 ± 0.187 |
| G+E 加性 | +0.022 ± 0.186 |
| geno-mean GBLUP-lite | +0.022 ± 0.186 |
| **random-z 对照** | -0.041 ± 0.155 |
| **Δ(e)** | **+0.006** ± 0.210 |

### 玉米 G2F（30 留出环境）

| 模型 | PCC (mean ± std) |
|------|------------------|
| G∘z（学习嵌入） | +0.135 ± 0.150 |
| G+E 加性 | +0.113 ± 0.180 |
| geno-mean GBLUP-lite | +0.113 ± 0.180 |
| **random-z 对照** | +0.009 ± 0.064 |
| **Δ(e)** | **+0.022** ± 0.128 |

## 2. 解读（pilot 性质，非科学结论）

1. **机制验证成功**：跨作物 LOEO、逐环境 PCC、全基线对比（G+E/GBLUP-lite/random-z）、Δ(e) 全部工作。
2. **信号方向正确**：
   - 玉米 G∘z (0.135) > G+E (0.113)，Δ=+0.022；random-z 对照 (0.009) ≈ 0 → 学习嵌入提供真实信息。
   - 小麦 G∘z (0.028) > random-z (-0.041)。
3. **random-z 阴性对照符合预期**（≈0），§5-12 对照机制正确。
4. **PCC 绝对值低是 pilot 预期**：模型 <2M 参数、150 epochs、ESWYT 每环境基因型少（~23 行/环境弱产量方差）、玉米用近似阶段特征（滚动均温锚定）。

## 3. 局限与正式训练差距

| 项 | pilot | 正式训练 |
|----|-------|----------|
| 环境数 | 小麦 50 + 玉米 30 | 距离分层 400-500 |
| 特征 | ESWYT 自带 + 玉米近似 | 统一 R1（AgERA5/T3 重算） |
| 基因型 | 学习 embedding（无标记） | 标记嵌入（GBLUP 内核可比） |
| epochs | 150 | 300+（batch 1024） |
| GPU | 15% 单卡 | 折级并行 2 卡 |
| 对照 | G+E/GBLUP-lite/random-z | 全 15 模型基线墙 |

## 4. 后续

- 对接 `specs/loe_scaling_assessment.md`（距离分层 LOEO）
- 对接 `specs/gpu_efficiency_formal_training.md`（batch 1024 / workers 8 / 折级并行）
- 玉米精确 DAP 阶段特征（W3 管道，替代滚动均温近似）
