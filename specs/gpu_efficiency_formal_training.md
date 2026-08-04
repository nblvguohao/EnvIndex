# 正式训练 GPU 效率方案

- **日期**：2026-08-04
- **触发**：服务器 pilot LOEO 观测到 GPU 利用率仅 15%（tiny model + 串行 LOEO + 小 batch）；用户确认正式训练采用效率手段
- **关联**：protocol_freeze_paper2.md §4.7（算力）；specs/loe_scaling_assessment.md

---

## 1. 观察（pilot 实测）

- 模型 <2M 参数，batch=128 → A100 每 batch 微秒级算完，GPU 空闲等 CPU
- LOEO 逐折重训串行 → 无法跨折利用并行
- 显存 986 MiB / 80 GB（几乎空）
- 进程 CPU 98.6%（单核），瓶颈在 Python/数据管线

## 2. 效率手段（已参数化进 loe_pilot）

| 手段 | 参数 | 预期 |
|------|------|------|
| 大 batch | `--batch-size 512`（正式训练 1024） | GPU 利用率 ↑ |
| 数据并行加载 | `--num-workers 4` | 消除数据加载停顿 |
| pin_memory | 自动（cuda 时开启） | 减少 H2D 拷贝阻塞 |

## 3. 多卡/多折并行（正式训练核心策略）

LOEO 各折**独立**（每折留出一个环境、其余重训），天然可并行：

### 3A. 折级并行（最简单，收益大）
- 用 `concurrent.futures.ProcessPoolExecutor`（或 joblib）把 N 折分配到 2× A100
- 每折独立进程、独立模型、独立 GPU
- 2 卡 → 2× 吞吐；N 折并行上限 = 卡数×每卡并发
- 实现要点：每折从零训练（协议要求），进程间无共享状态；结果按折收集

### 3B. 作物/臂级并行
- wheat 与 corn 的 LOEO 完全独立 → 分别分配到 2 卡并行
- 4 臂（R0/R1/R1'/R2）× 多种子 × 2 卡 → 静态划分

### 3C. 跨折 batch（数学上等价但实现复杂）
- 把多折的 (train, holdout) 拼成大步 batch → GPU 算满
- 但 fold 间模型独立、无共享梯度 → 本质是多任务，收益有限，不推荐优先

## 4. 正式训练配置建议

```
--batch-size 1024 --num-workers 8 --epochs 300
# 多卡：每卡跑不同 crop/arm/seed 切片
# 折级并行：ProcessPool over folds, 2 workers (每卡 1)
```

## 5. 预算复核（对接 loe_scaling_assessment）

- 距离分层抽样 400-500 环境 + 折级并行 2 卡 + batch 1024：
  - 单折训练时间大幅缩短（batch↑ + workers↑）
  - 折并行 2× → 总时长约减半
  - 原估算 2-4 周 → 目标 **1-2 周**（回到 §4.7 预算）

## 6. 待办

- [ ] `run_loe` 折级并行封装（ProcessPool，2 workers）
- [ ] 服务器 GPU 分配验证（每进程独立卡：`CUDA_VISIBLE_DEVICES`）
- [ ] 正式训练脚本（batch 1024 / workers 8 / strata 分层 / 多卡切片）
