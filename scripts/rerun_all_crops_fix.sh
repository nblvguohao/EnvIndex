#!/usr/bin/env bash
# rerun_all_crops_fix.sh — amendment §7.1 优先 0: D1/D2 修复后四作物全量重跑
# 统一配置（与修复验证大麦一致）: d_embed=32, d_geno=32, rank=4, epochs=60,
# batch=512, plot_cap=100, seed=0; learned + PCA 两臂, FW 由 run_loe 内置输出。
# 运行顺序: 短作物先行（尽早暴露远程环境问题）, 小麦最后（798 环境, 最大头）。
# LOE_GPU_BASE=1: 远程 GPU 0 被他人任务占用, fold worker 全部落在 GPU 1。
# 与主批次同配置; --n-envs-wheat 800 保持与原权威队列相同的 798 环境抽样
# (rng.choice, seed=0; 1000 会抽到不同的超集队列——已于首跑终止更正, 见续跑脚本)。
set -uo pipefail
cd /data/lgh/envindex/repo
PY=/data/lgh/envindex/env/bin/python
export LOE_GPU_BASE=1
COMMON="--epochs 60 --fold-workers 12 --n-gpus 1 --device cuda"

run() {  # run <label> <cmd...>
    echo ""
    echo "=== $1  START $(date '+%F %T') ==="
    "${@:2}"
    echo "=== $1  EXIT=$? $(date '+%F %T') ==="
}

run "[1/6] barley learned" $PY -u scripts/barley_loe.py --embed-mode learned --rank 4 \
    --out-results data/t3/loe_barley_fix_learned.parquet $COMMON
run "[2/6] barley pca" $PY -u scripts/barley_loe.py --embed-mode pca --rank 4 \
    --out-results data/t3/loe_barley_fix_pca.parquet $COMMON
run "[3/6] oat learned" $PY -u scripts/oat_loe.py --embed-mode learned \
    --out-results data/t3/loe_oat_fix_learned.parquet $COMMON
run "[4/6] oat pca" $PY -u scripts/oat_loe.py --embed-mode pca \
    --out-results data/t3/loe_oat_fix_pca.parquet $COMMON
run "[5/6] wheat+corn learned" $PY -u scripts/loe_pilot.py \
    --n-envs-wheat 800 --n-envs-corn 400 --d-embed 32 --d-geno 32 --rank 4 \
    --batch-size 512 --plot-cap 100 --embed-mode learned \
    --out-results data/t3/loe_fix_learned.parquet $COMMON
run "[6/6] wheat+corn pca" $PY -u scripts/loe_pilot.py \
    --n-envs-wheat 800 --n-envs-corn 400 --d-embed 32 --d-geno 32 --rank 4 \
    --batch-size 512 --plot-cap 100 --embed-mode pca \
    --out-results data/t3/loe_fix_pca.parquet $COMMON

echo ""
echo "ALL_DONE $(date '+%F %T')"
