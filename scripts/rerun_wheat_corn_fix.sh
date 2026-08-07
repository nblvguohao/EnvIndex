#!/usr/bin/env bash
# rerun_wheat_corn_fix.sh — amendment §7.1 优先 0 续跑: 小麦+玉米 learned/PCA
# 与主批次同配置; --n-envs-wheat 800 保持与原权威队列相同的 798 环境抽样
# (rng.choice, seed=0; 首次误用 1000 已于启动后 ~10 min 终止重跑)。
set -uo pipefail
cd /data/lgh/envindex/repo
PY=/data/lgh/envindex/env/bin/python
export LOE_GPU_BASE=1
COMMON="--epochs 60 --fold-workers 12 --n-gpus 1 --device cuda"

run() {
    echo ""
    echo "=== $1  START $(date '+%F %T') ==="
    "${@:2}"
    echo "=== $1  EXIT=$? $(date '+%F %T') ==="
}

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
