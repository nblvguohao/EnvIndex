#!/usr/bin/env bash
# seed_ensemble_run.sh — pillar A' route 2: seed-ensemble UQ gate.
# Repeats the pooled-SelectionGain LOEO (learned arm, pred dump) with
# train seeds 1 and 2.  Seed 0 already exists from the SG run, so after this
# we have 3 predictions per (env, geno) -> per-env cross-seed disagreement
# as the epistemic gate feature ("model unsure -> fall back to FW").
# --seed stays 0 everywhere: loader/cohort seed must match the §7.2 cohort.
set -uo pipefail
cd /data/lgh/envindex/repo
PY=/data/lgh/envindex/env/bin/python
export LOE_GPU_BASE=1

for S in 1 2; do
    echo ""
    echo "=== train-seed $S  START $(date '+%F %T') ==="
    $PY -u scripts/pooled_selection_gain.py --run --crops barley,oat,corn,wheat \
        --epochs 60 --fold-workers 12 --n-gpus 1 --device cuda \
        --seed 0 --train-seed $S --preds-suffix "_s$S" --n-boot 200 \
        --out-csv data/t3/sg_seed$S.csv
    echo "=== train-seed $S  EXIT=$? $(date '+%F %T') ==="
done
echo ""
echo "SEEDS_DONE $(date '+%F %T')"
