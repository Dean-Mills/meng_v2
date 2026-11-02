#!/usr/bin/env bash
# PEng diagnostics pass (2026-10-26), CPU only: error anatomy, graph coverage and perfect-grouping AP
# ceiling on HigherHRNet / YOLO11x-pose / RTMO-L. Output: outputs/diagnostics/.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
D=outputs/diagnostics; mkdir -p $D
BEST=outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt,outputs/k_head_aug_d4422c37/88389153/best.pt
NOIMG=outputs/finetune_pg_gat_hier_aug_coco/b59d602d/best.pt
declare -A ARR=([hrnet]=outputs/e2e_per_image/arrays [yolo]=outputs/e2e_cache_yolo/arrays [rtmo]=outputs/e2e_cache_rtmo/arrays)
declare -A APC=([hrnet]=outputs/e2e_ap/hrnet_cache [yolo]=outputs/e2e_ap_yolo/cache [rtmo]=outputs/e2e_ap_rtmo/cache)
for det in hrnet yolo rtmo; do
    python -m analysis.error_anatomy --arrays ${ARR[$det]} --det $det --model best=$BEST --model noimg=$NOIMG \
        --out_dir $D/anatomy_$det > $D/anatomy_$det.log 2>&1 &
done
for det in hrnet yolo rtmo; do
    python -m analysis.graph_coverage --arrays ${ARR[$det]} --out $D/coverage_$det.json > $D/coverage_$det.log 2>&1
done
wait
for det in hrnet yolo rtmo; do
    for t in 0 0.1; do
        python -m analysis.ap_ceiling --cache_dir ${APC[$det]} --conf_threshold $t --out $D/ap_ceiling_${det}_t$t.json > $D/ap_ceiling_${det}_t$t.log 2>&1 &
    done
done
wait
echo DIAGNOSTICS_DONE
