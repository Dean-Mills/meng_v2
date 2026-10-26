#!/usr/bin/env bash
# PEng: augmented K-head on the #3 image-evidence embeddings (372658c6), then the E2E gap
# table with K-head / clamp rows on all three cached detectors (HigherHRNet, YOLO11x-pose,
# RTMO-L). One GPU job. Launch via systemd-run --user (see scripts/launch_peng_jobs.sh).
cd "$(dirname "$0")/.."
set -x
G=372658c6; B=outputs/finetune_pg_gat_hier_img_coco/$G
python train_k_head_aug.py --checkpoint $B/best.pt --name k_head_aug_$G --epochs 10 || exit 1
KH=outputs/k_head_aug_$G/$(readlink outputs/k_head_aug_$G/latest)/best.pt
python eval_peng_gap.py --checkpoint $B/best.pt --k_head $KH --out_dir outputs/peng_gap/${G}_khaug
for DET in yolo rtmo; do
    python eval_peng_gap.py --checkpoint $B/best.pt --k_head $KH \
        --arrays outputs/e2e_cache_$DET/arrays --out_dir outputs/peng_gap/${DET}_${G}_khaug
done
echo K_HEAD_IMG_DONE
