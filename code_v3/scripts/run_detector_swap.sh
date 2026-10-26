#!/usr/bin/env bash
# PEng detector swap, zero-shot: cache YOLO11x-pose and RTMO-L outputs on COCO val2017 once
# (cache_e2e_detector.py), then score every backbone on identical scenes per detector with
# eval_peng_gap.py. Native grouping = the detector's own person assignment. Sequential (one GPU
# job at a time). Launch via systemd-run --user (see scripts/launch_peng_jobs.sh for the env).
cd "$(dirname "$0")/.."
set -x
[ -f outputs/e2e_cache_yolo/summary.json ] || python cache_e2e_detector.py --detector yolo --out_dir outputs/e2e_cache_yolo
[ -f outputs/e2e_cache_rtmo/summary.json ] || python cache_e2e_detector.py --detector rtmo --detector_device cpu --out_dir outputs/e2e_cache_rtmo
KH_AUG=outputs/k_head_aug_b59d602d/$(readlink outputs/k_head_aug_b59d602d/latest)/best.pt
for DET in yolo rtmo; do
    A=outputs/e2e_cache_$DET/arrays
    # MEng headline + MEng K-head (raw frame, as in the MEng detector swap)
    python eval_peng_gap.py --arrays $A --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --k_head outputs/k_head_meng_headline_coco/a636b1c4/best.pt --out_dir outputs/peng_gap/${DET}_3362f752
    # PG-GAT-H + aug + its augmented K-head (raw frame, as scored on HigherHRNet)
    python eval_peng_gap.py --arrays $A --checkpoint outputs/finetune_pg_gat_hier_aug_coco/b59d602d/best.pt \
        --k_head $KH_AUG --out_dir outputs/peng_gap/${DET}_b59d602d_khaug
    # #3 image evidence (letterbox frame, K_lb; its K-head is not trained yet)
    python eval_peng_gap.py --arrays $A --checkpoint outputs/finetune_pg_gat_hier_img_coco/372658c6/best.pt \
        --out_dir outputs/peng_gap/${DET}_372658c6
done
echo DETECTOR_SWAP_DONE
