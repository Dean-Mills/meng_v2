#!/usr/bin/env bash
# PEng AP@OKS on YOLO11x-pose and RTMO-L (one at a time): raw all-image caches
# (cache_ap_detector.py), then eval_e2e_ap_peng.py with the MEng and full PEng pipelines.
# WAIT_PID: optional process to wait for first. Launch via systemd-run --user.
cd "$(dirname "$0")/.."
set -x
if [ -n "${WAIT_PID:-}" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi
MODELS=(--model headline=outputs/pg_gat_finetune_sweep/3362f752/best.pt,outputs/k_head_meng_headline_coco/a636b1c4/best.pt
        --model peng=outputs/finetune_pg_gat_hier_img_coco/372658c6/best.pt,outputs/k_head_aug_372658c6/4579ce0c/best.pt)
python cache_ap_detector.py --detector yolo --cache_dir outputs/e2e_ap_yolo/cache
python eval_e2e_ap_peng.py --cache_dir outputs/e2e_ap_yolo/cache "${MODELS[@]}" --out_dir outputs/e2e_ap_yolo
python cache_ap_detector.py --detector rtmo --detector_device cpu --cache_dir outputs/e2e_ap_rtmo/cache
python eval_e2e_ap_peng.py --cache_dir outputs/e2e_ap_rtmo/cache "${MODELS[@]}" --out_dir outputs/e2e_ap_rtmo
echo AP_DETECTORS_DONE
