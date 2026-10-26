#!/usr/bin/env bash
# PEng crowd test, one GPU job end to end (launch via systemd-run --user):
#   1. fine-tune finetune_pg_gat_hier_img_crowd_coco from train_pg_gat_hier_aug/a0c59235 (+ THA evals,
#      fixed-scene gap table) - the image-evidence recipe with crowd over-sampling
#   2. augmented K-head on it (train_k_head_aug.py, same recipe as k_head_aug_372658c6)
#   3. PGA gap tables with K-head on the HigherHRNet / YOLO / RTMO caches
#   4. AP@OKS at threshold 0 on the three all-image caches
# A/B arm: finetune_pg_gat_hier_img_coco/372658c6 + k_head_aug_372658c6/4579ce0c (already scored).
cd "$(dirname "$0")/.."
set -x
EVAL_SYN=0 SYN_DIR=outputs/train_pg_gat_hier_aug/a0c59235 SYN_CFG=train_pg_gat_hier_aug \
    FT_CFG=finetune_pg_gat_hier_img_crowd_coco bash scripts/run_pg_gat_hier.sh || exit 1
FT=outputs/finetune_pg_gat_hier_img_crowd_coco/$(readlink outputs/finetune_pg_gat_hier_img_crowd_coco/latest)
G=$(basename "$FT")
python train_k_head_aug.py --checkpoint "$FT/best.pt" --name "k_head_aug_$G" --epochs 10 || exit 1
KH=outputs/k_head_aug_$G/$(readlink "outputs/k_head_aug_$G/latest")/best.pt
python eval_peng_gap.py --checkpoint "$FT/best.pt" --k_head "$KH" --out_dir "outputs/peng_gap/${G}_khaug"
for DET in yolo rtmo; do
    python eval_peng_gap.py --checkpoint "$FT/best.pt" --k_head "$KH" \
        --arrays "outputs/e2e_cache_$DET/arrays" --out_dir "outputs/peng_gap/${DET}_${G}_khaug"
done
python eval_e2e_ap_peng.py --conf_threshold 0 --model "crowd=$FT/best.pt,$KH" --out_dir "outputs/e2e_ap_crowd_${G}_hrnet_t0"
for DET in yolo rtmo; do
    python eval_e2e_ap_peng.py --conf_threshold 0 --cache_dir "outputs/e2e_ap_$DET/cache" \
        --model "crowd=$FT/best.pt,$KH" --out_dir "outputs/e2e_ap_crowd_${G}_${DET}_t0"
done
echo "CROWD_DONE $FT $KH"
