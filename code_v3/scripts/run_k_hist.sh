#!/usr/bin/env bash
# PEng experiment 1 (2026-10-26): K-head with the per-type count histogram (k_head_hist.KHistHead) on
# the frozen best backbone d4422c37. A/B arm: k_head_aug_d4422c37/88389153 (KEstimationHead, same recipe),
# canonical CPU numbers in outputs/verify_20261026/. Launch via systemd-run --user (unit peng-khist):
#   1. GPU: train_k_head_aug.py --head hist (10 epochs, held-out COCO-train selection)
#   2. CPU, seed 0: PGA gap tables on HigherHRNet / YOLO / RTMO + error anatomy (best vs khhist)
#   3. CPU, seed 0: AP@OKS at thresholds 0 and 0.1 on the three all-image caches
#   4. paired CIs against the verified 88389153 tables
cd "$(dirname "$0")/.."
set -x
FT=outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt
BASE_KH=outputs/k_head_aug_d4422c37/88389153/best.pt
python train_k_head_aug.py --checkpoint "$FT" --name k_head_hist_d4422c37 --head hist --epochs 10 || exit 1
KH=outputs/k_head_hist_d4422c37/$(readlink outputs/k_head_hist_d4422c37/latest)/best.pt
echo "GPU_DONE $KH"

export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
declare -A ARR=([hrnet]=outputs/e2e_per_image/arrays [yolo]=outputs/e2e_cache_yolo/arrays [rtmo]=outputs/e2e_cache_rtmo/arrays)
declare -A APC=([hrnet]=outputs/e2e_ap/hrnet_cache [yolo]=outputs/e2e_ap_yolo/cache [rtmo]=outputs/e2e_ap_rtmo/cache)
declare -A PFX=([hrnet]="" [yolo]=yolo_ [rtmo]=rtmo_)
D=outputs/diagnostics
for det in hrnet yolo rtmo; do
    python eval_peng_gap.py --checkpoint "$FT" --k_head "$KH" --arrays ${ARR[$det]} \
        --out_dir outputs/peng_gap/${PFX[$det]}d4422c37_khhist --device cpu --seed 0 \
        > outputs/peng_gap_${det}_d4422c37_khhist.log 2>&1 &
    python -m analysis.error_anatomy --arrays ${ARR[$det]} --det $det \
        --model best=$FT,$BASE_KH --model khhist=$FT,$KH \
        --out_dir $D/anatomy_khhist_$det > $D/anatomy_khhist_$det.log 2>&1 &
done
wait
for det in hrnet yolo rtmo; do
    for t in 0 0.1; do
        python eval_e2e_ap_peng.py --cache_dir ${APC[$det]} --conf_threshold $t --model "khhist=$FT,$KH" \
            --out_dir outputs/e2e_ap_khhist_${det}_t$t --device cpu --seed 0 \
            > outputs/e2e_ap_khhist_${det}_t$t.log 2>&1 &
    done
done
wait
for det in hrnet yolo rtmo; do
    G=outputs/peng_gap/${PFX[$det]}d4422c37_khhist
    python -m analysis.paired_ci --a $G/per_image.jsonl --b outputs/verify_20261026/gap/${det}_d4422c37_khaug/per_image.jsonl \
        --mode diff --rows clamp:tha clamp:copkm pred:tha pred:knn --out $G/ci_vs_88389153.json
    python -m analysis.paired_ci --a $G/per_image.jsonl --mode gap --rows clamp:tha clamp:copkm eff:tha --out $G/ci_gap.json
done
echo "KHIST_DONE $KH"
