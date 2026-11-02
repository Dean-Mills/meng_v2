#!/usr/bin/env bash
# PEng experiment 2 (2026-10-26): same-type edges in the joint graph (pg_gat_hier.same_type_edges).
# A/B arm: finetune_pg_gat_hier_img_crowd_coco/d4422c37 + k_head_aug_d4422c37/88389153, canonical CPU
# numbers in outputs/verify_20261026/. Launch via systemd-run --user (unit peng-stedges):
#   0. wait until experiment 1 (unit peng-khist) has released the GPU
#   1. GPU: fine-tune finetune_pg_gat_hier_img_crowd_stedges_coco from train_pg_gat_hier_aug/a0c59235
#      (run_pg_gat_hier.sh: + GPU THA evals and a K_lb gap table), then the augmented K-head (same recipe)
#   2. CPU, seed 0: PGA gap tables on HigherHRNet / YOLO / RTMO + error anatomy (best vs stedges)
#   3. CPU, seed 0: AP@OKS at threshold 0 on the three all-image caches
#   4. paired CIs against the verified d4422c37 tables (all scenes, >= 6 and >= 9 people)
cd "$(dirname "$0")/.."
set -x
while systemctl --user is-active --quiet peng-khist && ! grep -q GPU_DONE outputs/run_k_hist.log; do sleep 60; done
EVAL_SYN=0 SYN_DIR=outputs/train_pg_gat_hier_aug/a0c59235 SYN_CFG=train_pg_gat_hier_aug \
    FT_CFG=finetune_pg_gat_hier_img_crowd_stedges_coco bash scripts/run_pg_gat_hier.sh || exit 1
FT=outputs/finetune_pg_gat_hier_img_crowd_stedges_coco/$(readlink outputs/finetune_pg_gat_hier_img_crowd_stedges_coco/latest)
G=$(basename "$FT")
python train_k_head_aug.py --checkpoint "$FT/best.pt" --name "k_head_aug_$G" --epochs 10 || exit 1
KH=outputs/k_head_aug_$G/$(readlink "outputs/k_head_aug_$G/latest")/best.pt
echo "GPU_DONE $FT $KH"

export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
BEST=outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt,outputs/k_head_aug_d4422c37/88389153/best.pt
declare -A ARR=([hrnet]=outputs/e2e_per_image/arrays [yolo]=outputs/e2e_cache_yolo/arrays [rtmo]=outputs/e2e_cache_rtmo/arrays)
declare -A APC=([hrnet]=outputs/e2e_ap/hrnet_cache [yolo]=outputs/e2e_ap_yolo/cache [rtmo]=outputs/e2e_ap_rtmo/cache)
declare -A PFX=([hrnet]="" [yolo]=yolo_ [rtmo]=rtmo_)
D=outputs/diagnostics
for det in hrnet yolo rtmo; do
    python eval_peng_gap.py --checkpoint "$FT/best.pt" --k_head "$KH" --arrays ${ARR[$det]} \
        --out_dir outputs/peng_gap/${PFX[$det]}${G}_khaug_cpu --device cpu --seed 0 \
        > outputs/peng_gap_${det}_${G}_khaug_cpu.log 2>&1 &
    python -m analysis.error_anatomy --arrays ${ARR[$det]} --det $det --model best=$BEST \
        --model stedges=$FT/best.pt,$KH --out_dir $D/anatomy_stedges_$det > $D/anatomy_stedges_$det.log 2>&1 &
done
wait
for det in hrnet yolo rtmo; do
    python eval_e2e_ap_peng.py --cache_dir ${APC[$det]} --conf_threshold 0 --model "stedges=$FT/best.pt,$KH" \
        --out_dir outputs/e2e_ap_stedges_${G}_${det}_t0 --device cpu --seed 0 \
        > outputs/e2e_ap_stedges_${G}_${det}_t0.log 2>&1 &
done
wait
for det in hrnet yolo rtmo; do
    P=outputs/peng_gap/${PFX[$det]}${G}_khaug_cpu
    B=outputs/verify_20261026/gap/${det}_d4422c37_khaug/per_image.jsonl
    python -m analysis.paired_ci --a $P/per_image.jsonl --b $B --mode diff \
        --rows clamp:tha clamp:copkm lb:tha eff:tha --out $P/ci_vs_d4422c37.json
    for k in 6 9; do
        python -m analysis.paired_ci --a $P/per_image.jsonl --b $B --mode diff --min_k $k \
            --rows clamp:tha eff:tha --out $P/ci_vs_d4422c37_k$k.json
    done
    python -m analysis.paired_ci --a $P/per_image.jsonl --mode gap --rows clamp:tha clamp:copkm eff:tha --out $P/ci_gap.json
done
echo "STEDGES_DONE $FT $KH"
