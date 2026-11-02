#!/usr/bin/env bash
# PEng confidence-aware THA (2026-10-29, CPU only): eval_tha_conf_ap.py lam grid on the three all-image caches at t0,
# K_lb and clamp rows, for both crowd runs (d4422c37 + 88389153, b2cdc656 + 20adeeb3). lam 0 is checked against the
# verified eval_e2e_ap_peng files. lam is then chosen by 2-fold cross-fitting (analysis/tha_conf_cv.py).
# Launch via systemd-run --user (unit peng-thaconf); out outputs/tha_conf/<model>_<det>_t0/.
cd "$(dirname "$0")/.."
set -x
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4   # = verify_all.py; KMeans labels depend on the thread count
declare -A APC=([hrnet]=outputs/e2e_ap/hrnet_cache [yolo]=outputs/e2e_ap_yolo/cache [rtmo]=outputs/e2e_ap_rtmo/cache)
CROWD=outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt,outputs/k_head_aug_d4422c37/88389153/best.pt
CROWDR=outputs/finetune_pg_gat_hier_img_crowd_coco_rep/b2cdc656/best.pt,outputs/k_head_aug_b2cdc656/20adeeb3/best.pt
for det in ${DETS:-hrnet yolo rtmo}; do
    python eval_tha_conf_ap.py --model crowd=$CROWD --cache_dir ${APC[$det]} --conf_threshold 0 \
        --out_dir outputs/tha_conf/crowd_${det}_t0 --check_dir outputs/verify_20261026/ap/${det}_t0 \
        > outputs/tha_conf_crowd_${det}_t0.log 2>&1 &
    python eval_tha_conf_ap.py --model rep=$CROWDR --cache_dir ${APC[$det]} --conf_threshold 0 \
        --out_dir outputs/tha_conf/rep_${det}_t0 --check_dir outputs/e2e_ap_rep_b2cdc656_${det}_t0 \
        > outputs/tha_conf_rep_${det}_t0.log 2>&1 &
done
wait
for det in ${DETS:-hrnet yolo rtmo}; do for m in crowd rep; do grep -q "check: lam 0 equals" outputs/tha_conf_${m}_${det}_t0.log || echo "CHECK FAILED: ${m} ${det}"; done; done
echo THACONF_DONE
