#!/usr/bin/env bash
# PEng AP-gap anatomy (2026-10-29, CPU only): native_pooled vs ours (K_lb THA, AP verdict row) at threshold 0 on the
# three all-image caches, for both crowd runs (d4422c37 = verify_20261026 files, b2cdc656 = seed-repeat files) so the
# decomposition is checked against a second training run. analysis/ap_gap_anatomy.py; out outputs/diagnostics/ap_gap/.
cd "$(dirname "$0")/.."
set -x
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
declare -A APC=([hrnet]=outputs/e2e_ap/hrnet_cache [yolo]=outputs/e2e_ap_yolo/cache [rtmo]=outputs/e2e_ap_rtmo/cache)
O=outputs/diagnostics/ap_gap
mkdir -p $O
for det in hrnet yolo rtmo; do
    V=outputs/verify_20261026/ap/${det}_t0
    for m in "crowd:$V/results_crowd__lb_tha.json" "rep:outputs/e2e_ap_rep_b2cdc656_${det}_t0/results_rep__lb_tha.json"; do
        IFS=: read -r name B <<< "$m"
        python -m analysis.ap_gap_anatomy --cache_dir ${APC[$det]} --a $V/results_native_pooled.json --b $B \
            --out $O/${det}_${name}_lb.json > $O/${det}_${name}_lb.log 2>&1 &
    done
done
wait
echo AP_GAP_DONE

# threshold 0.1 (joints at or below 0.1 removed), d4422c37 K_lb THA from the K-head-hist run (K_lb rows do not use the K-head)
for det in hrnet yolo rtmo; do
    E=outputs/e2e_ap_khhist_${det}_t0.1
    python -m analysis.ap_gap_anatomy --cache_dir ${APC[$det]} --conf_threshold 0.1 --a $E/results_native_pooled.json \
        --b $E/results_khhist__lb_tha.json --out $O/${det}_crowd_lb_t01.json > $O/${det}_crowd_lb_t01.log 2>&1 &
done
wait
echo AP_GAP_T01_DONE
