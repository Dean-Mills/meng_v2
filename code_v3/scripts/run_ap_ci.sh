#!/usr/bin/env bash
# PEng AP CIs (2026-10-26): paired image-bootstrap BCa CIs (analysis/ap_bootstrap.py, 5000 draws, seed 0) on the
# CPU-verified AP@OKS results at threshold 0 (outputs/verify_20261026/ap/<det>_t0, the canonical numbers), CPU only.
#   gap_<model>_<row>: native_full - <model>/<row> (the AP gap) for the arc models headline (MEng), hier_aug,
#                      peng (+ image evidence), crowd (d4422c37, best), rows lb_tha / clamp_tha (+ MEng pred_knn)
#   exp1_clamp:        k_head_hist_d4422c37 clamp_tha - crowd clamp_tha
#   exp2_{lb,clamp}:   same-type edges 37de5701 - crowd, lb_tha / clamp_tha
# Subsets all / 1-5 / ge6. Output outputs/ap_ci/<det>_t0/<name>.json (+ .log). Launch via systemd-run --user
# (unit peng-apci); safe beside a GPU job (no CUDA).
cd "$(dirname "$0")/.."
set -x
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
for det in hrnet yolo rtmo; do
    V=outputs/verify_20261026/ap/${det}_t0
    O=outputs/ap_ci/${det}_t0
    mkdir -p $O
    pair() { python -m analysis.ap_bootstrap --a "$2" --b "$3" --subset all 1-5 ge6 --out $O/$1.json > $O/$1.log 2>&1 & }
    for m in headline hier_aug peng crowd; do
        for row in lb_tha clamp_tha; do
            pair gap_${m}_${row} $V/results_native_full.json $V/results_${m}__${row}.json
        done
    done
    pair gap_headline_pred_knn $V/results_native_full.json $V/results_headline__pred_knn.json
    pair exp1_clamp outputs/e2e_ap_khhist_${det}_t0/results_khhist__clamp_tha.json $V/results_crowd__clamp_tha.json
    for row in lb clamp; do
        pair exp2_$row outputs/e2e_ap_stedges_37de5701_${det}_t0/results_stedges__${row}_tha.json $V/results_crowd__${row}_tha.json
    done
    wait
done
echo "APCI_DONE"
