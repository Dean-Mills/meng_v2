#!/usr/bin/env bash
# PEng AP CIs for the seed repeat (2026-10-27): analysis/ap_bootstrap.py (5000 draws, seed 0, BCa + percentile) at
# threshold 0, CPU only, as run_ap_ci.sh:
#   rep_{lb,clamp}:  rep b2cdc656 + 20adeeb3 - d4422c37 + 88389153 (verify_20261026/ap), K_lb THA / clamp THA
#   khrep_clamp:     d4422c37 + second K-head 4fb1a9b5 - d4422c37 + 88389153, clamp THA
#   gap_rep_lb:      native_full - rep K_lb THA (the AP gap with the repeat)
# Output outputs/ap_ci/<det>_t0/<name>.json (+ .log). Launch via systemd-run --user (unit peng-apci-rep).
cd "$(dirname "$0")/.."
set -x
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
for det in hrnet yolo rtmo; do
    V=outputs/verify_20261026/ap/${det}_t0
    R=outputs/e2e_ap_rep_b2cdc656_${det}_t0
    K=outputs/e2e_ap_khrep_d4422c37_${det}_t0
    O=outputs/ap_ci/${det}_t0
    mkdir -p $O
    pair() { python -m analysis.ap_bootstrap --a "$2" --b "$3" --subset all 1-5 ge6 --out $O/$1.json > $O/$1.log 2>&1 & }
    for row in lb clamp; do
        pair rep_$row $R/results_rep__${row}_tha.json $V/results_crowd__${row}_tha.json
    done
    pair khrep_clamp $K/results_khrep__clamp_tha.json $V/results_crowd__clamp_tha.json
    pair gap_rep_lb $V/results_native_full.json $R/results_rep__lb_tha.json
done
wait
echo "APCIREP_DONE"
