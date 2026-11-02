#!/usr/bin/env bash
# PEng confidence-aware THA (2026-10-29): paired AP CIs (analysis/ap_bootstrap.py, BCa + percentile, 5000 draws, seed 0)
# on the cross-fitted shared-lam results (analysis/tha_conf_cv.py, results_<row>_cvshared.json), both crowd runs, t0:
#   conf_<row>      THA-conf - THA (lam 0), same row          (the effect of the confidence term)
#   gap_conf_<row>  native_pooled - THA-conf                  (remaining AP gap)
#   clampconf_lb    clamp THA-conf - K_lb THA (lam 0)         (does the autonomous row reach the K_lb row?)
# Launch via systemd-run --user (unit peng-apci-thaconf); out outputs/ap_ci/<det>_t0/thaconf_<model>_<name>.json.
cd "$(dirname "$0")/.."
set -x
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
for det in hrnet yolo rtmo; do
    N=outputs/verify_20261026/ap/${det}_t0/results_native_pooled.json
    O=outputs/ap_ci/${det}_t0
    mkdir -p $O
    for m in crowd rep; do
        T=outputs/tha_conf/${m}_${det}_t0
        for pr in "conf_lb:$T/results_lb_cvshared.json:$T/results_lb_lam0.json" \
                  "conf_clamp:$T/results_clamp_cvshared.json:$T/results_clamp_lam0.json" \
                  "gap_conf_lb:$N:$T/results_lb_cvshared.json" \
                  "gap_conf_clamp:$N:$T/results_clamp_cvshared.json" \
                  "clampconf_lb:$T/results_clamp_cvshared.json:$T/results_lb_lam0.json"; do
            IFS=: read -r name A B <<< "$pr"
            python -m analysis.ap_bootstrap --a $A --b $B --subset all 1-5 ge6 \
                --out $O/thaconf_${m}_$name.json > $O/thaconf_${m}_$name.log 2>&1 &
        done
    done
    wait
done
echo APCI_THACONF_DONE
