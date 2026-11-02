#!/usr/bin/env bash
# PEng AP CIs for experiments 3-4 (2026-10-27): analysis/ap_bootstrap.py (5000 draws, seed 0, BCa + percentile) at
# threshold 0, CPU only, as run_ap_ci.sh / run_ap_ci_seedrep.sh. Usage: run_ap_ci_exp34.sh <tag> <guid>, tag = scale | jimg:
#   <tag>_{lb,clamp}: <tag> <guid> + its K-head - d4422c37 + 88389153 (verify_20261026/ap), K_lb THA / clamp THA
# Output outputs/ap_ci/<det>_t0/<tag>_{lb,clamp}.json (+ .log). Launch via systemd-run --user.
cd "$(dirname "$0")/.."
set -x
TAG=$1; G=$2
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
for det in hrnet yolo rtmo; do
    V=outputs/verify_20261026/ap/${det}_t0
    E=outputs/e2e_ap_${TAG}_${G}_${det}_t0
    O=outputs/ap_ci/${det}_t0
    mkdir -p $O
    for row in lb clamp; do
        python -m analysis.ap_bootstrap --a $E/results_${TAG}__${row}_tha.json --b $V/results_crowd__${row}_tha.json \
            --subset all 1-5 ge6 --out $O/${TAG}_$row.json > $O/${TAG}_$row.log 2>&1 &
    done
done
wait
echo "APCI_${TAG}_DONE"
