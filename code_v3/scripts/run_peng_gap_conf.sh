#!/usr/bin/env bash
# PEng confidence-aware THA on the fixed PGA scene sets (2026-10-29, CPU only): eval_peng_gap_conf.py at lam
# 0 / 0.025 / 0.05 / 0.1 / 0.2 (0.2 = the clamp lam chosen by AP cross-fitting on every detector and run), both crowd
# runs, three detectors; tha_l0 checked against the verified eval_peng_gap files; then paired PGA CIs (all scenes,
# K_gt >= 6) of each lam against lam 0 on lb / clamp / eff, and the gap to native.
# Launch via systemd-run --user (unit peng-pgconf); out outputs/peng_gap_conf/<det>_<run>/.
cd "$(dirname "$0")/.."
set -x
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4   # = verify_all.py; KMeans labels depend on the thread count
declare -A ARR=([hrnet]=outputs/e2e_per_image/arrays [yolo]=outputs/e2e_cache_yolo/arrays [rtmo]=outputs/e2e_cache_rtmo/arrays)
declare -A APC=([hrnet]=outputs/e2e_ap/hrnet_cache [yolo]=outputs/e2e_ap_yolo/cache [rtmo]=outputs/e2e_ap_rtmo/cache)
declare -A PFX=([hrnet]="" [yolo]=yolo_ [rtmo]=rtmo_)
O=outputs/peng_gap_conf
mkdir -p $O
for det in hrnet yolo rtmo; do
    python eval_peng_gap_conf.py --checkpoint outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt \
        --k_head outputs/k_head_aug_d4422c37/88389153/best.pt --arrays ${ARR[$det]} --ap_cache ${APC[$det]} \
        --out_dir $O/${det}_d4422c37 --check outputs/verify_20261026/gap/${det}_d4422c37_khaug/per_image.jsonl \
        > $O/${det}_d4422c37.log 2>&1 &
    python eval_peng_gap_conf.py --checkpoint outputs/finetune_pg_gat_hier_img_crowd_coco_rep/b2cdc656/best.pt \
        --k_head outputs/k_head_aug_b2cdc656/20adeeb3/best.pt --arrays ${ARR[$det]} --ap_cache ${APC[$det]} \
        --out_dir $O/${det}_b2cdc656 --check outputs/peng_gap/${PFX[$det]}b2cdc656_khaug_cpu/per_image.jsonl \
        > $O/${det}_b2cdc656.log 2>&1 &
done
wait
for det in hrnet yolo rtmo; do for g in d4422c37 b2cdc656; do
    P=$O/${det}_$g/per_image.jsonl
    grep -q "check: tha_l0 equals" $O/${det}_$g.log || echo "CHECK FAILED: $det $g"
    for lam in 0.025 0.05 0.1 0.2; do
        rows="lb:tha_l$lam=lb:tha_l0 clamp:tha_l$lam=clamp:tha_l0 eff:tha_l$lam=eff:tha_l0"
        python -m analysis.paired_ci --a $P --b $P --mode diff --rows $rows --out $O/${det}_$g/ci_l$lam.json
        python -m analysis.paired_ci --a $P --b $P --mode diff --min_k 6 --rows $rows --out $O/${det}_$g/ci_l${lam}_k6.json
    done
    python -m analysis.paired_ci --a $P --mode gap --rows clamp:tha_l0 clamp:tha_l0.2 lb:tha_l0 lb:tha_l0.2 \
        --out $O/${det}_$g/ci_gap.json
done; done
echo PGCONF_DONE
