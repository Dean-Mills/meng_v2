#!/usr/bin/env bash
# PEng no-crowd repeat (2026-10-27): a second run of the 372658c6 recipe (image evidence, no crowd over-sampling) so
# crowd over-sampling is judged on 2 crowd runs (d4422c37, b2cdc656) x 2 no-crowd runs (372658c6, this one) instead of
# one pair (its single-run gain +0.0013 / +0.0017 / +0.0005 PGA autonomous is <= 2x one training-noise draw).
#   nocrowdrep = finetune_pg_gat_hier_img_coco recipe again (own folder via FT_NAME, so
#                finetune_pg_gat_hier_img_coco/latest stays 372658c6) + its augmented K-head
# Launch via systemd-run --user (unit peng-nocrowdrep):
#   1. GPU: fine-tune (run_pg_gat_hier.sh: + GPU THA evals and a K_lb gap table), K-head on it
#   2. CPU, seed 0: COCO GT / synthetic THA evals; PGA gap tables on HigherHRNet / YOLO / RTMO + error anatomy
#      (crowd d4422c37 vs crowd rep b2cdc656 vs no-crowd 372658c6 vs no-crowd rep)
#   3. CPU, seed 0: AP@OKS at threshold 0 on the three all-image caches
#   4. paired PGA CIs (all scenes, >= 6 people) and AP CIs (analysis/ap_bootstrap.py) for the four crowd - no-crowd
#      pairs and the no-crowd run-to-run pair
cd "$(dirname "$0")/.."
set -x
FT_NAME=finetune_pg_gat_hier_img_coco_rep EVAL_SYN=0 SYN_DIR=outputs/train_pg_gat_hier_aug/a0c59235 \
    SYN_CFG=train_pg_gat_hier_aug FT_CFG=finetune_pg_gat_hier_img_coco bash scripts/run_pg_gat_hier.sh || exit 1
FT=outputs/finetune_pg_gat_hier_img_coco_rep/$(readlink outputs/finetune_pg_gat_hier_img_coco_rep/latest)
G=$(basename "$FT")
python train_k_head_aug.py --checkpoint "$FT/best.pt" --name "k_head_aug_$G" --epochs 10 || exit 1
KH=outputs/k_head_aug_$G/$(readlink "outputs/k_head_aug_$G/latest")/best.pt
echo "GPU_DONE $FT $KH"

export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
declare -A ARR=([hrnet]=outputs/e2e_per_image/arrays [yolo]=outputs/e2e_cache_yolo/arrays [rtmo]=outputs/e2e_cache_rtmo/arrays)
declare -A APC=([hrnet]=outputs/e2e_ap/hrnet_cache [yolo]=outputs/e2e_ap_yolo/cache [rtmo]=outputs/e2e_ap_rtmo/cache)
declare -A PFX=([hrnet]="" [yolo]=yolo_ [rtmo]=rtmo_)
CROWD=outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt,outputs/k_head_aug_d4422c37/88389153/best.pt
CROWDR=outputs/finetune_pg_gat_hier_img_crowd_coco_rep/b2cdc656/best.pt,outputs/k_head_aug_b2cdc656/20adeeb3/best.pt
NOCROWD=outputs/finetune_pg_gat_hier_img_coco/372658c6/best.pt,outputs/k_head_aug_372658c6/4579ce0c/best.pt
D=outputs/diagnostics
T=outputs/tha_cpu/$G
mkdir -p $T && ln -sfn "$(realpath $FT/best.pt)" $T/best.pt
THA="import numpy,torch,sys,runpy; numpy.random.seed(0); torch.manual_seed(0); sys.argv=['eval_hungarian_grouping.py']+sys.argv[1:]; runpy.run_path('eval_hungarian_grouping.py', run_name='__main__')"
python -c "$THA" --checkpoint $T/best.pt --device cpu --wandb_run_id none --virtual_dir data/virtual --split test > $T/virtual.log 2>&1 &
python -c "$THA" --checkpoint $T/best.pt --device cpu --wandb_run_id none --coco_img_dir data/coco2017/val2017 \
    --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json > $T/coco.log 2>&1 &
for det in hrnet yolo rtmo; do
    python eval_peng_gap.py --checkpoint "$FT/best.pt" --k_head "$KH" --arrays ${ARR[$det]} \
        --out_dir outputs/peng_gap/${PFX[$det]}${G}_khaug_cpu --device cpu --seed 0 \
        > outputs/peng_gap_${det}_${G}_khaug_cpu.log 2>&1 &
    python -m analysis.error_anatomy --arrays ${ARR[$det]} --det $det --model crowd=$CROWD --model crowdrep=$CROWDR \
        --model nocrowd=$NOCROWD --model nocrowdrep=$FT/best.pt,$KH \
        --out_dir $D/anatomy_nocrowdrep_$det > $D/anatomy_nocrowdrep_$det.log 2>&1 &
done
wait
for det in hrnet yolo rtmo; do
    python eval_e2e_ap_peng.py --cache_dir ${APC[$det]} --conf_threshold 0 --model "nocrowdrep=$FT/best.pt,$KH" \
        --out_dir outputs/e2e_ap_nocrowdrep_${G}_${det}_t0 --device cpu --seed 0 \
        > outputs/e2e_ap_nocrowdrep_${G}_${det}_t0.log 2>&1 &
done
wait

# paired PGA CIs: A - B, A = crowd arm, B = no-crowd arm (and no-crowd rep - no-crowd = the no-crowd draw)
for det in hrnet yolo rtmo; do
    V=outputs/verify_20261026/gap
    C1=$V/${det}_d4422c37_khaug/per_image.jsonl
    C2=outputs/peng_gap/${PFX[$det]}b2cdc656_khaug_cpu/per_image.jsonl
    N1=$V/${det}_372658c6_khaug/per_image.jsonl
    P=outputs/peng_gap/${PFX[$det]}${G}_khaug_cpu
    N2=$P/per_image.jsonl
    for pr in "nocrowd_draw:$N2:$N1" "crowd1_nocrowd2:$C1:$N2" "crowd2_nocrowd2:$C2:$N2" "crowd2_nocrowd1:$C2:$N1"; do
        IFS=: read -r name A B <<< "$pr"
        python -m analysis.paired_ci --a $A --b $B --mode diff --rows clamp:tha clamp:copkm lb:tha eff:tha \
            --out $P/ci_$name.json
        python -m analysis.paired_ci --a $A --b $B --mode diff --min_k 6 --rows clamp:tha lb:tha eff:tha \
            --out $P/ci_${name}_k6.json
    done
    python -m analysis.paired_ci --a $N2 --mode gap --rows clamp:tha clamp:copkm eff:tha --out $P/ci_gap.json
done

# AP CIs (BCa + percentile, 5000 draws, seed 0), same four pairs + the draw, K_lb THA and clamp THA
export OMP_NUM_THREADS=1
for det in hrnet yolo rtmo; do
    V=outputs/verify_20261026/ap/${det}_t0
    R=outputs/e2e_ap_rep_b2cdc656_${det}_t0
    N=outputs/e2e_ap_nocrowdrep_${G}_${det}_t0
    O=outputs/ap_ci/${det}_t0
    mkdir -p $O
    for row in lb clamp; do
        C1=$V/results_crowd__${row}_tha.json; C2=$R/results_rep__${row}_tha.json
        N1=$V/results_peng__${row}_tha.json; N2=$N/results_nocrowdrep__${row}_tha.json
        for pr in "nocrowd_draw:$N2:$N1" "crowd1_nocrowd2:$C1:$N2" "crowd2_nocrowd2:$C2:$N2" "crowd2_nocrowd1:$C2:$N1"; do
            IFS=: read -r name A B <<< "$pr"
            python -m analysis.ap_bootstrap --a $A --b $B --subset all 1-5 ge6 \
                --out $O/${name}_$row.json > $O/${name}_$row.log 2>&1 &
        done
    done
    wait
done
echo "NOCROWDREP_DONE $FT $KH"
