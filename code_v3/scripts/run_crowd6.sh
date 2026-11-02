#!/usr/bin/env bash
# PEng crowd emphasis (2026-10-29): crowd over-sampling 6x instead of 3x (finetune_pg_gat_hier_img_crowd6_coco.yaml,
# the only change), two runs (S1, S2) judged 2x2 against the two 3x runs (C1 d4422c37, C2 b2cdc656); pre-fixed
# criterion in results.md § "Crowd emphasis".
#   S1 = finetune_pg_gat_hier_img_crowd6_coco      + its augmented K-head
#   S2 = finetune_pg_gat_hier_img_crowd6_coco_rep  (same config, own folder via FT_NAME) + its augmented K-head
# Launch via systemd-run --user (unit peng-crowd6):
#   1. GPU, sequential: S1 fine-tune (run_pg_gat_hier.sh: + GPU THA evals, K_lb gap table) + K-head, then S2 the same
#   2. CPU (OMP_NUM_THREADS=4, seed 0), after the GPU part so CPU evals do not slow the fine-tunes: COCO GT /
#      synthetic THA; PGA gap tables on HigherHRNet / YOLO / RTMO; error anatomy (C1, C2, S1, S2)
#   3. CPU: AP@OKS at threshold 0 on the three all-image caches; AP-gap anatomy (native vs K_lb THA)
#   4. paired PGA CIs (all scenes, K_gt >= 6) and AP CIs (analysis/ap_bootstrap.py) for the four S - C pairs, the
#      S2 - S1 draw, and the gap to native
cd "$(dirname "$0")/.."
set -x
declare -A S
for tag in s1:finetune_pg_gat_hier_img_crowd6_coco s2:finetune_pg_gat_hier_img_crowd6_coco_rep; do
    IFS=: read -r t FTN <<< "$tag"
    FT_NAME=$FTN EVAL_SYN=0 SYN_DIR=outputs/train_pg_gat_hier_aug/a0c59235 SYN_CFG=train_pg_gat_hier_aug \
        FT_CFG=finetune_pg_gat_hier_img_crowd6_coco bash scripts/run_pg_gat_hier.sh || exit 1
    FT=outputs/$FTN/$(readlink outputs/$FTN/latest)
    G=$(basename "$FT")
    python train_k_head_aug.py --checkpoint "$FT/best.pt" --name "k_head_aug_$G" --epochs 10 || exit 1
    KH=outputs/k_head_aug_$G/$(readlink "outputs/k_head_aug_$G/latest")/best.pt
    S[$t]="$FT/best.pt,$KH"; S[${t}_g]=$G
    echo "GPU_DONE $t $FT $KH"
done
G1=${S[s1_g]}; G2=${S[s2_g]}

export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4   # = verify_all.py; KMeans labels depend on the thread count
declare -A ARR=([hrnet]=outputs/e2e_per_image/arrays [yolo]=outputs/e2e_cache_yolo/arrays [rtmo]=outputs/e2e_cache_rtmo/arrays)
declare -A APC=([hrnet]=outputs/e2e_ap/hrnet_cache [yolo]=outputs/e2e_ap_yolo/cache [rtmo]=outputs/e2e_ap_rtmo/cache)
declare -A PFX=([hrnet]="" [yolo]=yolo_ [rtmo]=rtmo_)
CROWD=outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt,outputs/k_head_aug_d4422c37/88389153/best.pt
CROWDR=outputs/finetune_pg_gat_hier_img_crowd_coco_rep/b2cdc656/best.pt,outputs/k_head_aug_b2cdc656/20adeeb3/best.pt
D=outputs/diagnostics
THA="import numpy,torch,sys,runpy; numpy.random.seed(0); torch.manual_seed(0); sys.argv=['eval_hungarian_grouping.py']+sys.argv[1:]; runpy.run_path('eval_hungarian_grouping.py', run_name='__main__')"
for t in s1 s2; do
    G=${S[${t}_g]}; CK=${S[$t]%%,*}; KH=${S[$t]##*,}
    T=outputs/tha_cpu/$G
    mkdir -p $T && ln -sfn "$(realpath $CK)" $T/best.pt
    python -c "$THA" --checkpoint $T/best.pt --device cpu --wandb_run_id none --virtual_dir data/virtual --split test > $T/virtual.log 2>&1 &
    python -c "$THA" --checkpoint $T/best.pt --device cpu --wandb_run_id none --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json > $T/coco.log 2>&1 &
    for det in hrnet yolo rtmo; do
        python eval_peng_gap.py --checkpoint "$CK" --k_head "$KH" --arrays ${ARR[$det]} \
            --out_dir outputs/peng_gap/${PFX[$det]}${G}_khaug_cpu --device cpu --seed 0 \
            > outputs/peng_gap_${det}_${G}_khaug_cpu.log 2>&1 &
    done
done
for det in hrnet yolo rtmo; do
    python -m analysis.error_anatomy --arrays ${ARR[$det]} --det $det --model crowd=$CROWD --model crowdrep=$CROWDR \
        --model crowd6=${S[s1]} --model crowd6rep=${S[s2]} \
        --out_dir $D/anatomy_crowd6_$det > $D/anatomy_crowd6_$det.log 2>&1 &
done
wait
for t in s1 s2; do
    G=${S[${t}_g]}; NM=$([ $t = s1 ] && echo crowd6 || echo crowd6rep)
    for det in hrnet yolo rtmo; do
        python eval_e2e_ap_peng.py --cache_dir ${APC[$det]} --conf_threshold 0 --model "$NM=${S[$t]}" \
            --out_dir outputs/e2e_ap_${NM}_${G}_${det}_t0 --device cpu --seed 0 \
            > outputs/e2e_ap_${NM}_${G}_${det}_t0.log 2>&1 &
    done
done
wait

# paired PGA CIs: A - B; S = 6x arm, C = 3x arm; draw = S2 - S1
for det in hrnet yolo rtmo; do
    C1=outputs/verify_20261026/gap/${det}_d4422c37_khaug/per_image.jsonl
    C2=outputs/peng_gap/${PFX[$det]}b2cdc656_khaug_cpu/per_image.jsonl
    P=outputs/peng_gap/${PFX[$det]}${G2}_khaug_cpu
    S1=outputs/peng_gap/${PFX[$det]}${G1}_khaug_cpu/per_image.jsonl; S2=$P/per_image.jsonl
    for pr in "crowd6_draw:$S2:$S1" "s1_c1:$S1:$C1" "s1_c2:$S1:$C2" "s2_c1:$S2:$C1" "s2_c2:$S2:$C2"; do
        IFS=: read -r name A B <<< "$pr"
        python -m analysis.paired_ci --a $A --b $B --mode diff --rows clamp:tha clamp:copkm lb:tha eff:tha \
            --out $P/ci_$name.json
        python -m analysis.paired_ci --a $A --b $B --mode diff --min_k 6 --rows clamp:tha lb:tha eff:tha \
            --out $P/ci_${name}_k6.json
    done
    for s in $S1 $S2; do
        python -m analysis.paired_ci --a $s --mode gap --rows clamp:tha clamp:copkm eff:tha --out $(dirname $s)/ci_gap.json
    done
done

# AP CIs (BCa + percentile, 5000 draws, seed 0), K_lb THA and clamp THA; AP-gap anatomy on K_lb THA
export OMP_NUM_THREADS=1
for det in hrnet yolo rtmo; do
    V=outputs/verify_20261026/ap/${det}_t0
    O=outputs/ap_ci/${det}_t0
    mkdir -p $O outputs/diagnostics/ap_gap
    E1=outputs/e2e_ap_crowd6_${G1}_${det}_t0; E2=outputs/e2e_ap_crowd6rep_${G2}_${det}_t0
    for row in lb clamp; do
        C1=$V/results_crowd__${row}_tha.json; C2=outputs/e2e_ap_rep_b2cdc656_${det}_t0/results_rep__${row}_tha.json
        S1=$E1/results_crowd6__${row}_tha.json; S2=$E2/results_crowd6rep__${row}_tha.json
        for pr in "crowd6_draw:$S2:$S1" "s1_c1:$S1:$C1" "s1_c2:$S1:$C2" "s2_c1:$S2:$C1" "s2_c2:$S2:$C2" \
                  "gap_s1:$V/results_native_pooled.json:$S1" "gap_s2:$V/results_native_pooled.json:$S2"; do
            IFS=: read -r name A B <<< "$pr"
            python -m analysis.ap_bootstrap --a $A --b $B --subset all 1-5 ge6 \
                --out $O/crowd6_${name}_$row.json > $O/crowd6_${name}_$row.log 2>&1 &
        done
    done
    for t in s1 s2; do
        E=$([ $t = s1 ] && echo $E1/results_crowd6__lb_tha.json || echo $E2/results_crowd6rep__lb_tha.json)
        python -m analysis.ap_gap_anatomy --cache_dir ${APC[$det]} --a $V/results_native_pooled.json --b $E \
            --out outputs/diagnostics/ap_gap/${det}_crowd6${t}_lb.json > outputs/diagnostics/ap_gap/${det}_crowd6${t}_lb.log 2>&1 &
    done
    wait
done
echo "CROWD6_DONE ${S[s1]} ${S[s2]}"
