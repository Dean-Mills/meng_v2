#!/usr/bin/env bash
# PEng fine-tune seed repeat (2026-10-26): the noise floor for single-run A/Bs. The trainer sets no
# seed, so re-running the unchanged d4422c37 recipe (run_crowd.sh) draws a fresh one. Two controls:
#   rep   = finetune_pg_gat_hier_img_crowd_coco recipe again (own folder via FT_NAME, so
#           finetune_pg_gat_hier_img_crowd_coco/latest stays d4422c37) + its augmented K-head
#           -> model + K-head spread, the floor for experiments 2-4
#   khrep = a second augmented K-head on d4422c37 (k_head_aug_d4422c37_rep, so
#           k_head_aug_d4422c37/latest stays 88389153) -> K-head-only spread, the floor for experiment 1
# Launch via systemd-run --user (unit peng-seedrep):
#   1. GPU: fine-tune (run_pg_gat_hier.sh: + GPU THA evals and a K_lb gap table), K-head on it,
#      second K-head on d4422c37
#   2. CPU, seed 0: COCO GT / synthetic THA evals; PGA gap tables on HigherHRNet / YOLO / RTMO for rep
#      and khrep + error anatomy (best vs rep vs khrep)
#   3. CPU, seed 0: AP@OKS at threshold 0 on the three all-image caches (rep, khrep)
#   4. paired CIs: rep vs d4422c37 (all scenes, >= 6 and >= 9 people), khrep vs 88389153
cd "$(dirname "$0")/.."
set -x
FT_NAME=finetune_pg_gat_hier_img_crowd_coco_rep EVAL_SYN=0 SYN_DIR=outputs/train_pg_gat_hier_aug/a0c59235 \
    SYN_CFG=train_pg_gat_hier_aug FT_CFG=finetune_pg_gat_hier_img_crowd_coco bash scripts/run_pg_gat_hier.sh || exit 1
FT=outputs/finetune_pg_gat_hier_img_crowd_coco_rep/$(readlink outputs/finetune_pg_gat_hier_img_crowd_coco_rep/latest)
G=$(basename "$FT")
python train_k_head_aug.py --checkpoint "$FT/best.pt" --name "k_head_aug_$G" --epochs 10 || exit 1
KH=outputs/k_head_aug_$G/$(readlink "outputs/k_head_aug_$G/latest")/best.pt
BASE=outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt
BASE_KH=outputs/k_head_aug_d4422c37/88389153/best.pt
python train_k_head_aug.py --checkpoint "$BASE" --name k_head_aug_d4422c37_rep --epochs 10 || exit 1
KR=outputs/k_head_aug_d4422c37_rep/$(readlink outputs/k_head_aug_d4422c37_rep/latest)/best.pt
echo "GPU_DONE $FT $KH $KR"

export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
declare -A ARR=([hrnet]=outputs/e2e_per_image/arrays [yolo]=outputs/e2e_cache_yolo/arrays [rtmo]=outputs/e2e_cache_rtmo/arrays)
declare -A APC=([hrnet]=outputs/e2e_ap/hrnet_cache [yolo]=outputs/e2e_ap_yolo/cache [rtmo]=outputs/e2e_ap_rtmo/cache)
declare -A PFX=([hrnet]="" [yolo]=yolo_ [rtmo]=rtmo_)
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
    python -m analysis.error_anatomy --arrays ${ARR[$det]} --det $det --model best=$BASE,$BASE_KH \
        --model rep=$FT/best.pt,$KH --model khrep=$BASE,$KR \
        --out_dir $D/anatomy_seedrep_$det > $D/anatomy_seedrep_$det.log 2>&1 &
done
wait
for det in hrnet yolo rtmo; do
    python eval_peng_gap.py --checkpoint "$BASE" --k_head "$KR" --arrays ${ARR[$det]} \
        --out_dir outputs/peng_gap/${PFX[$det]}d4422c37_khrep_cpu --device cpu --seed 0 \
        > outputs/peng_gap_${det}_d4422c37_khrep_cpu.log 2>&1 &
    python eval_e2e_ap_peng.py --cache_dir ${APC[$det]} --conf_threshold 0 --model "rep=$FT/best.pt,$KH" \
        --out_dir outputs/e2e_ap_rep_${G}_${det}_t0 --device cpu --seed 0 \
        > outputs/e2e_ap_rep_${G}_${det}_t0.log 2>&1 &
done
wait
for det in hrnet yolo rtmo; do
    python eval_e2e_ap_peng.py --cache_dir ${APC[$det]} --conf_threshold 0 --model "khrep=$BASE,$KR" \
        --out_dir outputs/e2e_ap_khrep_d4422c37_${det}_t0 --device cpu --seed 0 \
        > outputs/e2e_ap_khrep_d4422c37_${det}_t0.log 2>&1 &
done
wait
for det in hrnet yolo rtmo; do
    B=outputs/verify_20261026/gap/${det}_d4422c37_khaug/per_image.jsonl
    P=outputs/peng_gap/${PFX[$det]}${G}_khaug_cpu
    python -m analysis.paired_ci --a $P/per_image.jsonl --b $B --mode diff \
        --rows clamp:tha clamp:copkm lb:tha eff:tha --out $P/ci_vs_d4422c37.json
    for k in 6 9; do
        python -m analysis.paired_ci --a $P/per_image.jsonl --b $B --mode diff --min_k $k \
            --rows clamp:tha eff:tha --out $P/ci_vs_d4422c37_k$k.json
    done
    python -m analysis.paired_ci --a $P/per_image.jsonl --mode gap --rows clamp:tha clamp:copkm eff:tha --out $P/ci_gap.json
    R=outputs/peng_gap/${PFX[$det]}d4422c37_khrep_cpu
    python -m analysis.paired_ci --a $R/per_image.jsonl --b $B --mode diff \
        --rows clamp:tha clamp:copkm pred:tha pred:knn --out $R/ci_vs_88389153.json
    python -m analysis.paired_ci --a $R/per_image.jsonl --mode gap --rows clamp:tha clamp:copkm eff:tha --out $R/ci_gap.json
done
echo "SEEDREP_DONE $FT $KH $KR"
